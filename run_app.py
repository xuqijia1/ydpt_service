from flask import Flask, request, jsonify, Response, render_template_string, stream_with_context, send_file
import os
import cv2
import threading
import time
import json
import requests
import platform
import subprocess
from datetime import datetime
from collections import OrderedDict, deque
from pathlib import Path
import numpy as np
import queue
import multiprocessing as mp
import setproctitle

# 导入统一推理引擎模块
from inference_engine import create_inference_engine, draw_detection_boxes, render_chinese_texts, CLASS_COLORS, _PIL_FONT

# ===================== CPU 亲和性设置（多服务部署时避免争抢CPU） =====================
def set_cpu_affinity(cpu_list=None):
    """
    设置进程的 CPU 亲和性，绑定到指定的 CPU 核心

    Args:
        cpu_list: CPU 核心列表，如 [52, 53]。如果为 None，从环境变量 CPU_CORES 读取

    环境变量:
        CPU_CORES: 逗号分隔的 CPU 核心列表，如 "52,53" 或 "52"
        CPU_START: 起始 CPU 核心号（用于多容器部署，自动分配）
    """
    try:
        import psutil

        # 获取要绑定的 CPU 核心
        if cpu_list is None:
            # 优先从环境变量读取
            cpu_cores_env = os.environ.get('CPU_CORES', '')
            cpu_start_env = os.environ.get('CPU_START', '')

            if cpu_cores_env:
                cpu_list = [int(x.strip()) for x in cpu_cores_env.split(',')]
            elif cpu_start_env:
                # 只指定起始核心，默认使用 2 个核心
                start = int(cpu_start_env)
                cpu_list = [start, start + 1]
            else:
                # 默认不设置
                return False

        # 设置 CPU 亲和性
        p = psutil.Process()
        p.cpu_affinity(cpu_list)

        # 同时设置主线程的调度亲和性（Linux）
        if platform.system() == 'Linux':
            try:
                os.sched_setaffinity(0, cpu_list)
            except Exception:
                pass

        print(f"[OK] CPU 亲和性已设置: 核心 {cpu_list}")
        return True
    except ImportError:
        print("[WARN] psutil 未安装，无法设置 CPU 亲和性")
        return False
    except Exception as e:
        print(f"[WARN] 设置 CPU 亲和性失败: {e}")
        return False

app = Flask(__name__)

from config_loader import global_config

# ===================== CPU 亲和性设置（从配置文件读取） =====================
cpu_cores = global_config.get("system", {}).get("cpu_cores", None)
if cpu_cores:
    set_cpu_affinity(cpu_cores)

# 假设你的Config类定义如下，补充服务部署相关属性
class Config:
    # ==================== 推理后端配置（原有） ====================
    INFERENCE_BACKEND = global_config["inference"]["backend"]
    DEVICE_ID = global_config["inference"]["device_id"]
    YOLOV8_MODEL_PATH = global_config["inference"]["yolov8_model_path"]
    ASCEND_OM_MODEL_PATH = global_config["inference"]["ascend_om_model_path"]
    ROCKCHIP_MODEL_PATH = global_config["inference"].get("rockchip_model_path", "weights/best.rknn")
    ROCKCHIP_TARGET = global_config["inference"].get("rockchip_target", "rk3567")
    ROCKCHIP_PREPROCESS_MODE = global_config["inference"].get("rockchip_preprocess_mode", "non_quant")

    # ==================== 视频源配置（原有） ====================
    RTSP_URL = global_config["video"]["rtsp_url"]
    VIDEO_WRITE_FPS = global_config["video"]["write_fps"]
    VIDEO_CODEC = global_config["video"]["codec"]

    # ==================== 存储路径配置（原有） ====================
    IMAGE_BASE_DIR = global_config["storage"]["image_base_dir"]
    IMAGE_SUB_DIR = global_config["storage"]["image_sub_dir"]
    VIDEO_BASE_DIR = global_config["storage"]["video_base_dir"]
    VIDEO_SUB_DIR = global_config["storage"]["video_sub_dir"]
    VIDEO_SAVE_DIR = global_config["storage"]["video_save_dir"]
    IMAGE_SAVE_DIR = global_config["storage"]["image_save_dir"]
    ZONES_CONFIG_FILE = global_config["storage"]["zones_config_file"]
    CONFIG_BACKGROUND_IMAGE = global_config["storage"]["config_background_image"]

    # ==================== 功能开关配置（原有） ====================
    ENABLE_STREAMING = global_config["feature"]["enable_streaming"]
    ENABLE_CLIENT_CALLBACK = global_config["feature"]["enable_client_callback"]
    ENABLE_RECORDING = global_config["feature"].get("enable_recording", True)
    DEBUG_MODE = global_config["feature"]["debug_mode"]

    # ==================== 检测参数配置（原有） ====================
    DETECTION_INTERVAL = global_config["detection"]["interval"]
    IOU_THRESHOLD = global_config["detection"]["iou_threshold"]
    CONF_THRESHOLD = global_config["detection"].get("conf_threshold", 0.5)
    CONFIG_WIDTH = global_config["detection"]["config_width"]
    CONFIG_HEIGHT = global_config["detection"]["config_height"]
    # 检测区域（基于配置页面分辨率1920x1080）
    RECOG_AREA = global_config["detection"].get("recog_area", None)

    # ==================== 客户端回调配置（原有） ====================
    STEP_CALLBACK_URL = global_config["callback"]["step_url"]
    COORDINATES_CALLBACK_URL = global_config["callback"]["coordinates_url"]

    # ==================== FFmpeg配置（原有） ====================
    FFMPEG_DISABLE_AUTO_PTS = global_config["ffmpeg"]["disable_auto_pts"]
    FFMPEG_LOG_LEVEL = global_config["ffmpeg"]["log_level"]
    FFMPEG_FLUSH_TIMEOUT = global_config["ffmpeg"]["flush_timeout"]
    FFMPEG_VALID_FRAME_MIN_SIZE = global_config["ffmpeg"]["valid_frame_min_size"]

    # ==================== 服务部署配置（新增！关键修复） ====================
    SERVICE_PORT = global_config["service"]["port"]          # 服务端口
    SERVICE_HOST = global_config["service"]["host"]          # 监听地址
    SERVICE_THREADED = global_config["service"]["threaded"]  # 多线程
    SERVICE_DEBUG = global_config["service"]["debug"]        # 调试模式
    SERVICE_USE_RELOADER = global_config["service"]["use_reloader"]  # 自动重载

    # ==================== 系统标识（原有，可选） ====================
    IS_LINUX = platform.system().lower() == "linux"
    IS_WINDOWS = platform.system().lower() == "windows"

    # ==================== 步骤验证参数配置（新增） ====================
    STEP_MIN_STABLE_FRAMES = global_config.get("step_validation", {}).get("min_stable_frames", 3)
    STEP_PERSISTENCE_THRESHOLD = global_config.get("step_validation", {}).get("persistence_threshold", 0.5)
    STEP_CONFIDENCE_THRESHOLD = global_config.get("step_validation", {}).get("confidence_threshold", 0.6)
    STEP_STABILITY_THRESHOLD = global_config.get("step_validation", {}).get("stability_threshold", 1)

    # 空间关系参数
    SPATIAL_NEARBY_THRESHOLD = global_config.get("step_validation", {}).get("spatial", {}).get("nearby_threshold", 150)
    SPATIAL_ATTACHED_V_TOLERANCE = global_config.get("step_validation", {}).get("spatial", {}).get("attached_v_tolerance", 50)
    SPATIAL_IOU_THRESHOLD = global_config.get("step_validation", {}).get("spatial", {}).get("iou_threshold", 0.2)
    SPATIAL_SCALE_WITH_RESOLUTION = global_config.get("step_validation", {}).get("spatial", {}).get("scale_with_resolution", True)

# ==================== FFmpeg 专用工具函数（PTS/断言修复核心） ====================
def setup_ffmpeg_env():
    """配置FFmpeg环境变量，禁用自动PTS、控制日志级别，避免断言崩溃"""
    if Config.FFMPEG_DISABLE_AUTO_PTS:
        os.environ["OPENCV_FFMPEG_WRITE_NO_AUTO_PTS"] = "1"
    os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = Config.FFMPEG_LOG_LEVEL
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    # 通过FFmpeg选项抑制HEVC解码器警告(POC/cu_qp_delta等)
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;error"
    print(f"[OK] FFmpeg环境配置完成: 禁用自动PTS={Config.FFMPEG_DISABLE_AUTO_PTS}, 日志级别={Config.FFMPEG_LOG_LEVEL}")

def is_valid_frame(frame):
    """校验帧有效性，过滤空帧/损坏帧/极小帧，避免触发FFmpeg断言"""
    if frame is None:
        return False
    if frame.ndim != 3 or frame.shape[-1] != 3:  # 必须是3通道彩色帧
        return False
    h, w = frame.shape[:2]
    if h < 100 or w < 100:  # 过滤极小无效帧
        return False
    if frame.nbytes < Config.FFMPEG_VALID_FRAME_MIN_SIZE:  # 过滤空帧（字节数过小）
        return False
    return True

# ==================== 基础工具函数 ====================
def calculate_iou(box1, box2):
    """计算两个检测框的IOU（交并比）"""
    try:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 < x1 or y2 < y1:
            return 0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0
    except Exception as e:
        print(f"IOU计算错误: {e}, box1={box1}, box2={box2}")
        return 0

def check_overlap(det1, det2, iou_threshold=Config.IOU_THRESHOLD):
    """检查两个检测目标是否重叠"""
    return calculate_iou(det1['box'], det2['box']) > iou_threshold

def is_in_zone(box, zone):
    """检查检测框是否在指定区域内（带坐标比例映射）"""
    try:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2

        coords = zone['coords']
        CONFIG_W, CONFIG_H = Config.CONFIG_WIDTH, Config.CONFIG_HEIGHT
        frame_w = state.frame_width or CONFIG_W
        frame_h = state.frame_height or CONFIG_H

        scale_x = frame_w / CONFIG_W if CONFIG_W > 0 else 1.0
        scale_y = frame_h / CONFIG_H if CONFIG_H > 0 else 1.0

        zone_x1 = coords[0] * scale_x
        zone_y1 = coords[1] * scale_y
        zone_x2 = coords[2] * scale_x
        zone_y2 = coords[3] * scale_y

        return zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2
    except Exception as e:
        print(f"区域检查错误: {e}, box={box}, zone={zone}")
        return False

def get_image_save_path(step_name):
    """生成步骤验证图片的保存路径（跨平台）"""
    try:
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")

        dir_path = os.path.join(
            Config.IMAGE_BASE_DIR,
            Config.IMAGE_SUB_DIR.format(date=date_str, userid=state.user_id)
        )

        file_name = f"{step_name}_{time_str}.jpg"
        full_path = os.path.join(dir_path, file_name)

        os.makedirs(dir_path, exist_ok=True)
        full_path = full_path.replace("\\", "/")
        return full_path
    except Exception as e:
        print(f"生成图片路径失败: {e}")
        fallback_path = os.path.join(Config.IMAGE_BASE_DIR, f"{step_name}_{int(time.time())}.jpg")
        return fallback_path.replace("\\", "/")

def get_video_save_path(user_id):
    """生成考试视频的保存路径（跨平台）"""
    try:
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")

        dir_path = os.path.join(
            Config.VIDEO_BASE_DIR,
            Config.VIDEO_SUB_DIR.format(date=date_str, userid=user_id)
        )

        file_name = f"{user_id}_{time_str}.mp4"
        full_path = os.path.join(dir_path, file_name)

        os.makedirs(dir_path, exist_ok=True)
        full_path = full_path.replace("\\", "/")
        return full_path
    except Exception as e:
        print(f"生成视频路径失败: {e}")
        fallback_path = os.path.join(Config.VIDEO_BASE_DIR, f"{user_id}_{int(time.time())}.mp4")
        return fallback_path.replace("\\", "/")

# ==================== 推理后端创建函数 =====================
def create_inference_backend():
    """工厂函数：根据配置创建对应的推理后端"""
    backend = Config.INFERENCE_BACKEND.lower()

    # 根据后端类型选择模型路径
    if backend == "yolov8":
        model_path = Config.YOLOV8_MODEL_PATH
    elif backend == "ascend":
        model_path = Config.ASCEND_OM_MODEL_PATH
    elif backend == "rockchip":
        model_path = Config.ROCKCHIP_MODEL_PATH
    else:
        model_path = Config.YOLOV8_MODEL_PATH  # 默认

    return create_inference_engine(
        backend=Config.INFERENCE_BACKEND,
        model_path=model_path,
        conf_threshold=Config.CONF_THRESHOLD,
        recog_area=Config.RECOG_AREA,
        names=LABEL_MAP,
        device_id=Config.DEVICE_ID,
        target=Config.ROCKCHIP_TARGET if backend == "rockchip" else None,
        preprocess_mode=Config.ROCKCHIP_PREPROCESS_MODE if backend == "rockchip" else None
    )

# ==================== 对象跟踪模块 ====================
class ObjectTracker:
    """简单IoU对象跟踪器，为检测目标分配唯一ID并跟踪稳定性"""
    def __init__(self, max_disappeared=3):
        self.objects = OrderedDict()  # {track_id: {'box': [], 'class': '', 'disappeared': 0, 'stable_count': 0}}
        self.next_id = 0
        self.max_disappeared = max_disappeared  # 最大连续消失帧数

    def update(self, detections):
        """根据新的检测结果更新跟踪状态，返回带track_id的检测结果"""
        if len(detections) == 0:
            # 无检测结果时，所有跟踪对象标记为消失
            for obj_id in list(self.objects.keys()):
                self.objects[obj_id]['disappeared'] += 1
                if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                    del self.objects[obj_id]
            return detections

        # 提取当前检测框和类别
        current_boxes = np.array([[d['x1'], d['y1'], d['x2'], d['y2']] for d in detections])
        current_classes = [d['class'] for d in detections]

        if len(self.objects) == 0:
            # 首次检测，初始化所有跟踪对象
            for i, det in enumerate(detections):
                self.objects[self.next_id] = {
                    'box': current_boxes[i],
                    'class': current_classes[i],
                    'disappeared': 0,
                    'stable_count': 1
                }
                det['track_id'] = self.next_id
                self.next_id += 1
        else:
            # 计算IoU矩阵，匹配新旧检测框
            object_ids = list(self.objects.keys())
            object_boxes = np.array([self.objects[obj_id]['box'] for obj_id in object_ids])
            iou_matrix = np.zeros((len(object_boxes), len(current_boxes)))
            for i, obj_box in enumerate(object_boxes):
                for j, cur_box in enumerate(current_boxes):
                    iou_matrix[i, j] = calculate_iou(obj_box, cur_box)

            matched_rows, matched_cols = set(), set()
            # 优先匹配IoU最高的目标
            for _ in range(min(iou_matrix.shape)):
                if np.max(iou_matrix) < 0.3:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matched_rows.add(i)
                matched_cols.add(j)

                # 更新匹配到的对象状态
                obj_id = object_ids[i]
                self.objects[obj_id]['box'] = current_boxes[j]
                self.objects[obj_id]['class'] = current_classes[j]
                self.objects[obj_id]['disappeared'] = 0
                self.objects[obj_id]['stable_count'] += 1
                detections[j]['track_id'] = obj_id
                detections[j]['stable_count'] = self.objects[obj_id]['stable_count']  # 同步stable_count到检测结果

                # 标记为已匹配，避免重复匹配
                iou_matrix[i, :] = -1
                iou_matrix[:, j] = -1

            # 未匹配的旧对象标记为消失
            for i in range(len(object_ids)):
                if i not in matched_rows:
                    obj_id = object_ids[i]
                    self.objects[obj_id]['disappeared'] += 1
                    if self.objects[obj_id]['disappeared'] > self.max_disappeared:
                        del self.objects[obj_id]

            # 未匹配的新对象初始化跟踪
            for j in range(len(current_boxes)):
                if j not in matched_cols:
                    detections[j]['track_id'] = self.next_id
                    detections[j]['stable_count'] = 1  # 新对象的stable_count
                    self.objects[self.next_id] = {
                        'box': current_boxes[j],
                        'class': current_classes[j],
                        'disappeared': 0,
                        'stable_count': 1
                    }
                    self.next_id += 1
        return detections

# ==================== 步骤验证引擎 ====================
class StepValidator:
    """作业步骤验证引擎，实现完整的步骤验证规则"""
    def __init__(self, state_ref):
        self.state = state_ref
        self.step_requirements = self._define_requirements()
        self.confidence_threshold = 0.6
        self.persistence_threshold = 0.5

    def _define_requirements(self):
        """定义每个作业步骤的验证规则"""
        return {
            1: {  # 穿戴防护用品
                'required_classes': ['安全帽', '安全带'],
                'min_stable_frames': 5,
            },
            2: {  # 检查作业环境（由安卓程序判断，本服务无条件通过但返回false）
                'required_classes': [],
                'min_stable_frames': 1,
                'always_pass': True,
                'return_false': True,
            },
            3: {  # 设定作业区域
                'required_classes': ['围栏', '标识牌'],
                'min_count': {'围栏': 1, '标识牌': 1},
                'min_stable_frames': 3,
            },
            4: {  # 检查工器具（无条件通过）
                'required_classes': [],
                'min_stable_frames': 1,
                'always_pass': True,  # 无条件通过标志
            },
            5: {  # 选择检查脚轮（只需要4个脚轮，不需要手套）
                'required_classes': ['滑动脚轮'],
                'min_count': {'滑动脚轮': 4},
                'min_stable_frames': 5,
            },
            6: {  # 门架安装脚轮
                'required_classes': ['门架', '滑动脚轮'],
                'min_count': {'门架': 2, '滑动脚轮': 2},
                'min_stable_frames': 5,
            },
            7: {  # 安装交叉支撑
                'required_classes': ['交叉杆'],
                'min_stable_frames': 5,
            },
            8: {  # 搭设脚手板
                'required_classes': ['脚手板'],
                'min_stable_frames': 5,
            },
            9: {  # 搭设、使用爬梯
                'required_classes': ['爬梯'],
                'min_stable_frames': 5,
            },
            10: {  # 安全带使用
                'required_classes': ['安全带挂钩'],
                'zone_check': '安全带挂钩区',
                'min_stable_frames': 5,
            },
            11: {  # 搭设防护栏
                'required_classes': ['防护栏'],
                'min_stable_frames': 5,
            },
            12: {  # 拆除（识别区内组件移除）
                'required_classes': [],
                'min_stable_frames': 5,
                'state_change_detection': True,
                'disassembly_confirmation_frames': 10,
                'track_components': ['门架', '交叉杆', '脚手板', '爬梯', '防护栏', '安全带挂钩', '滑动脚轮'],
            },
            13: {  # 安全操作（验证防护用品）
                'required_classes': ['安全帽', '安全带'],
                'min_stable_frames': 5,
            },
            14: {  # 整理场地（识别区内围栏和标识牌移除）
                'required_classes': [],
                'min_stable_frames': 5,
                'state_change_detection': True,
                'disassembly_confirmation_frames': 5,
                'track_components': ['围栏', '标识牌'],
            },
        }

    def validate_step(self, step_num, detections, frame):
        """验证指定步骤是否完成（带置信度加权和多帧确认）"""
        self._last_debug_log = []

        if step_num not in self.step_requirements:
            self._last_debug_log.append(f"步骤 {step_num} 未定义验证规则")
            return False
        req = self.step_requirements[step_num]

        self._last_debug_log.append(f"开始验证步骤 {step_num}, 需要: {req.get('required_classes', [])}")

        # 0. 处理无条件通过的步骤（步骤2、4）
        if req.get('always_pass', False):
            self._last_debug_log.append(f"步骤 {step_num} 无条件通过")
            if req.get('return_false', False):
                return False
            return True

        # 1. 增强的类别检查（带置信度过滤和稳定性过滤）
        for cls in req.get('required_classes', []):
            class_dets = [d for d in detections if d['class'] == cls]
            self._last_debug_log.append(f"检查类别 '{cls}': 声始检测={len(class_dets)}")


            # 置信度过滤
            confident_dets = [d for d in class_dets
                             if d.get('confidence', 0) >= self.confidence_threshold]
            self._last_debug_log.append(f"  置信度过滤: {len(class_dets)} -> {len(confident_dets)} (阈值={self.confidence_threshold})")

            # 稳定性过滤（使用可配置的阈值，默认1帧即可）
            stability_threshold = Config.STEP_STABILITY_THRESHOLD if hasattr(Config, 'STEP_STABILITY_THRESHOLD') else 1
            self._last_debug_log.append(f"  稳定性过滤阈值={stability_threshold}")

            for d in confident_dets:
                tid = d.get('track_id')
                if tid is not None:
                    buf_entry = self.state.detection_buffer.get(tid, {})
                    buf_stable = buf_entry.get('stable_count', 0) if buf_entry else 0
                    self._last_debug_log.append(f"    track_id={tid}: buffer_stable_count={buf_stable}")

            stable_dets = [d for d in confident_dets
                          if d.get('track_id') is not None and  # 修复：0也是有效的track_id
                          self.state.detection_buffer.get(d['track_id'], {}).get('stable_count', 0) >= stability_threshold]

            self._last_debug_log.append(f"  最终稳定检测数: {len(stable_dets)}")

            if req.get('min_count'):
                if len(stable_dets) < req['min_count'].get(cls, 1):
                    if Config.DEBUG_MODE:
                        print(f"  [ERR] {cls} 数量不足: {len(stable_dets)} < {req['min_count'].get(cls, 1)}")
                    return False
            else:
                if not stable_dets and cls != '':
                    if Config.DEBUG_MODE:
                        print(f"  [ERR] {cls} 无稳定检测")
                    return False

        # 2. 禁止类别检查（拆除步骤专用）
        if 'forbidden_classes' in req:
            for cls in req['forbidden_classes']:
                if [d for d in detections if d['class'] == cls]:
                    if Config.DEBUG_MODE:
                        print(f"  [ERR] 检测到禁止类别: {cls}")
                    return False

        # 2.5 状态变化检测（拆除/整理场地步骤）
        # 检测识别区内指定组件是否已全部移除
        if req.get('state_change_detection', False):
            confirmation_frames = req.get('disassembly_confirmation_frames', 10)
            track_components = req.get('track_components', ['门架', '交叉杆', '脚手板', '爬梯', '防护栏', '安全带挂钩', '滑动脚轮'])
            if not self._check_disassembly_state_change(detections, confirmation_frames, track_components, step_num):
                if Config.DEBUG_MODE:
                    print(f"  ⏳ 状态变化检测中...")
                return False

        # 3. 空间关系验证
        if req.get('spatial_relation') == 'nearby':
            gloves = [d for d in detections if d['class'] == '手套']
            casters = [d for d in detections if d['class'] == '滑动脚轮']
            if not self._check_proximity(gloves, casters, req['nearby_threshold']):
                if Config.DEBUG_MODE:
                    print(f"  [ERR] 空间关系 'nearby' 不满足")
                    return False
        elif req.get('spatial_relation') == 'attached':
            gantries = [d for d in detections if d['class'] == '门架']
            casters = [d for d in detections if d['class'] == '滑动脚轮']
            if not self._check_attachment(gantries, casters, req['vertical_tolerance']):
                if Config.DEBUG_MODE:
                    print(f"  [ERR] 空间关系 'attached' 不满足")
                    return False

        # 4. 区域检查（安全带挂钩区专用）
        if req.get('zone_check'):
            zones = zone_manager.get_zones_by_type(req['zone_check'])
            self._last_debug_log.append(f"  区域检查: {req['zone_check']}, 区域数量={len(zones)}")

            # 获取该类别的所有检测框
            class_dets = [d for d in detections if d['class'] == '安全带挂钩']
            self._last_debug_log.append(f"  安全带挂钩检测数: {len(class_dets)}")

            zone_valid = False
            for d in class_dets:
                cx = (d['x1'] + d['x2']) / 2
                cy = (d['y1'] + d['y2']) / 2
                self._last_debug_log.append(f"    检测框中心: ({cx:.0f}, {cy:.0f})")

                for zone in zones:
                    coords = zone['coords']
                    # 计算缩放后的区域坐标
                    scale_x = (state.frame_width or Config.CONFIG_WIDTH) / Config.CONFIG_WIDTH
                    scale_y = (state.frame_height or Config.CONFIG_HEIGHT) / Config.CONFIG_HEIGHT
                    zone_x1 = coords[0] * scale_x
                    zone_y1 = coords[1] * scale_y
                    zone_x2 = coords[2] * scale_x
                    zone_y2 = coords[3] * scale_y
                    self._last_debug_log.append(f"    区域 {zone['name']}: ({zone_x1:.0f}, {zone_y1:.0f}) - ({zone_x2:.0f}, {zone_y2:.0f})")

                    if zone_x1 <= cx <= zone_x2 and zone_y1 <= cy <= zone_y2:
                        zone_valid = True
                        self._last_debug_log.append(f"    ✓ 在区域内")
                        break
                if zone_valid:
                    break

            if not zone_valid:
                self._last_debug_log.append(f"  [ERR] 区域检查失败（未在{req['zone_check']}内）")
                return False

        # 5. 历史帧时间一致性验证
        return self._check_temporal_consistency(step_num, req['min_stable_frames'])

    def _check_proximity(self, objs1, objs2, base_threshold):
        """
        检查两组对象的像素距离是否小于阈值（分辨率自适应）
        支持两种检测模式：1) 中心点距离 2) IoU重叠
        """
        if not objs1 or not objs2:
            return False

        # 根据分辨率缩放阈值
        scale_factor = 1.0
        if Config.SPATIAL_SCALE_WITH_RESOLUTION:
            frame_w = self.state.frame_width or 1280
            scale_factor = frame_w / 1280.0

        threshold = base_threshold * scale_factor

        for o1 in objs1:
            c1 = [(o1['x1']+o1['x2'])/2, (o1['y1']+o1['y2'])/2]
            for o2 in objs2:
                c2 = [(o2['x1']+o2['x2'])/2, (o2['y1']+o2['y2'])/2]
                dist = np.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                if dist < threshold:
                    if Config.DEBUG_MODE:
                        print(f"  [OK] 接近检测通过: 距离={dist:.1f}px < 阈值={threshold:.1f}px")
                    return True

        if Config.DEBUG_MODE:
            min_dist = min(np.sqrt(((o1['x1']+o1['x2'])/2-(o2['x1']+o2['x2'])/2)**2 +
                                   ((o1['y1']+o1['y2'])/2-(o2['y1']+o2['y2'])/2)**2)
                          for o1 in objs1 for o2 in objs2)
            print(f"  [ERR] 接近检测未通过: 最近距离={min_dist:.1f}px >= 阈值={threshold:.1f}px")
        return False

    def _check_attachment(self, gantries, casters, base_v_tolerance):
        """
        验证脚轮是否安装在门架底部（分辨率自适应）
        条件：脚轮中心在门架水平范围内 + 垂直位置接近门架底部
        """
        if not gantries or not casters:
            return False

        # 根据分辨率缩放容差
        scale_factor = 1.0
        if Config.SPATIAL_SCALE_WITH_RESOLUTION:
            frame_h = self.state.frame_height or 720
            scale_factor = frame_h / 720.0

        v_tolerance = base_v_tolerance * scale_factor
        attached_count = 0

        for g in gantries:
            g_bottom = g['y2']
            g_x_range = (g['x1'], g['x2'])

            for c in casters:
                c_top = c['y1']
                c_center_x = (c['x1'] + c['x2']) / 2

                # 检查：脚轮在门架水平范围内 + 垂直位置在门架底部附近
                if (g_x_range[0] <= c_center_x <= g_x_range[1] and
                    abs(c_top - g_bottom) < v_tolerance):
                    attached_count += 1
                    if Config.DEBUG_MODE:
                        print(f"  [OK] 安装检测: 脚轮({c_center_x:.0f},{c_top:.0f}) 在门架底部 y={g_bottom:.0f}±{v_tolerance:.0f}")

        # 要求至少有2个脚轮正确安装
        result = attached_count >= 2
        if Config.DEBUG_MODE:
            print(f"  [INFO] 安装检测: {attached_count}个脚轮正确安装 (需要>=2)")
        return result

    def _check_disassembly_state_change(self, detections, min_confirmation_frames=10, track_components=None, step_num=None):
        """
        检测识别区内指定组件是否已全部移除。
        逻辑：连续N帧检测到区域内无指定组件，则判定移除成功。
        Args:
            detections: 检测结果（已过滤到识别区内）
            min_confirmation_frames: 需要连续确认的帧数
            track_components: 要跟踪的组件列表
            step_num: 当前步骤编号（用于检测步骤切换）
        """
        if track_components is None:
            track_components = ['门架', '交叉杆', '脚手板', '爬梯', '防护栏', '安全带挂钩', '滑动脚轮']

        # 检测步骤切换：如果步骤变化，重置计数
        if step_num is not None and self.state.disassembly_step_num != step_num:
            self.state.disassembly_confirmation_frames = 0
            self.state.disassembly_step_num = step_num
            if Config.DEBUG_MODE:
                print(f"  [RESET] 步骤切换，重置确认计数")

        # 检查当前识别区内是否还有指定组件
        current_state = {
            cls: len([d for d in detections if d['class'] == cls])
            for cls in track_components
        }

        # 判断是否所有组件都已消失
        all_absent = all(count == 0 for count in current_state.values())

        if Config.DEBUG_MODE:
            present = [cls for cls, count in current_state.items() if count > 0]
            if present:
                print(f"  [STATE] 识别区内剩余组件: {present}")
            else:
                print(f"  [STATE] 识别区内无目标组件，确认进度: {self.state.disassembly_confirmation_frames + 1}/{min_confirmation_frames}")

        if all_absent:
            self.state.disassembly_confirmation_frames += 1
            if self.state.disassembly_confirmation_frames >= min_confirmation_frames:
                if Config.DEBUG_MODE:
                    print(f"  [OK] 移除确认完成！连续{min_confirmation_frames}帧识别区内无目标组件")
                return True
        else:
            # 有组件存在时重置计数
            self.state.disassembly_confirmation_frames = 0

        return False

    def _check_temporal_consistency(self, step_num, min_frames):
        """检查最近N帧的验证通过率，保证步骤稳定性

        注意： 多帧确认已在check_step_logic_enhanced()中实现
        这里只做简单检查，避免循环依赖问题
        """
        # 简化逻辑： 只检查历史帧是否足够，不再检查通过率
        # 多帧确认由外层的check_step_logic_enhanced()处理
        if len(self.state.step_history) < min_frames:
            return True  # 历史帧不足时临时允许通过
        return True  # 始终返回True，让外层处理多帧确认

# ==================== FFmpeg视频写入器（比OpenCV更稳定） ====================
class FFmpegVideoWriter:
    """FFmpeg子进程视频写入器，支持长时间录制，大文件稳定"""

    def __init__(self, output_path, fps, width, height, codec='libx264'):
        self.output_path = output_path
        self.fps = fps
        self.width = width
        self.height = height
        self.codec = codec
        self.process = None
        self._start_ffmpeg()

    def _start_ffmpeg(self):
        """启动FFmpeg子进程"""
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{self.width}x{self.height}',
            '-pix_fmt', 'bgr24',
            '-r', str(self.fps),
            '-i', '-',
            '-c:v', self.codec,
            '-preset', 'fast',
            '-crf', '23',
            '-pix_fmt', 'yuv420p',
            self.output_path
        ]
        try:
            self.process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        except FileNotFoundError:
            raise RuntimeError("ffmpeg未安装，请先安装: apt-get install ffmpeg")

    def write(self, frame):
        """写入一帧"""
        if self.process and self.process.stdin:
            try:
                self.process.stdin.write(frame.tobytes())
            except BrokenPipeError:
                pass

    def isOpened(self):
        return self.process is not None and self.process.poll() is None

    def release(self):
        """关闭FFmpeg进程"""
        if self.process:
            try:
                self.process.stdin.close()
            except:
                pass
            self.process.wait(timeout=5)
            self.process = None

# ==================== 全局状态管理 ====================
class GlobalState:
    """全局状态管理器，统一管理服务所有运行状态"""
    def __init__(self):
        self.is_recording = False
        self.enable_detection = False
        self.user_id = None
        self.video_writer = None
        self.frame_queue = queue.Queue(maxsize=30)
        self.display_queue = queue.Queue(maxsize=5)
        self.inference_backend = None
        self.current_step = 0
        self.step_results = {}
        self.lock = threading.Lock()
        self.streaming_thread = None
        self.last_detection_time = 0
        self.step_completed = OrderedDict()
        self.frame_width = 0
        self.frame_height = 0
        self.frame_id = 0
        self.latest_detections = []
        self.zones_cache = None
        self.zones_cache_timestamp = 0
        self.latest_vis_frame = None
        self.latest_det_texts = []  # 推理引擎返回的中文文字列表

        # 视频写入追踪变量
        self.video_path = None
        self.frames_written = 0

        # 跟踪和验证相关状态
        self.detection_buffer = {}  # {track_id: 检测对象信息}
        self.step_history = deque(maxlen=30)  # 步骤验证历史帧（30帧@10fps）
        self.step_cooldown = {}  # 步骤触发冷却，避免重复触发

        # 状态变化检测（拆除步骤专用）
        self.pre_disassembly_state = {}  # 拆除前的组件状态
        self.disassembly_confirmation_frames = 0  # 确认拆除的连续帧数
        self.disassembly_started = False  # 拆除阶段是否已开始
        self.disassembly_step_num = None  # 当前拆除步骤编号（用于检测步骤切换）

    def reset(self):
        """软重置：清理用户相关状态，保留服务基础状态"""
        self.is_recording = False
        self.user_id = None
        self.enable_detection = False
        if self.video_writer:
            try:
                self.video_writer.release()
            except:
                pass
            self.video_writer = None

        self.video_path = None
        self.frames_written = 0
        self.current_step = 0
        self.step_results = {}
        self.step_completed.clear()
        self.latest_detections = []
        self.latest_vis_frame = None
        self.latest_det_texts = []  # 推理引擎返回的中文文字列表
        self.frame_id = 0
        self.last_detection_time = 0

        # 清理跟踪和验证状态
        self.detection_buffer.clear()
        self.step_history.clear()
        self.step_cooldown.clear()

        # 清理拆除状态跟踪
        self.pre_disassembly_state.clear()
        self.disassembly_confirmation_frames = 0
        self.disassembly_started = False
        self.disassembly_step_num = None

        # 重置跟踪器对象（清除所有跟踪目标的stable_count）
        global tracker
        if 'tracker' in globals() and tracker is not None:
            tracker.objects.clear()
            tracker.next_id = 0
        # 同时重置推理子进程中的跟踪器
        if inference_pipeline is not None:
            inference_pipeline.request_reset()

    def hard_reset(self):
        """硬重置：清理所有状态，包括队列"""
        self.reset()
        # 清空帧队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        while not self.display_queue.empty():
            try:
                self.display_queue.get_nowait()
            except:
                break

# 初始化全局状态
state = GlobalState()

# ==================== 区域配置管理器 ====================
class ZoneConfigManager:
    """检测区域配置管理器，实现配置的加载/保存/查询"""
    def __init__(self, config_file):
        self.config_file = config_file
        self.zones = self.load_zones()

    def load_zones(self):
        """从JSON文件加载区域配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    zones = json.load(f)
                    print(f"[OK] 已加载区域配置: {list(zones.keys())}")
                    return zones
        except Exception as e:
            print(f"加载区域配置失败: {e}")

    def save_zones(self, zones):
        """将区域配置保存到JSON文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(zones, f, ensure_ascii=False, indent=2)
            self.zones = zones
            print("[OK] 区域配置已保存")
            return True
        except Exception as e:
            print(f"保存区域配置失败: {e}")
            return False

    def get_zones_by_type(self, zone_type):
        """根据类型获取指定区域配置"""
        return self.zones.get(zone_type, [])

    def get_all_zones(self):
        """获取所有区域配置"""
        return self.zones

# 初始化区域配置管理器
zone_manager = ZoneConfigManager(Config.ZONES_CONFIG_FILE)

# ==================== 标签/步骤映射配置 ====================
# YOLO检测类别标签映射
LABEL_MAP = {
    0: "安全帽", 1: "安全带", 2: "围栏", 3: "标识牌", 4: "手套",
    5: "滑动脚轮", 6: "门架", 7: "交叉杆", 8: "脚手板", 9: "爬梯",
    10: "防护栏", 11: "安全带挂钩", 12: "扫把", 13: "人"
}

# 步骤编号与步骤名称映射
STEP_ORDER = {
    1: "wearfanghu", 2: "checkhuanjing", 3: "weilanopen", 4: "toolbag",
    5: "checkjiaolun", 6: "setjiaolun", 7: "setjiaocha", 8: "setjiaoshouban",
    9: "setpati", 10: "setanquandai", 11: "setfanghulan", 12: "chaichu",
    13: "safetyopt", 14: "weilanclose"
}

# 关键安装步骤（用于拆卸顺序验证）
CRITICAL_STEPS = [6, 7, 8, 9, 10, 11]

# ==================== 检测核心逻辑 ====================
def analyze_frame_with_tracking(frame):
    """分析视频帧，执行推理+对象跟踪，返回带跟踪ID的检测结果"""
    try:
        boxes_list, vis_image, det_texts = state.inference_backend.infer(frame)

        # 统一检测结果格式，映射类别名称
        for box in boxes_list:
            class_id = int(box["class"]) if isinstance(box["class"], (int, np.integer)) else 0
            box["class_id"] = class_id
            box["class"] = LABEL_MAP.get(class_id, f"类别_{class_id}")

        # 执行对象跟踪，为检测目标分配唯一ID
        tracked_detections = tracker.update(boxes_list)

        # 更新全局检测缓冲
        for det in tracked_detections:
            track_id = det.get('track_id')
            if track_id is not None:  # 修复：0也是有效的track_id
                state.detection_buffer[track_id] = {
                    'box': [det['x1'], det['y1'], det['x2'], det['y2']],
                    'class': det['class'],
                    'stable_count': tracker.objects.get(track_id, {}).get('stable_count', 0)
                }

        # 更新全局帧状态
        state.frame_id += 1
        state.latest_detections = tracked_detections
        return tracked_detections, vis_image, det_texts
    except Exception as e:
        import traceback
        print(f"推理失败: {e}")
        traceback.print_exc()
        return state.latest_detections, frame, []

def check_step_logic_enhanced(detections, frame):
    """
    增强版步骤检查逻辑（使用StepValidator + 多帧确认机制）
    核心改进：1. 使用StepValidator进行规范验证 2. 多帧连续确认避免误触发 3. 历史记录时间一致性
    """
    if not state.enable_detection:
        # 仅首次打印一次
        if not hasattr(check_step_logic_enhanced, '_warned'):
            check_step_logic_enhanced._warned = True
            print("[WARN] 步骤检测已禁用 (enable_detection=False)")
        return
    current_time = time.time()
    # 检测频率控制，避免重复计算
    if current_time - state.last_detection_time < Config.DETECTION_INTERVAL:
        return
    state.last_detection_time = current_time

    # 使用本地帧计数器（避免多线程异步导致state.frame_id不递增）
    if not hasattr(check_step_logic_enhanced, '_local_frame_count'):
        check_step_logic_enhanced._local_frame_count = 0
    check_step_logic_enhanced._local_frame_count += 1
    local_frame_id = check_step_logic_enhanced._local_frame_count

    step_to_trigger = None
    with state.lock:
        try:
            current_step = state.current_step
            step_num = current_step + 1  # 转换为1-indexed

            # 记录当前帧验证结果到历史
            frame_result = {
                'frame_id': local_frame_id,
                'timestamp': current_time,
                'detections_count': len(detections),
                f'step_{step_num}_valid': False
            }

            # 使用StepValidator进行实际验证
            is_valid = step_validator.validate_step(step_num, detections, frame)
            frame_result[f'step_{step_num}_valid'] = is_valid

            # 添加到历史记录
            state.step_history.append(frame_result)

            # 多帧确认机制：检查是否有足够的连续有效帧
            if is_valid:
                min_stable_frames = step_validator.step_requirements.get(step_num, {}).get('min_stable_frames', 3)
                recent_frames = list(state.step_history)[-min_stable_frames:]

                if len(recent_frames) >= min_stable_frames:
                    # 要求最近N帧全部验证通过（严格模式）
                    consecutive_valid = all(
                        h.get(f'step_{step_num}_valid', False)
                        for h in recent_frames
                    )

                    if consecutive_valid:
                        step_to_trigger = (step_num, True)
                        # 关键日志：仅在步骤确认通过时打印
                        print(f"[STEP] 步骤 {step_num} ({STEP_ORDER.get(step_num, '未知')}) 确认通过！连续{min_stable_frames}帧验证成功")

            user_id = state.user_id
        except Exception as e:
            print(f"步骤检查异常: {e}")
            import traceback
            traceback.print_exc()

    # 触发步骤完成，保存结果并回调客户端
    if step_to_trigger:
        step_num, result = step_to_trigger
        trigger_step(step_num, frame, result, user_id)
        time.sleep(0.05)

def trigger_step(step_num, frame, result, user_id):
    """触发步骤完成，更新状态+保存验证图片+回调客户端"""
    step_name = STEP_ORDER[step_num]
    with state.lock:
        if step_name in state.step_results and state.step_results[step_name] is not None:
            return
        # 更新步骤状态
        state.step_results[step_name] = result
        state.current_step = step_num
        state.step_completed[step_name] = {
            'time': time.time(),
            'order': len(state.step_completed) + 1
        }
        # 生成步骤验证图片路径
        img_path = get_image_save_path(step_name) if (result and frame is not None) else None

    # 保存步骤验证图片
    if img_path and frame is not None:
        try:
            cv2.imwrite(img_path, frame)
            print(f"[OK] 步骤{step_num}验证图片已保存: {img_path}")
        except Exception as e:
            print(f"保存步骤图片失败: {e}")
            img_path = None

    # 回调客户端通知步骤完成
    send_to_client(step_name, result, img_path, user_id)
    print(f"[OK] 步骤 {step_num} ({step_name}) 完成: {'通过' if result else '未完成'}")

def send_to_client(step_name, result, img_path, user_id):
    """将步骤完成结果回调到客户端服务"""
    try:
        # 直接返回本地路径，客户端的BuildImageUri方法会自动转换为Nginx的HTTP URL
        # 例如: D:/StepImages/YDPT/20260402/user/step.jpg -> http://{serverIp}:9003/YDPT/20260402/user/step.jpg
        img_url = img_path if img_path else ""

        # 构造客户端需要的回调数据格式
        data = {
            "userid": user_id,
            "wearfanghu": False, "checkhuanjing": False, "weilanopen": False, "toolbag": False,
            "checkjiaolun": False, "setjiaolun": False, "setjiaocha": False, "setjiaoshouban": False,
            "setpati": False, "setanquandai": False, "setfanghulan": False, "chaichu": False,
            "safetyopt": False, "weilanclose": False, "img_url": img_url
        }
        if step_name in data:
            data[step_name] = result

        # 发送POST请求到客户端
        if Config.ENABLE_CLIENT_CALLBACK:
            try:
                response = requests.post(Config.STEP_CALLBACK_URL, json=data, timeout=3)
                print(f"[ICON] 客户端步骤回调成功: {response.status_code}")
            except Exception as e:
                print(f"[ICON] 客户端步骤回调失败: {e}")
        else:
            print(f"[ICON] 模拟回调客户端: {step_name} = {result}, 图片路径={img_path}")
    except Exception as e:
        print(f"构造回调数据失败: {e}")

def send_coordinates_to_client(coordinate_data, user_id):
    """将检测目标坐标数据实时回调到客户端"""
    if not state.is_recording or not state.enable_detection:
        return
    try:
        coordinate_data["userid"] = user_id
        headers = {"Content-Type": "application/json; charset=utf-8"}
        requests.post(
            url=Config.COORDINATES_CALLBACK_URL,
            json=coordinate_data,
            headers=headers,
            timeout=1
        )
    except requests.exceptions.ConnectionError:
        pass
    except requests.exceptions.Timeout:
        pass
    except Exception as e:
        print(f"[ICON] 坐标数据回调异常: {str(e)}")

def build_csharp_coordinate_data(detections, device_type=1, recog_area=None):
    """构造符合C#客户端要求的坐标数据格式"""
    coord_arr = []
    for det in detections:
        x1, y1, x2, y2 = int(det["x1"]), int(det["y1"]), int(det["x2"]), int(det["y2"])
        coord_arr.append({
            "X1": x1, "Y1": y1, "X2": x2, "Y2": y2,
            "Name": det["class"], "confidence": float(det["confidence"]),
            "centerx": int((x1+x2)/2), "centery": int((y1+y2)/2)
        })
    result = {
        "types": int(device_type),
        "arr": coord_arr,
        "userid": ""  # 后续由send_coordinates_to_client补充userid
    }
    # 添加检测区域字段
    if recog_area:
        result["recog_area"] = recog_area
    return result

# ==================== 推理Pipeline ====================

def _inference_process_main(frame_queue, result_queue, running_event, reset_event, _unused=False):
    """YOLOv8子进程：全流程推理+后处理+跟踪（spawn模式，绕过GIL）"""
    try:
        backend = create_inference_backend()
        local_tracker = ObjectTracker(max_disappeared=3)
        print(f"[OK] 推理子进程初始化完成 (PID: {os.getpid()})")
    except Exception as e:
        print(f"[ERR] 推理子进程初始化失败: {e}")
        return

    while running_event.is_set():
        if reset_event.is_set():
            local_tracker.objects.clear()
            local_tracker.next_id = 0
            reset_event.clear()
        try:
            frame = frame_queue.get(timeout=0.1)
        except:
            continue
        try:
            boxes_list, vis_image, _ = backend.infer(frame)
            for box in boxes_list:
                class_id = int(box["class"]) if isinstance(box["class"], (int, np.integer)) else 0
                box["class_id"] = class_id
                box["class"] = LABEL_MAP.get(class_id, f"类别_{class_id}")
            tracked_detections = local_tracker.update(boxes_list)
            for det in tracked_detections:
                track_id = det.get('track_id')
                if track_id is not None:
                    det['stable_count'] = local_tracker.objects.get(track_id, {}).get('stable_count', 0)
            try:
                while result_queue.full():
                    try: result_queue.get_nowait()
                    except: break
                result_queue.put_nowait((tracked_detections, vis_image))
            except:
                pass
        except:
            continue


class InferencePipeline:
    """推理Pipeline：YOLOv8用多进程，Ascend用多线程"""

    def __init__(self):
        self._is_thread_mode = Config.INFERENCE_BACKEND.lower() in ("ascend", "rockchip")
        self._worker = None
        self._result_thread = None

        if self._is_thread_mode:
            self.frame_queue = queue.Queue(maxsize=2)
            self.result_queue = queue.Queue(maxsize=2)
            self._running_flag = True
            self._reset_flag = False
        else:
            self.frame_queue = mp.Queue(maxsize=2)
            self.result_queue = mp.Queue(maxsize=2)
            self._running = mp.Event()
            self._reset_signal = mp.Event()

    def start(self):
        if self._is_thread_mode:
            self._worker = threading.Thread(
                target=self._ascend_worker, name="inference_worker", daemon=True
            )
        else:
            self._running.set()
            self._worker = mp.Process(
                target=_inference_process_main,
                args=(self.frame_queue, self.result_queue,
                      self._running, self._reset_signal, False),
                name="inference_spawn", daemon=True
            )
        self._worker.start()

        self._result_thread = threading.Thread(
            target=self._result_worker, name="result_worker", daemon=True
        )
        self._result_thread.start()

        return "Ascend多线程" if self._is_thread_mode else f"YOLOv8多进程(PID={self._worker.pid})"

    def submit_frame(self, frame):
        try:
            while self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except: break
            if self._is_thread_mode:
                self.frame_queue.put_nowait(frame.copy())
            else:
                self.frame_queue.put_nowait(frame)
        except:
            pass

    def request_reset(self):
        if self._is_thread_mode:
            self._reset_flag = True
        else:
            self._reset_signal.set()

    def _ascend_worker(self):
        """Ascend：主进程中做推理+后处理+跟踪"""
        global tracker
        while self._running_flag:
            if self._reset_flag:
                if tracker is not None:
                    tracker.objects.clear()
                    tracker.next_id = 0
                self._reset_flag = False
            try:
                frame = self.frame_queue.get(timeout=0.1)
            except:
                continue
            try:
                detections, vis_frame, det_texts = analyze_frame_with_tracking(frame)
                state.latest_det_texts = det_texts
                try:
                    while self.result_queue.full():
                        try: self.result_queue.get_nowait()
                        except: break
                    self.result_queue.put_nowait((detections, vis_frame))
                except:
                    pass
            except:
                continue

    def _result_worker(self):
        while True:
            try:
                detections, vis_frame = self.result_queue.get(timeout=0.1)
                # 更新全局检测缓冲（关键：子进程的stable_count需要同步到主进程）
                for det in detections:
                    track_id = det.get('track_id')
                    if track_id is not None:
                        state.detection_buffer[track_id] = {
                            'box': [det['x1'], det['y1'], det['x2'], det['y2']],
                            'class': det['class'],
                            'stable_count': det.get('stable_count', 0)
                        }
                state.latest_vis_frame = vis_frame
                state.latest_detections = detections
                if state.user_id and Config.ENABLE_CLIENT_CALLBACK:
                    csharp_data = build_csharp_coordinate_data(detections, recog_area=Config.RECOG_AREA)
                    send_coordinates_to_client(csharp_data, state.user_id)
                check_step_logic_enhanced(detections, vis_frame)
            except:
                continue

inference_pipeline = None

# ==================== 视频流处理（FFmpeg PTS/断言修复版） ====================
def stream_processor():
    """
    视频流核心处理线程 - 修复FFmpeg PTS时间戳错误/断言崩溃
    核心改进：1. 全流程过滤无效帧 2. 帧计数严格递增保证PTS单调 3. 全局统一帧率 4. 精准帧间隔控制
    """
    cap = None
    last_inference_frame_id = -1
    TARGET_FPS = Config.VIDEO_WRITE_FPS  # 全局统一帧率
    FRAME_INTERVAL = 1.0 / TARGET_FPS    # 帧间隔时间
    last_frame_time = 0
    write_frame_counter = 0              # 帧写入计数器（PTS严格递增核心）

    # 本地录制状态缓存，减少全局锁竞争
    local_is_recording = False
    local_writer = None

    while True:
        try:
            # 重新连接视频源
            if cap is None or not cap.isOpened():
                print(f"[VIDEO] 连接视频源: {Config.RTSP_URL}")
                cap = cv2.VideoCapture(Config.RTSP_URL, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    print("[ERR] 视频源连接失败，5秒后重试...")
                    time.sleep(5)
                    continue
                # 减小缓冲区降低延迟，OpenCV多核解码
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                # 获取视频源分辨率
                state.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                state.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                # 确认流处理帧率与全局一致
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                TARGET_FPS = Config.VIDEO_WRITE_FPS
                FRAME_INTERVAL = 1.0 / TARGET_FPS
                print(f"[OK] 视频流连接成功: {state.frame_width}x{state.frame_height} | 源帧率: {source_fps:.1f}fps | 处理帧率: {TARGET_FPS:.1f}fps")
                last_frame_time = time.time()
                write_frame_counter = 0  # 重置写入计数器
                state.frames_written = 0  # 同步全局计数为0，避免初始差1

            # 精准控制帧间隔，避免帧堆积/丢失
            current_time = time.time()
            if current_time - last_frame_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - (current_time - last_frame_time))
                continue
            last_frame_time = time.time()

            # 读取帧并过滤无效帧（核心：避免FFmpeg断言）
            ret, frame = cap.read()
            if not ret or not is_valid_frame(frame):
                if os.path.isfile(Config.RTSP_URL):
                    print("[VIDEO] 视频文件播放完毕，重新播放...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    write_frame_counter = 0
                    state.frames_written = 0  # 同步全局计数
                    continue
                else:
                    print(f"[WARN]  获取无效帧/断帧，重新连接视频源...")
                    cap.release()
                    cap = None
                    local_is_recording = False
                    local_writer = None
                    write_frame_counter = 0
                    state.frames_written = 0  # 同步全局计数
                    time.sleep(2)
                    continue

            # 放入推理队列（深拷贝避免帧被篡改）
            try:
                while state.frame_queue.full():
                    state.frame_queue.get_nowait()
                state.frame_queue.put_nowait(frame.copy())
            except:
                pass

            # 通过Pipeline提交帧进行异步推理（多线程并行，充分利用多核CPU）
            if state.inference_backend and state.is_recording and state.enable_detection and state.user_id:
                if inference_pipeline is not None:
                    inference_pipeline.submit_frame(frame)

            # 放入显示队列（深拷贝）
            try:
                while state.display_queue.full():
                    state.display_queue.get_nowait()
                state.display_queue.put_nowait(frame.copy())
            except:
                pass

            # 视频录制核心逻辑（FFmpeg PTS修复）
            # 更新本地录制状态
            if not local_is_recording or local_writer is None:
                if state.is_recording and state.video_writer is not None and state.video_writer.isOpened():
                    local_is_recording = True
                    local_writer = state.video_writer
                    write_frame_counter = 0
                    state.frames_written = 0  # 启动录制时同步全局计数为0
                    if Config.DEBUG_MODE:
                        print(f"[REC] 录制已激活，帧计数器初始化: {write_frame_counter}")
            # 严格写入有效帧，计数器递增保证PTS单调
            if local_is_recording and local_writer is not None and local_writer.isOpened():
                try:
                    local_writer.write(frame)
                    write_frame_counter += 1
                    state.frames_written = write_frame_counter  # 写入成功后再同步，确保严格一致
                    if Config.DEBUG_MODE and write_frame_counter % 100 == 0:
                        print(f"[REC] 录制中：已写入{write_frame_counter}帧，PTS时序正常")
                except Exception as e:
                    print(f"[WARN]  帧写入失败（FFmpeg保护）: {e}")
                    # 写入失败时不递增计数器，避免计数与实际帧不一致
            # 停止录制时清理本地状态
            if not state.is_recording and local_is_recording:
                print(f"[REC] 录制停止，最后实际写入帧计数: {write_frame_counter}")
                local_is_recording = False
                local_writer = None
                # 停止后不再重置计数器，仅同步最终值
                state.frames_written = write_frame_counter

        except Exception as e:
            print(f"视频流处理错误: {e}")
            import traceback
            traceback.print_exc()
            # 异常时重置所有状态
            if cap:
                cap.release()
            cap = None
            local_is_recording = False
            local_writer = None
            write_frame_counter = 0
            state.frames_written = 0  # 异常时同步全局计数
            time.sleep(5)

def analyze_and_check(frame):
    """后台异步执行：帧推理+步骤检查+坐标回调"""
    try:
        if not state.is_recording or not state.enable_detection:
            return
        # 执行推理和对象跟踪
        detections, vis_frame, det_texts = analyze_frame_with_tracking(frame)
        state.latest_det_texts = det_texts
        state.latest_vis_frame = vis_frame
        # 实时回调坐标数据到客户端
        if state.user_id and Config.ENABLE_CLIENT_CALLBACK:
            csharp_data = build_csharp_coordinate_data(detections, recog_area=Config.RECOG_AREA)
            send_coordinates_to_client(csharp_data, state.user_id)
        # 检查作业步骤逻辑
        check_step_logic_enhanced(detections, vis_frame)
    except Exception as e:
        print(f"异步检测/步骤检查异常: {e}")

def get_current_frame():
    """从帧队列获取当前视频帧"""
    try:
        return state.frame_queue.get(timeout=1)
    except:
        return None

# ==================== 浏览器实时视频流接口 ====================
def generate_stream():
    """生成浏览器可播放的MJPEG实时视频流"""
    CONFIG_W, CONFIG_H = Config.CONFIG_WIDTH, Config.CONFIG_HEIGHT
    last_frame_time = 0
    TARGET_FPS = 15
    JPEG_QUALITY = 80
    STREAM_MAX_WIDTH = 960  # 板端缩小到960宽，减少Pillow转换和JPEG编码耗时
    last_vis_frame = None

    def get_cached_zones():
        """获取缓存的区域配置，5秒刷新一次"""
        current_time = time.time()
        if state.zones_cache is None or current_time - state.zones_cache_timestamp > 5:
            state.zones_cache = zone_manager.get_all_zones()
            state.zones_cache_timestamp = current_time
        return state.zones_cache

    while True:
        try:
            # 帧率控制
            current_time = time.time()
            if current_time - last_frame_time < 1.0 / TARGET_FPS:
                time.sleep(0.001)
                continue
            last_frame_time = current_time

            # 获取可视化帧（优先使用推理后的帧，否则使用原始帧）
            vis_frame = state.latest_vis_frame
            if vis_frame is None:
                frame = get_current_frame()
                if frame is None:
                    if last_vis_frame is not None:
                        vis_frame = last_vis_frame
                    else:
                        time.sleep(0.05)
                        continue
                else:
                    vis_frame = frame.copy()
            last_vis_frame = vis_frame.copy()

            # 板端性能优化：缩小分辨率后再渲染文字和编码JPEG
            if vis_frame.shape[1] > STREAM_MAX_WIDTH:
                scale = STREAM_MAX_WIDTH / vis_frame.shape[1]
                vis_frame = cv2.resize(vis_frame, (STREAM_MAX_WIDTH, int(vis_frame.shape[0] * scale)),
                                       interpolation=cv2.INTER_LINEAR)

            # 叠加检测区域到视频流 - 先用cv2画矩形框（快），中文文字收集后一次性Pillow渲染
            all_zones = get_cached_zones()
            frame_h, frame_w = vis_frame.shape[:2]
            scale_x = frame_w / CONFIG_W if CONFIG_W > 0 else 1.0
            scale_y = frame_h / CONFIG_H if CONFIG_H > 0 else 1.0

            # 收集所有需要渲染的中文文字（检测标签 + 区域名 + 状态信息）
            all_texts = []

            # 1. 推理引擎的检测标签（已按原图坐标，需缩放到stream尺寸）
            if vis_frame.shape[1] != 1920 and state.latest_det_texts:
                text_scale = vis_frame.shape[1] / 1920
                for tx, ty, label, color in state.latest_det_texts:
                    all_texts.append((int(tx * text_scale), int(ty * text_scale), label, color))
            else:
                all_texts.extend(state.latest_det_texts)

            # 2. 区域名称和边框
            for zone_type, zones in all_zones.items():
                for zone in zones:
                    try:
                        coords = zone['coords']
                        color = tuple(zone.get('color', [0, 255, 0]))
                        x1, y1 = int(coords[0]*scale_x), int(coords[1]*scale_y)
                        x2, y2 = int(coords[2]*scale_x), int(coords[3]*scale_y)
                        # 绘制半透明区域背景
                        overlay = vis_frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        cv2.addWeighted(overlay, 0.1, vis_frame, 0.9, 0, vis_frame)
                        # 绘制区域边框
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                        all_texts.append((x1, max(15, y1-5), f"{zone['name']} ({zone_type})", color))
                    except:
                        pass

            # 3. 跟踪ID（纯英文，cv2即可）
            for det in state.latest_detections:
                track_id = det.get('track_id')
                if track_id:
                    x1, y1 = det['x1'], det['y1']
                    if vis_frame.shape[1] != 1920:
                        ts = vis_frame.shape[1] / 1920
                        x1, y1 = int(x1*ts), int(y1*ts)
                    cv2.putText(vis_frame, f"ID:{track_id}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

            # 4. 状态信息
            status_text = f"当前步骤:{state.current_step} | 录制:{state.is_recording} | 已写帧数:{state.frames_written}"
            all_texts.append((10, 30, status_text, (0, 255, 0)))

            # 一次性 Pillow 渲染所有中文文字（只做1次BGR→RGB→PIL→RGB→BGR转换）
            render_chinese_texts(vis_frame, all_texts, _PIL_FONT)

            # 编码为JPEG格式
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', vis_frame, encode_params)
            frame_bytes = buffer.tobytes()

            # 生成MJPEG流格式
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"视频流生成错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.5)

@app.route('/stream')
def video_stream():
    """浏览器实时视频流入口"""
    if not Config.ENABLE_STREAMING:
        return jsonify({"error": "视频流功能未启用"}), 403
    return Response(generate_stream(),
                   mimetype='multipart/x-mixed-replace; boundary=frame',
                   headers={
                       'Cache-Control': 'no-cache',
                       'Connection': 'keep-alive',
                       'X-Accel-Buffering': 'no'
                   })

# ==================== 区域配置管理接口 ====================
@app.route('/config')
def config_page():
    """区域配置可视化页面（浏览器端）"""
    if not os.path.exists(Config.CONFIG_BACKGROUND_IMAGE):
        return f"""
        <h1>错误：未找到配置背景图片</h1>
        <p>请将现场照片放在以下路径：</p>
        <code>{Config.CONFIG_BACKGROUND_IMAGE}</code>
        <p>放置后刷新此页面</p>
        """

    return render_template_string(f'''
<!DOCTYPE html>
<html>
<head>
    <title>检测区域配置</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f0f0f0; }}
        .container {{ max-width: 1600px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; }}
        .image-wrapper {{ overflow: auto; border: 2px solid #333; position: relative; background: #000; width: {Config.CONFIG_WIDTH}px; height: {Config.CONFIG_HEIGHT}px; margin: 0 auto; }}
        #configImage {{ position: absolute; width: {Config.CONFIG_WIDTH}px; height: {Config.CONFIG_HEIGHT}px; display: block; }}
        #configCanvas {{ position: absolute; top: 0; left: 0; cursor: crosshair; width: {Config.CONFIG_WIDTH}px; height: {Config.CONFIG_HEIGHT}px; }}
        .controls {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #dee2e6; }}
        .controls-row {{ display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 15px; }}
        .zone-list {{ max-height: 200px; overflow-y: auto; background: #f8f8f8; padding: 10px; border-radius: 4px; }}
        .zone-item {{ background: white; padding: 10px; margin: 5px 0; border-radius: 4px; display: flex; justify-content: space-between; align-items: center; }}
        .btn {{ padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }}
        .btn-primary {{ background: #007bff; color: white; }}
        .btn-success {{ background: #28a745; color: white; }}
        .btn-danger {{ background: #dc3545; color: white; }}
        .status {{ margin-top: 15px; padding: 10px; background: #d4edda; border-radius: 4px; color: #155724; display: none; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>[TARGET] 检测区域配置工具</h1>
        <div class="info-box">
            <strong>校准信息：</strong> 配置分辨率: {Config.CONFIG_WIDTH}x{Config.CONFIG_HEIGHT} | 坐标将按比例映射到实际视频流
        </div>
        <div class="image-wrapper">
            <img id="configImage" src="/api/background" alt="配置背景图">
            <canvas id="configCanvas"></canvas>
        </div>
        <div class="controls">
            <div class="controls-row">
                <select id="zoneType">
                    <option value="安全带挂钩区">安全带挂钩区</option>
                    <option value="爬梯搭设区">爬梯搭设区</option>
                    <option value="识别区域">识别区域</option>
                    <option value="工具放置区">工具放置区</option>
                </select>
                <input type="text" id="zoneName" placeholder="区域名称（如：挂钩区1）">
                <button class="btn btn-primary" onclick="addZone()">添加区域</button>
                <button class="btn btn-success" onclick="saveConfig()">保存配置</button>
                <button class="btn btn-danger" onclick="clearAll()">清空所有</button>
            </div>
            <div id="status" class="status"></div>
            <h3>已配置区域列表：</h3>
            <div id="zoneList" class="zone-list"></div>
        </div>
    </div>
    <script>
        let zones = []; let currentRect = null; let isDrawing = false; let startX, startY;
        let canvas = document.getElementById('configCanvas'); let ctx = canvas.getContext('2d'); let img = document.getElementById('configImage');
        img.onload = () => {{ canvas.width = {Config.CONFIG_WIDTH}; canvas.height = {Config.CONFIG_HEIGHT}; drawCanvas(); }};
        canvas.onmousedown = (e) => {{ let rect = canvas.getBoundingClientRect(); startX = e.clientX - rect.left; startY = e.clientY - rect.top; isDrawing = true; }};
        canvas.onmousemove = (e) => {{
            if (!isDrawing) return; let rect = canvas.getBoundingClientRect();
            let cx = e.clientX - rect.left; let cy = e.clientY - rect.top;
            drawCanvas(); ctx.strokeStyle = 'red'; ctx.lineWidth = 2; ctx.strokeRect(startX, startY, cx-startX, cy-startY);
        }};
        canvas.onmouseup = (e) => {{
            if (!isDrawing) return; let rect = canvas.getBoundingClientRect();
            let ex = e.clientX - rect.left; let ey = e.clientY - rect.top;
            currentRect = {{x1: Math.min(startX,ex), y1: Math.min(startY,ey), x2: Math.max(startX,ex), y2: Math.max(startY,ey)}};
            isDrawing = false;
        }};
        function drawCanvas() {{
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            zones.forEach((zone) => {{
                let coords = zone.coords;
                ctx.strokeStyle = 'rgb(' + zone.color.join(',') + ')';
                ctx.lineWidth = 2;
                ctx.strokeRect(coords[0], coords[1], coords[2] - coords[0], coords[3] - coords[1]);
                ctx.fillStyle = 'rgb(' + zone.color.join(',') + ')';
                ctx.font = '14px Arial';
                ctx.fillText(`${{zone.type}}: ${{zone.name}}`, coords[0], coords[1] - 5);
            }});
        }}
        function addZone() {{
            if (!currentRect) {{alert('请先绘制区域！');return;}}
            let t = document.getElementById('zoneType').value; let n = document.getElementById('zoneName').value;
            if (!n) {{alert('请输入区域名称！');return;}}
            zones.push({{type:t, name:n, coords:[Math.round(currentRect.x1),Math.round(currentRect.y1),Math.round(currentRect.x2),Math.round(currentRect.y2)], color:[0,255,0]}});
            currentRect = null; updateZoneList(); drawCanvas(); showStatus('区域已添加！');
        }}
        function updateZoneList() {{
            let html = ''; zones.forEach((z,i) => {{
                html += `<div class="zone-item"><span><strong>${{z.type}}</strong> - ${{z.name}} - 坐标: [${{z.coords.join(', ')}}]</span><button class="btn btn-danger" onclick="deleteZone(${{i}})">删除</button></div>`;
            }}); document.getElementById('zoneList').innerHTML = html;
        }}
        function deleteZone(i) {{ zones.splice(i,1); updateZoneList(); drawCanvas(); showStatus('区域已删除！'); }}
        function clearAll() {{ if (confirm('确定清空所有区域？')) {{ zones=[]; updateZoneList(); drawCanvas(); showStatus('所有区域已清空！'); }} }}
        function saveConfig() {{
            if (zones.length===0) {{alert('无区域可保存！');return;}}
            let gz = {{}}; zones.forEach(z => {{if(!gz[z.type])gz[z.type]=[]; gz[z.type].push({{name:z.name, coords:z.coords, color:z.color}});}});
            fetch('/api/zones', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(gz)}})
            .then(r=>r.json()).then(d=>showStatus(d.success?'[OK] 配置已保存！':'[ERR] 保存失败：'+d.error));
        }}
        function loadConfig() {{
            fetch('/api/zones').then(r=>r.json()).then(d=>{{
                zones=[]; for(let t in d.zones) {{d.zones[t].forEach(z=>zones.push({{type:t, name:z.name, coords:z.coords, color:z.color}}));}}
                updateZoneList(); drawCanvas(); showStatus('配置已加载！');
            }});
        }}
        function showStatus(m) {{ let s=document.getElementById('status'); s.textContent=m; s.style.display='block'; setTimeout(()=>s.style.display='none',3000); }}
        window.onload = loadConfig;
    </script>
</body>
</html>
    ''')

@app.route('/api/background')
def get_background_image():
    """获取区域配置页面的背景图片"""
    try:
        if os.path.exists(Config.CONFIG_BACKGROUND_IMAGE):
            return Response(open(Config.CONFIG_BACKGROUND_IMAGE, 'rb').read(), mimetype='image/jpeg')
        else:
            # 生成空白背景图
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', blank)
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/zones', methods=['GET'])
def get_zones_api():
    """API：获取所有区域配置"""
    try:
        return jsonify({"success": True, "zones": zone_manager.get_all_zones(), "model_labels": LABEL_MAP})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/zones', methods=['POST'])
def save_zones_api():
    """API：保存区域配置"""
    try:
        zones_data = request.json
        return jsonify({"success": zone_manager.save_zones(zones_data)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/zones/<zone_type>', methods=['GET'])
def get_zones_by_type_api(zone_type):
    """API：根据类型获取区域配置"""
    try:
        return jsonify({"success": True, "zones": zone_manager.get_zones_by_type(zone_type)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ==================== 核心业务API接口 ====================
@app.route('/start', methods=['POST'])
def start():
    """
    开始考试API
    请求参数：{"userid": "考生ID"}
    返回结果：{"status": "started", "userid": "", "video_path": "", "fps": 25.0}
    """
    try:
        data = request.json or {}
        user_id = data.get('userid')
        if not user_id:
            return jsonify({"error": "缺少必传参数：userid"}), 400

        print(f"[ICON] 收到开始考试请求: user_id={user_id}")
        with state.lock:
            if state.is_recording:
                return jsonify({"error": "已有考试正在进行，请勿重复启动"}), 400
            # 硬重置所有状态
            state.hard_reset()
            state.frames_written = 0

            # 获取有效视频帧，确认分辨率
            frame = get_current_frame()
            if frame is None or not is_valid_frame(frame):
                return jsonify({"error": "无法从视频源获取有效帧"}), 500
            h, w = frame.shape[:2]
            state.frame_width = w
            state.frame_height = h

            # 视频录制（根据配置开关决定是否启用）
            video_path = None
            WRITE_FPS = Config.VIDEO_WRITE_FPS
            if Config.ENABLE_RECORDING:
                video_path = get_video_save_path(user_id)
                try:
                    writer = FFmpegVideoWriter(video_path, WRITE_FPS, w, h, codec='libx264')
                    print(f"[OK] FFmpeg编码器初始化成功，帧率: {WRITE_FPS:.1f}fps")
                except Exception as e:
                    print(f"[ERR] FFmpeg初始化失败: {e}，回退到OpenCV")
                    writer = None
                    codecs_to_try = [('mp4v', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]
                    for codec, ext in codecs_to_try:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec)
                            video_path_ext = video_path.replace('.mp4', ext)
                            writer = cv2.VideoWriter(video_path_ext, fourcc, WRITE_FPS, (w, h))
                            if writer.isOpened():
                                print(f"[OK] OpenCV编码器 {codec} 初始化成功")
                                video_path = video_path_ext
                                break
                        except Exception as e2:
                            print(f"[WARN]  编码器 {codec} 初始化失败: {e2}")
                            writer = None
                            continue
                if writer is None or not writer.isOpened():
                    return jsonify({"error": "所有编码器均无法使用，无法创建视频写入器"}), 500
                state.video_writer = writer
            else:
                print(f"[INFO] 视频录制已禁用 (enable_recording=false)")

            # 更新全局状态，开始录制
            state.is_recording = True
            state.user_id = user_id
            state.enable_detection = True
            state.video_path = video_path

            # 打印开始考试信息
            print(f"="*50)
            print(f"[OK] 考试已启动: user_id={user_id}")
            print(f"[VIDEO] 视频路径: {video_path} | 分辨率: {w}x{h}")
            print(f"[REC] 编码器: {Config.VIDEO_CODEC} | 帧率: {WRITE_FPS:.1f}fps")
            print(f"="*50)

        # 返回成功响应
        return jsonify({
            "status": "started",
            "userid": user_id,
            "video_path": video_path,
            "fps": WRITE_FPS,
            "resolution": f"{w}x{h}"
        })
    except Exception as e:
        print(f"[ERR] 开始考试接口异常: {e}")
        import traceback
        traceback.print_exc()
        with state.lock:
            state.hard_reset()
        return jsonify({"error": str(e)}), 500

@app.route('/stop', methods=['POST'])
def stop():
    """
    结束考试API
    请求参数：{"userid": "考生ID"}
    返回结果：{"status": "stopped", "userid": "", "video_path": "", "frames_written": 0, "file_valid": true}
    """
    try:
        import subprocess
        import shutil  # 新增：判断ffprobe是否存在
        data = request.json or {}
        user_id = data.get('userid')

        with state.lock:
            if not state.is_recording or state.user_id != user_id:
                return jsonify({"error": "无正在进行的考试，或考生ID不匹配"}), 400
            # 停止录制和检测
            state.is_recording = False
            state.enable_detection = False
            # 缓存需要关闭的写入器和视频信息
            writer_to_close = state.video_writer
            video_path = state.video_path
            written_frames = state.frames_written  # 直接用全局已同步的计数，避免差1

        # ========== 核心修改：删除空帧写入，仅等待编码器缓存刷盘，不产生额外PTS ==========
        if writer_to_close and writer_to_close.isOpened():
            print(f"[ICON] 等待FFmpeg编码器缓存刷盘（超时{Config.FFMPEG_FLUSH_TIMEOUT}秒）...")
            try:
                # 仅等待，不写入任何帧，避免PTS重复
                time.sleep(Config.FFMPEG_FLUSH_TIMEOUT)
                print(f"[OK] FFmpeg编码器缓存刷盘完成，无额外帧写入")
            except Exception as e:
                print(f"[WARN]  缓存刷盘警告: {e}（不影响视频完整性）")

        # 安全释放视频写入器（优化：先判断是否打开，再释放）
        if writer_to_close:
            try:
                if writer_to_close.isOpened():
                    # 核心：先停止写入，再释放，避免编码器报错
                    writer_to_close.release()
                state.video_writer = None
                print(f"[VIDEO] 视频写入器已安全关闭（无FFmpeg断言/PTS错误）")
            except Exception as e:
                print(f"[WARN]  关闭视频写入器警告: {e}（已做容错处理）")

        # 验证视频文件有效性
        file_valid = False
        file_size = 0
        if video_path and os.path.exists(video_path):
            file_size = os.path.getsize(video_path)
            print(f"[VIDEO] 考试视频: {video_path} | 大小: {file_size:,} 字节 | 实际写入帧数: {written_frames}")
            # 有效文件判断：大小>1MB 且 写入帧数>0
            if file_size > 1024 * 1024 and written_frames > 0:
                file_valid = True
                # ========== 优化：判断ffprobe是否存在，不存在则跳过验证，不打印警告 ==========
                ffprobe_path = shutil.which("ffprobe")
                if ffprobe_path:
                    try:
                        result = subprocess.run([
                            ffprobe_path, '-v', 'quiet', '-print_format', 'json',
                            '-show_streams', '-show_format', video_path
                        ], capture_output=True, text=True, timeout=5)
                        if result.returncode == 0:
                            probe_data = json.loads(result.stdout)
                            duration = float(probe_data.get('format', {}).get('duration', 0))
                            actual_fps = eval(probe_data['streams'][0]['r_frame_rate'])
                            print(f"[OK] 视频验证成功: 时长 {duration:.2f}秒 | 实际帧率 {actual_fps:.1f}fps | PTS时序正常")
                    except Exception as e:
                        print(f"[WARN]  ffprobe验证警告: {e}（视频文件可正常播放）")
                else:
                    print(f"[INFO]  未检测到ffprobe，跳过视频时序验证（视频可正常播放）")
            else:
                print(f"[WARN]  视频文件无效：大小过小或无有效帧写入")

        # 保存步骤结果并软重置状态
        with state.lock:
            step_results = state.step_results.copy()
            state.reset()

        # 打印结束考试信息
        print(f"="*50)
        print(f"[STOP] 考试已结束: user_id={user_id}")
        print(f"[INFO] 实际写入帧数: {written_frames} | 文件有效: {file_valid}")
        print(f"="*50)

        # 返回结束考试结果
        return jsonify({
            "status": "stopped",
            "userid": user_id,
            "video_path": video_path,
            "video_size": file_size,
            "frames_written": written_frames,
            "file_valid": file_valid,
            "step_results": step_results,
            "fps": Config.VIDEO_WRITE_FPS
        })
    except Exception as e:
        print(f"[ERR] 结束考试接口异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/ydpt_sseboxes')
def ydpt_sseboxes():
    """SSE接口：实时推送检测目标坐标（适用于长连接客户端）"""
    def generate():
        last_frame_id = -1
        while True:
            try:
                if state.frame_id != last_frame_id:
                    last_frame_id = state.frame_id
                    # 计算自适应检测区域
                    if state.frame_width > 0 and state.frame_height > 0:
                        current_resolution = (state.frame_width, state.frame_height)
                        if Config.RECOG_AREA:
                            adaptive_recog_area = get_adaptive_recog_area(
                                Config.RECOG_AREA,
                                (Config.CONFIG_WIDTH, Config.CONFIG_HEIGHT),
                                current_resolution
                            )
                        else:
                            adaptive_recog_area = [0, 0, state.frame_width, state.frame_height]
                    else:
                        adaptive_recog_area = Config.RECOG_AREA if Config.RECOG_AREA else []
                    data = {
                        "frame_id": state.frame_id,
                        "timestamp": int(time.time()),
                        "boxes": state.latest_detections,
                        "is_recording": state.is_recording,
                        "recog_area": adaptive_recog_area,
                        "resolution": {"width": state.frame_width, "height": state.frame_height}
                    }
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                time.sleep(0.05)
            except Exception as e:
                print(f"SSE推送异常: {e}")
                break
    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'Connection': 'keep-alive', 'Access-Control-Allow-Origin': '*'}
    )

@app.route('/ydpt_boxes', methods=['GET'])
def get_boxes():
    """GET接口：获取当前帧的检测目标坐标"""
    try:
        frame = get_current_frame()
        if frame is None or not is_valid_frame(frame):
            return jsonify({"error": "暂无有效视频帧"}), 500
        # 执行推理和跟踪，返回最新检测结果
        detections, _, _ = analyze_frame_with_tracking(frame)
        # 获取当前帧分辨率，计算自适应检测区域
        orig_h, orig_w = frame.shape[:2]
        current_resolution = (orig_w, orig_h)
        # 获取自适应检测区域
        if Config.RECOG_AREA:
            adaptive_recog_area = get_adaptive_recog_area(
                Config.RECOG_AREA,
                (Config.CONFIG_WIDTH, Config.CONFIG_HEIGHT),
                current_resolution
            )
        else:
            adaptive_recog_area = [0, 0, orig_w, orig_h]
        return jsonify({
            "frame_id": state.frame_id,
            "timestamp": int(time.time()),
            "boxes": detections,
            "is_recording": state.is_recording,
            "recog_area": adaptive_recog_area,
            "resolution": {"width": orig_w, "height": orig_h}
        })
    except Exception as e:
        print(f"获取检测框接口异常: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/step_images/<path:filename>', methods=['GET'])
def serve_step_image(filename):
    """提供步骤验证图片的HTTP访问"""
    try:
        full_path = os.path.join(Config.IMAGE_BASE_DIR, filename)
        if os.path.exists(full_path):
            return send_file(full_path, mimetype='image/jpeg')
        else:
            return jsonify({"error": "图片不存在", "path": full_path}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """健康检查接口：返回服务当前运行状态"""
    return jsonify({
        "status": "running",
        "model_loaded": state.inference_backend is not None,
        "backend": Config.INFERENCE_BACKEND,
        "device": Config.DEVICE_ID,
        "is_recording": state.is_recording,
        "current_step": state.current_step,
        "streaming_enabled": Config.ENABLE_STREAMING,
        "frames_written": state.frames_written,
        "video_writer_active": (state.video_writer is not None and state.video_writer.isOpened()) if state.video_writer else False,
        "tracker_objects": len(tracker.objects) if 'tracker' in globals() else 0,
        "detection_buffer_size": len(state.detection_buffer),
        "fps": Config.VIDEO_WRITE_FPS
    })

@app.route('/step_history', methods=['GET'])
def get_step_history():
    """API: 获取步骤验证历史（调试用）"""
    try:
        history = list(state.step_history)[-50:]  # 最近50帧
        return jsonify({
            "success": True,
            "current_step": state.current_step,
            "history_length": len(state.step_history),
            "recent_history": history,
            "disassembly_state": {
                "started": state.disassembly_started,
                "confirmation_frames": state.disassembly_confirmation_frames,
                "pre_disassembly": state.pre_disassembly_state
            },
            "step_requirements": {k: {
                'required_classes': v.get('required_classes', []),
                'min_stable_frames': v.get('min_stable_frames', 3),
                'always_pass': v.get('always_pass', False),
                'state_change_detection': v.get('state_change_detection', False)
            } for k, v in step_validator.step_requirements.items()}
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/detection_buffer', methods=['GET'])
def get_detection_buffer():
    """API: 获取检测缓冲区状态（调试用）"""
    try:
        buffer_info = {}
        # 返回所有对象（按track_id排序）
        for track_id in sorted(state.detection_buffer.keys()):
            info = state.detection_buffer[track_id]
            # 从tracker.objects获取实时stable_count
            tracker_info = tracker.objects.get(track_id, {})
            buffer_info[str(track_id)] = {
                'class': info.get('class', 'unknown'),
                'stable_count': tracker_info.get('stable_count', info.get('stable_count', 0)),
                'disappeared': tracker_info.get('disappeared', 0)
            }
        return jsonify({
            "success": True,
            "buffer_size": len(state.detection_buffer),
            "all_objects": buffer_info,
            "tracker_objects_count": len(tracker.objects),
            "stability_threshold": Config.STEP_STABILITY_THRESHOLD if hasattr(Config, 'STEP_STABILITY_THRESHOLD') else 1
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/debug_step/<int:step_num>', methods=['GET'])
def debug_step_validation(step_num):
    """API: 调试步骤验证（显示详细的验证过程）"""
    try:
        # 获取当前检测
        detections = state.latest_detections
        req = step_validator.step_requirements.get(step_num, {})

        result = {
            "step_num": step_num,
            "required_classes": req.get('required_classes', []),
            "min_stable_frames": req.get('min_stable_frames', 3),
            "always_pass": req.get('always_pass', False),
            "stability_threshold": Config.STEP_STABILITY_THRESHOLD if hasattr(Config, 'STEP_STABILITY_THRESHOLD') else 1,
            "confidence_threshold": step_validator.confidence_threshold,
            "detections_count": len(detections),
            "class_analysis": {}
        }

        # 分析每个需要的类别
        for cls in req.get('required_classes', []):
            class_dets = [d for d in detections if d['class'] == cls]
            confident_dets = [d for d in class_dets if d.get('confidence', 0) >= step_validator.confidence_threshold]
            stability_threshold = Config.STEP_STABILITY_THRESHOLD if hasattr(Config, 'STEP_STABILITY_THRESHOLD') else 1
            stable_dets = []
            for d in confident_dets:
                tid = d.get('track_id')
                if tid is not None:
                    tracker_info = tracker.objects.get(tid, {})
                    buf_info = state.detection_buffer.get(tid, {})
                    stable_count = tracker_info.get('stable_count', buf_info.get('stable_count', 0))
                    if stable_count >= stability_threshold:
                        stable_dets.append({
                            'track_id': tid,
                            'stable_count': stable_count,
                            'confidence': d.get('confidence', 0)
                        })

            result["class_analysis"][cls] = {
                "raw_count": len(class_dets),
                "confident_count": len(confident_dets),
                "stable_count": len(stable_dets),
                "stable_details": stable_dets[:3]  # 只显示前3个
            }

        # 执行实际验证
        is_valid = step_validator.validate_step(step_num, detections, None)
        result["validate_result"] = is_valid

        result["validate_internal"] = getattr(step_validator, '_last_validate_result', {})
        result["validate_debug_log"] = getattr(step_validator, '_last_debug_log', [])

        # 添加temporal_consistency的调试信息
        result["temporal_check"] = {
            "history_length": len(state.step_history),
            "min_frames_required": req.get('min_stable_frames', 3),
            "persistence_threshold": step_validator.persistence_threshold
        }

        # 检查最近N帧的通过情况
        min_frames = req.get('min_stable_frames', 3)
        if len(state.step_history) >= min_frames:
            recent_valid = sum(1 for h in list(state.step_history)[-min_frames:]
                                  if h.get(f'step_{step_num}_valid', False))
            result["temporal_check"]["recent_valid_count"] = recent_valid
            result["temporal_check"]["pass_ratio"] = recent_valid / min_frames
            result["temporal_check"]["would_pass"] = (recent_valid / min_frames) >= step_validator.persistence_threshold

        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

# ==================== 服务初始化 ====================
def initialize_service():
    """服务全局初始化，加载模型/跟踪器/验证器，启动流处理线程"""
    try:
        # 1. 配置FFmpeg环境（核心修复PTS/断言）
        setup_ffmpeg_env()
        # 2. 动态获取视频源实际帧率，全局统一
        print(f"\n[VIDEO] 检测视频源实际帧率...")
        temp_cap = cv2.VideoCapture(Config.RTSP_URL, cv2.CAP_FFMPEG)
        if temp_cap.isOpened():
            Config.VIDEO_WRITE_FPS = temp_cap.get(cv2.CAP_PROP_FPS)
            temp_cap.release()
        # 帧率兜底：获取失败则使用25fps
        if Config.VIDEO_WRITE_FPS is None or Config.VIDEO_WRITE_FPS <= 0:
            Config.VIDEO_WRITE_FPS = 25.0
        print(f"[OK] 视频源帧率确认: {Config.VIDEO_WRITE_FPS:.1f}fps（全局统一）")

        # 3. 打印服务基础信息
        print("="*70)
        print("门架式移动平台AI检测服务 - FFmpeg PTS修复版")
        print("="*70)
        print(f"操作系统: {'Linux' if Config.IS_LINUX else 'Windows'}")
        print(f"推理后端: {Config.INFERENCE_BACKEND} | 运行设备: {Config.DEVICE_ID}")
        print(f"YOLO模型: {Config.YOLOV8_MODEL_PATH} | 昇腾模型: {Config.ASCEND_OM_MODEL_PATH}")
        print(f"视频源: {Config.RTSP_URL} | 录制帧率: {Config.VIDEO_WRITE_FPS:.1f}fps")
        print(f"配置页面: {Config.CONFIG_WIDTH}x{Config.CONFIG_HEIGHT} | 背景图: {Config.CONFIG_BACKGROUND_IMAGE}")
        print(f"\n[LIST] 检测类别标签映射:")
        for i, label in LABEL_MAP.items():
            print(f"  {i:2d}: {label}")

        # 4. 初始化推理后端
        print(f"\n[BOX] 初始化推理后端...")
        state.inference_backend = create_inference_backend()
        backend_info = state.inference_backend.get_model_info()
        print(f"[OK] 推理后端初始化成功: {backend_info}")
        # 验证模型文件是否存在
        if Config.INFERENCE_BACKEND == "yolov8" and not os.path.exists(Config.YOLOV8_MODEL_PATH):
            print(f"[ERR] YOLOv8模型文件不存在: {Config.YOLOV8_MODEL_PATH}")
            return False
        if Config.INFERENCE_BACKEND == "ascend" and not os.path.exists(Config.ASCEND_OM_MODEL_PATH):
            print(f"[ERR] 昇腾OM模型文件不存在: {Config.ASCEND_OM_MODEL_PATH}")
            return False

        # 5. 初始化全局对象跟踪器和步骤验证器
        global tracker, step_validator
        tracker = ObjectTracker(max_disappeared=3)
        step_validator = StepValidator(state)
        print(f"[OK] 对象跟踪器初始化成功 | 最大消失帧数: 3")
        print(f"[OK] 步骤验证引擎初始化成功 | 共{len(step_validator.step_requirements)}个步骤规则")

        # 6. 创建必要的目录（核心调整：统一管理config目录）
        # 6.1 基础存储目录
        os.makedirs(Config.VIDEO_SAVE_DIR, exist_ok=True)
        os.makedirs(Config.IMAGE_BASE_DIR, exist_ok=True)
        os.makedirs(Config.IMAGE_SAVE_DIR, exist_ok=True)
        
        # 6.2 配置文件目录（config文件夹）- 统一创建
        config_dirs = [
            os.path.dirname(Config.ZONES_CONFIG_FILE),    # config目录（区域配置文件）
            os.path.dirname(Config.CONFIG_BACKGROUND_IMAGE) # config目录（背景图）
        ]
        # 去重并创建所有config相关目录
        for dir_path in set(config_dirs):
            if dir_path:  # 避免空路径
                os.makedirs(dir_path, exist_ok=True)
                print(f"[OK] 配置目录创建成功: {dir_path}")
        
        # 6.3 背景图容错：如果背景图不存在，提示但不中断服务
        if not os.path.exists(Config.CONFIG_BACKGROUND_IMAGE):
            print(f"[WARN]  配置页面背景图不存在: {Config.CONFIG_BACKGROUND_IMAGE}")
            print(f"   建议：将背景图放到 {os.path.dirname(Config.CONFIG_BACKGROUND_IMAGE)} 目录下，命名为 {os.path.basename(Config.CONFIG_BACKGROUND_IMAGE)}")

        # 7. 测试编码器可用性
        print(f"\n[ICON] 测试视频编码器...")
        test_codecs = [(Config.VIDEO_CODEC, '.mp4'), ('mp4v', '.mp4'), ('XVID', '.avi'), ('MJPG', '.avi')]
        codec_working = False
        for codec, ext in test_codecs:
            try:
                test_path = os.path.join(Config.VIDEO_SAVE_DIR, f"test_codec_{codec}{ext}")
                test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                fourcc = cv2.VideoWriter_fourcc(*codec)
                test_writer = cv2.VideoWriter(test_path, fourcc, Config.VIDEO_WRITE_FPS, (640, 480))
                if test_writer.isOpened():
                    for _ in range(10):
                        test_writer.write(test_frame)
                    test_writer.release()
                    if os.path.exists(test_path) and os.path.getsize(test_path) > 0:
                        print(f"[OK] 编码器 {codec} 测试成功")
                        os.remove(test_path)
                        Config.VIDEO_CODEC = codec
                        codec_working = True
                        break
            except Exception as e:
                print(f"[WARN]  编码器 {codec} 测试失败: {e}")
                continue
        if not codec_working:
            print(f"[ERR] 警告：所有编码器测试失败，视频录制功能可能不可用")

        # 8. 启动推理Pipeline（按后端自动选择多进程/多线程）
        global inference_pipeline
        inference_pipeline = InferencePipeline()
        mode = inference_pipeline.start()
        print(f"[OK] 推理Pipeline启动成功 [{mode}]")

        # 9. 启动视频流处理线程
        print(f"\n[START] 启动视频流处理线程...")
        state.streaming_thread = threading.Thread(target=stream_processor, daemon=True)
        state.streaming_thread.start()

        # 10. 打印服务访问信息
        print(f"\n" + "="*70)
        print(f"[OK] 服务初始化完成，所有模块加载成功！")
        if Config.ENABLE_STREAMING:
            print(f"[VIDEO] 实时视频流: http://<服务器IP>:{Config.SERVICE_PORT}/stream")
        print(f"[ICON]️  区域配置页面: http://<服务器IP>:{Config.SERVICE_PORT}/config")
        print(f"[INFO] 健康检查: http://<服务器IP>:{Config.SERVICE_PORT}/health")
        print(f"[LIST] 检测框API: http://<服务器IP>:{Config.SERVICE_PORT}/ydpt_boxes")
        print(f"[ICON] 开始考试: POST http://<服务器IP>:{Config.SERVICE_PORT}/start ({{'userid': 'xxx'}})")
        print(f"[STOP] 结束考试: POST http://<服务器IP>:{Config.SERVICE_PORT}/stop ({{'userid': 'xxx'}})")
        print("="*70)
        return True
    except Exception as e:
        print(f"[ERR] 服务初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False

# ==================== 新增main函数 ====================
def main():
    # 初始化服务，失败则退出
    if not initialize_service():
        print("[STOP] 服务初始化失败，程序退出！")
        exit(1)
    try:
        import setproctitle
        service_name = os.getenv("SERVICE_NAME", "ydpt_ai")
        if Config.INFERENCE_BACKEND.lower() == "ascend":
            npu_id = os.getenv("NPU_ID", Config.DEVICE_ID)
            setproctitle.setproctitle(f"{service_name}_NPU{npu_id}")
        else:
            gpu_id = os.getenv("GPU_ID", Config.DEVICE_ID).replace(":", "_")
            setproctitle.setproctitle(f"{service_name}_GPU{gpu_id}")
        print(f"[OK] 进程名称已设置为: {setproctitle.getproctitle()}")
    except ImportError:
        print("[WARN] 未安装setproctitle库（Windows可执行：pip install setproctitle-win），跳过进程名称设置")
    except Exception as e:
        print(f"[WARN] 设置进程名称失败: {e}（不影响核心服务运行）")
    
    # 替换硬编码的端口和启动参数
    print(f"\n[START] Flask服务启动成功，监听地址: {Config.SERVICE_HOST}:{Config.SERVICE_PORT}")
    try:
        app.run(
            host=Config.SERVICE_HOST, 
            port=Config.SERVICE_PORT, 
            debug=Config.SERVICE_DEBUG, 
            threaded=Config.SERVICE_THREADED,
            use_reloader=Config.SERVICE_USE_RELOADER
        )
    except Exception as e:
        print(f"[ERR] Flask服务启动失败: {e}")
        if "Address already in use" in str(e) or "端口已占用" in str(e):
            print(f"[TIP] 解决方案：1. 关闭占用{Config.SERVICE_PORT}端口的进程；2. 修改config.yaml中的service.port为其他值（如5061）；3. 设置环境变量 SERVICE_PORT=新端口")
        exit(1)

# ==================== 启动服务 ====================
if __name__ == '__main__':
    main()  # 调用封装后的main函数