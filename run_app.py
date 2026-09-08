from flask import Flask, request, jsonify, Response, render_template_string, stream_with_context, send_file
import os
import cv2
import threading
import time
import json
import requests
import platform
import subprocess
import signal
import sys
from datetime import datetime
from collections import OrderedDict, deque
from pathlib import Path
import numpy as np
import queue
import multiprocessing as mp
import setproctitle

# 导入统一推理引擎模块
from inference_engine import create_inference_engine, draw_detection_boxes, render_chinese_texts, CLASS_COLORS, _PIL_FONT, get_adaptive_recog_area
from exam_state import ExamState

# 导入视频帧处理模块
import video_processor

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

    # ==================== DVPP 硬解码配置（Ascend 后端） ====================
    DVPP_DECODE_ENABLED = global_config.get("dvpp", {}).get("enabled", False)
    DVPP_DEVICE_ID = global_config.get("dvpp", {}).get("device_id", 0)
    DVPP_CHANNEL_ID = global_config.get("dvpp", {}).get("channel_id", None)
    DVPP_EN_TYPE = global_config.get("dvpp", {}).get("en_type", "H265")
    DVPP_OUT_FORMAT = global_config.get("dvpp", {}).get("out_format", "NV12")
    DVPP_AUTO_DETECT_CODEC = global_config.get("dvpp", {}).get("auto_detect_codec", True)
    DVPP_MAX_RECONNECT = global_config.get("dvpp", {}).get("max_reconnect_attempts", 3)
    ASCEND_AIPP = global_config.get("dvpp", {}).get("ascend_aipp", False)

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
            5: {  # 选择检查脚轮（放宽到3个即可，不需要手套）
                'required_classes': ['滑动脚轮'],
                'min_count': {'滑动脚轮': 3},
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
            10: {  # 安全带使用（识别到安全带挂钩即可，去掉挂钩区区域检查）
                'required_classes': ['安全带挂钩'],
                'min_stable_frames': 3,
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
            # return_false: 内部仍返回True让步骤推进，由trigger_step回调时返回False给客户端
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

        注意： 多帧确认已在video_processor.check_step_logic_enhanced()中实现
        这里只做简单检查，避免循环依赖问题
        """
        # 简化逻辑： 只检查历史帧是否足够，不再检查通过率
        # 多帧确认由外层的video_processor.check_step_logic_enhanced()处理
        if len(self.state.step_history) < min_frames:
            return True  # 历史帧不足时临时允许通过
        return True  # 始终返回True，让外层处理多帧确认

class _LockTimeout(Exception):
    """带超时锁获取失败时抛出，供 /start 捕获后快速返回 503。"""
    pass


class _TimedLock:
    """带超时的锁上下文管理器：__enter__ 获取不到锁时抛 _LockTimeout。

    替代 /start 全程持有 state.lock（无超时，三级恢复卡死时永久阻塞），
    确保并发 /start 不会因拿不到锁而无限阻塞，保证服务持续可响应。
    """

    def __init__(self, lock, timeout, name="lock"):
        self._lock = lock
        self._timeout = timeout
        self._name = name
        self._acquired = False

    def __enter__(self):
        self._acquired = self._lock.acquire(timeout=self._timeout)
        if not self._acquired:
            raise _LockTimeout(
                f"系统繁忙，上一次考试启动未结束（等待 {self._timeout:.0f}s 未获得 {self._name}），请稍后重试")
        return self

    def __exit__(self, *exc):
        if self._acquired:
            self._lock.release()
        return False


START_LOCK = threading.Lock()
START_LOCK_TIMEOUT = 10.0


# ==================== 全局状态管理 ====================
class GlobalState:
    """全局状态管理器，统一管理服务所有运行状态"""
    def __init__(self):
        self.exam_state = ExamState.IDLE
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
        self.orig_size = None  # DVPP 路径原始分辨率 (src_h, src_w)，用于坐标缩放
        self.dvpp_decoder = None  # DVPP 解码器引用（reader/soft_reset/退出清理用；不再做服务端录制）
        self.is_healthy = False       # AIPP 流健康标志（IDLE 时 reader 探测维护，/start 非阻塞检查）
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
        self.exam_state = ExamState.IDLE
        self.user_id = None
        # 不重置 is_healthy：它由 reader IDLE 探测维护（见 video_processor.probe_health），
        # /start 非阻塞读快照。此处清 False 会让 /start 在 hard_reset 后读到 False 永远 500。

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
        if video_processor.get_inference_pipeline() is not None:
            video_processor.get_inference_pipeline().request_reset()

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


@app.route('/stream')
def video_stream():
    """浏览器实时视频流入口"""
    if not Config.ENABLE_STREAMING:
        return jsonify({"error": "视频流功能未启用"}), 403
    return Response(video_processor.generate_stream(),
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

        try:
            with _TimedLock(START_LOCK, START_LOCK_TIMEOUT, "START_LOCK"):
                # 短持 state.lock 做状态转换（不再全程持有，避免阻塞 /stop 与并发 /start）
                with state.lock:
                    if state.exam_state != ExamState.IDLE:
                        return jsonify({"error": "已有考试正在进行，请勿重复启动"}), 400
                    state.hard_reset()
                    state.frames_written = 0
                    # 提前设 STARTING 唤醒 reader（IDLE 时 DVPP reader 在做健康探测）
                    state.exam_state = ExamState.STARTING

                use_aipp = getattr(Config, 'ASCEND_AIPP', False)
                use_dvpp = (Config.DVPP_DECODE_ENABLED and
                            Config.INFERENCE_BACKEND.lower() in ("ascend",))

                # 视频源就绪检查（毫秒~秒级，非阻塞；不再做三级恢复）
                ready_frame = None
                if use_dvpp and use_aipp:
                    # AIPP：reader IDLE 探测已维护 is_healthy，直接读快照
                    if not state.is_healthy:
                        with state.lock:
                            state.exam_state = ExamState.IDLE
                        return jsonify({"error": "视频源未就绪，请稍后重试"}), 500
                else:
                    # 非 AIPP（DVPP BGR / CPU）：短超时验证帧可用
                    ready_frame = video_processor.get_current_frame(max_wait=3)
                    if ready_frame is None or not video_processor.is_valid_frame(ready_frame):
                        with state.lock:
                            state.exam_state = ExamState.IDLE
                        return jsonify({"error": "无法从视频源获取有效帧，请检查视频连接"}), 500

                # 分辨率：DVPP 由 reader 用 src_width/src_height 设为源分辨率；CPU 取就绪帧 shape
                if state.dvpp_decoder is not None:
                    w, h = state.frame_width, state.frame_height
                else:
                    h, w = ready_frame.shape[:2]
                    state.frame_width, state.frame_height = w, h

                WRITE_FPS = Config.VIDEO_WRITE_FPS

                # 进入 RUNNING（启动推理；服务端录制已移除，转摄像头刻录机）
                with state.lock:
                    state.exam_state = ExamState.RUNNING
                    state.user_id = user_id
                    state.video_path = None

                print("=" * 50)
                print(f"[OK] 考试已启动: user_id={user_id}")
                print(f"[VIDEO] 分辨率: {w}x{h} | 帧率: {WRITE_FPS:.1f}fps")
                print("=" * 50)

                return jsonify({
                    "status": "started",
                    "userid": user_id,
                    "video_path": None,
                    "fps": WRITE_FPS,
                    "resolution": f"{w}x{h}"
                })
        except _LockTimeout as e:
            return jsonify({"error": str(e)}), 503

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
        data = request.json or {}
        user_id = data.get('userid')

        with state.lock:
            if state.exam_state == ExamState.IDLE or state.user_id != user_id:
                return jsonify({"error": "无正在进行的考试，或考生ID不匹配"}), 400
            # 停止检测，回到待考（reader 回到 IDLE 健康探测；服务端录制已移除）
            state.exam_state = ExamState.IDLE
            video_path = state.video_path
            written_frames = state.frames_written
            step_results = state.step_results.copy()
            state.reset()

        print("=" * 50)
        print(f"[STOP] 考试已结束: user_id={user_id}")
        print("=" * 50)

        # 返回结束考试结果（视频由摄像头刻录机保存，服务端无视频文件）
        return jsonify({
            "status": "stopped",
            "userid": user_id,
            "video_path": video_path,
            "video_size": 0,
            "frames_written": written_frames,
            "file_valid": False,
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
                        "is_recording": state.exam_state == ExamState.RUNNING,
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
        frame = video_processor.get_current_frame()
        if frame is None or not video_processor.is_valid_frame(frame):
            return jsonify({"error": "暂无有效视频帧"}), 500
        # 执行推理和跟踪，返回最新检测结果
        detections, _, _ = video_processor.analyze_frame_with_tracking(frame)
        # 获取当前帧分辨率，计算自适应检测区域
        # DVPP 模式 frame 是 VPC resize 后的 640×640，但 recog_area 配置在 1920×1080 空间
        # 应使用 state.frame_width/height（源分辨率），否则 recog_area 会被错误缩放到 1/3
        orig_w = state.frame_width or frame.shape[1]
        orig_h = state.frame_height or frame.shape[0]
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
            "is_recording": state.exam_state == ExamState.RUNNING,
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
        "is_recording": state.exam_state == ExamState.RUNNING,
        "current_step": state.current_step,
        "streaming_enabled": Config.ENABLE_STREAMING,
        "frames_written": state.frames_written,
        "video_writer_active": False,
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
        # 0. 初始化视频帧处理模块（注入依赖，必须在所有video_processor调用之前）
        # 注意：tracker/step_validator尚未创建，后续通过setter更新
        global tracker, step_validator
        tracker = None
        step_validator = None
        video_processor.init_video_processor(
            config=Config, state=state, label_map=LABEL_MAP,
            step_order=STEP_ORDER, critical_steps=CRITICAL_STEPS,
            tracker=tracker, step_validator=step_validator,
            zone_manager=zone_manager
        )

        # 1. 配置FFmpeg环境（核心修复PTS/断言）
        video_processor.setup_ffmpeg_env()
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
        print(f"视频源: {Config.RTSP_URL} | 处理帧率: {Config.VIDEO_WRITE_FPS:.1f}fps")
        print(f"配置页面: {Config.CONFIG_WIDTH}x{Config.CONFIG_HEIGHT} | 背景图: {Config.CONFIG_BACKGROUND_IMAGE}")

        # 4. 初始化推理后端
        print(f"\n[BOX] 初始化推理后端...")
        state.inference_backend = video_processor.create_inference_backend()
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
        tracker = ObjectTracker(max_disappeared=3)
        step_validator = StepValidator(state)
        # 更新 video_processor 中的 tracker/step_validator 引用
        video_processor._tracker = tracker
        video_processor._step_validator = step_validator
        print(f"[OK] 对象跟踪器初始化成功 | 最大消失帧数: 3")
        print(f"[OK] 步骤验证引擎初始化成功 | 共{len(step_validator.step_requirements)}个步骤规则")

        # 6. 创建必要的目录（核心调整：统一管理config目录）
        # 6.1 基础存储目录
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

        # 8. 启动推理Pipeline（按后端自动选择多进程/多线程）
        pipeline = video_processor.InferencePipeline()
        video_processor.set_inference_pipeline(pipeline)
        mode = pipeline.start()
        print(f"[OK] 推理Pipeline启动成功 [{mode}]")

        # 9. 启动视频流处理线程
        print(f"\n[START] 启动视频流处理线程...")
        state.streaming_thread = threading.Thread(target=video_processor.stream_processor, daemon=True)
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
def _cleanup_on_exit():
    """服务退出时清理 DVPP 解码器等资源"""
    try:
        import video_processor
        # 清理 DVPP 解码器
        if hasattr(video_processor, '_state') and video_processor._state.dvpp_decoder is not None:
            try:
                video_processor._cleanup_dvpp(video_processor._state.dvpp_decoder)
                video_processor._state.dvpp_decoder = None
            except Exception as e:
                print(f"[CLEANUP] DVPP 清理异常: {e}")
    except Exception as e:
        print(f"[CLEANUP] 清理异常: {e}")


def _sigterm_handler(signum, frame):
    print(f"[SIGNAL] 接收到信号 {signum}，触发优雅退出...")
    _cleanup_on_exit()
    sys.exit(0)


def main():
    # 注册信号处理：容器停止时优雅清理
    signal.signal(signal.SIGTERM, _sigterm_handler)

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