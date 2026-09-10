#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
视频帧处理模块 - 从 run_app.py 提取
包含：FFmpeg环境配置、帧校验、推理管线、视频流处理、步骤检查、回调发送、MJPEG流生成
"""

import os
import cv2
import time
import queue
import threading
import numpy as np
import multiprocessing as mp
from datetime import datetime
from collections import OrderedDict

from exam_state import ExamState
from inference_engine import (
    create_inference_engine, draw_detection_boxes,
    render_chinese_texts, CLASS_COLORS, _PIL_FONT
)

# ==================== 模块级依赖（由 init_video_processor 注入） ====================
_config = None          # Config 类
_state = None           # GlobalState 单例
_label_map = None       # LABEL_MAP
_step_order = None      # STEP_ORDER
_critical_steps = None  # CRITICAL_STEPS
_tracker = None         # ObjectTracker
_step_validator = None  # StepValidator
_zone_manager = None    # ZoneConfigManager

# 模块级单例
_callback_sender = None
_inference_pipeline = None


def init_video_processor(config, state, label_map, step_order, critical_steps,
                         tracker, step_validator, zone_manager):
    """
    注入依赖，由 run_app.py 在 initialize_service() 中调用

    Args:
        config: Config 类（含所有配置属性）
        state: GlobalState 单例
        label_map: LABEL_MAP 字典
        step_order: STEP_ORDER 字典
        critical_steps: CRITICAL_STEPS 列表
        tracker: ObjectTracker 实例
        step_validator: StepValidator 实例
        zone_manager: ZoneConfigManager 实例
    """
    global _config, _state, _label_map, _step_order, _critical_steps
    global _tracker, _step_validator, _zone_manager
    global _callback_sender

    _config = config
    _state = state
    _label_map = label_map
    _step_order = step_order
    _critical_steps = critical_steps
    _tracker = tracker
    _step_validator = step_validator
    _zone_manager = zone_manager

    # 创建异步回调发送器
    _callback_sender = CallbackSender()

    print("[OK] video_processor 模块初始化完成")


# ==================== FFmpeg 环境配置 ====================

def setup_ffmpeg_env():
    """配置FFmpeg环境变量，禁用自动PTS、控制日志级别，避免断言崩溃"""
    if _config.FFMPEG_DISABLE_AUTO_PTS:
        os.environ["OPENCV_FFMPEG_WRITE_NO_AUTO_PTS"] = "1"
    os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = _config.FFMPEG_LOG_LEVEL
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "loglevel;error"
    print(f"[OK] FFmpeg环境配置完成: 禁用自动PTS={_config.FFMPEG_DISABLE_AUTO_PTS}, 日志级别={_config.FFMPEG_LOG_LEVEL}")


def is_valid_frame(frame):
    """校验帧有效性，过滤空帧/损坏帧/极小帧，避免触发FFmpeg断言"""
    if frame is None:
        return False
    if frame.ndim != 3 or frame.shape[-1] != 3:
        return False
    h, w = frame.shape[:2]
    if h < 100 or w < 100:
        return False
    if frame.nbytes < _config.FFMPEG_VALID_FRAME_MIN_SIZE:
        return False
    return True

# ==================== 路径工具函数 ====================

def get_image_save_path(step_name):
    """生成步骤验证图片的保存路径（跨平台）"""
    try:
        date_str = datetime.now().strftime("%Y%m%d")
        time_str = datetime.now().strftime("%H%M%S")
        dir_path = os.path.join(
            _config.IMAGE_BASE_DIR,
            _config.IMAGE_SUB_DIR.format(date=date_str, userid=_state.user_id)
        )
        file_name = f"{step_name}_{time_str}.jpg"
        full_path = os.path.join(dir_path, file_name)
        os.makedirs(dir_path, exist_ok=True)
        return full_path.replace("\\", "/")
    except Exception as e:
        print(f"生成图片路径失败: {e}")
        fallback_path = os.path.join(_config.IMAGE_BASE_DIR, f"{step_name}_{int(time.time())}.jpg")
        return fallback_path.replace("\\", "/")


def select_step_frame(vis_frame, bgr_frame=None, use_aipp=False):
    """Choose a real BGR frame for step evidence in AIPP mode."""
    if use_aipp:
        return bgr_frame
    return vis_frame


def unpack_inference_result(result):
    """Normalize inference results from thread and process workers."""
    if len(result) == 2:
        detections, vis_frame = result
        return detections, vis_frame, None
    return result

# ==================== 推理后端创建 ====================

def create_inference_backend():
    """工厂函数：根据配置创建对应的推理后端"""
    backend = _config.INFERENCE_BACKEND.lower()
    if backend == "yolov8":
        model_path = _config.YOLOV8_MODEL_PATH
    elif backend == "ascend":
        model_path = _config.ASCEND_OM_MODEL_PATH
    elif backend == "rockchip":
        model_path = _config.ROCKCHIP_MODEL_PATH
    else:
        model_path = _config.YOLOV8_MODEL_PATH
    return create_inference_engine(
        backend=_config.INFERENCE_BACKEND,
        model_path=model_path,
        conf_threshold=_config.CONF_THRESHOLD,
        recog_area=_config.RECOG_AREA,
        names=_label_map,
        device_id=_config.DEVICE_ID,
        target=_config.ROCKCHIP_TARGET if backend == "rockchip" else None,
        preprocess_mode=_config.ROCKCHIP_PREPROCESS_MODE if backend == "rockchip" else None,
        aipp=getattr(_config, 'ASCEND_AIPP', False)
    )


# ==================== 检测核心逻辑 ====================

def analyze_frame_with_tracking(frame):
    """分析视频帧，执行推理+对象跟踪，返回带跟踪ID的检测结果

    Args:
        frame: BGR numpy 数组，或 AIPP 模式下的 device buffer dict
               dict: {'buffer': dev_nv12_ptr, 'size': int}
    """
    try:
        boxes_list, vis_image, det_texts = _state.inference_backend.infer(frame, orig_size=getattr(_state, 'orig_size', None))
        for box in boxes_list:
            class_id = int(box["class"]) if isinstance(box["class"], (int, np.integer)) else 0
            box["class_id"] = class_id
            box["class"] = _label_map.get(class_id, f"类别_{class_id}")
        tracked_detections = _tracker.update(boxes_list)
        for det in tracked_detections:
            track_id = det.get('track_id')
            if track_id is not None:
                _state.detection_buffer[track_id] = {
                    'box': [det['x1'], det['y1'], det['x2'], det['y2']],
                    'class': det['class'],
                    'stable_count': _tracker.objects.get(track_id, {}).get('stable_count', 0)
                }
        _state.frame_id += 1
        _state.latest_detections = tracked_detections
        return tracked_detections, vis_image, det_texts
    except Exception as e:
        import traceback
        print(f"推理失败: {e}")
        traceback.print_exc()
        return _state.latest_detections, frame, []


def check_step_logic_enhanced(detections, frame):
    """增强版步骤检查逻辑（使用StepValidator + 多帧确认机制）"""
    if _state.exam_state != ExamState.RUNNING:
        if not hasattr(check_step_logic_enhanced, '_warned'):
            check_step_logic_enhanced._warned = True
            print("[WARN] 步骤检测已禁用 (exam_state != RUNNING)")
        return
    current_time = time.time()
    if current_time - _state.last_detection_time < _config.DETECTION_INTERVAL:
        return
    _state.last_detection_time = current_time

    if not hasattr(check_step_logic_enhanced, '_local_frame_count'):
        check_step_logic_enhanced._local_frame_count = 0
    check_step_logic_enhanced._local_frame_count += 1
    local_frame_id = check_step_logic_enhanced._local_frame_count

    step_to_trigger = None
    with _state.lock:
        try:
            current_step = _state.current_step
            step_num = current_step + 1
            frame_result = {
                'frame_id': local_frame_id,
                'timestamp': current_time,
                'detections_count': len(detections),
                f'step_{step_num}_valid': False
            }
            is_valid = _step_validator.validate_step(step_num, detections, frame)
            frame_result[f'step_{step_num}_valid'] = is_valid
            _state.step_history.append(frame_result)
            if is_valid:
                min_stable_frames = _step_validator.step_requirements.get(step_num, {}).get('min_stable_frames', 3)
                recent_frames = list(_state.step_history)[-min_stable_frames:]
                if len(recent_frames) >= min_stable_frames:
                    consecutive_valid = all(
                        h.get(f'step_{step_num}_valid', False)
                        for h in recent_frames
                    )
                    if consecutive_valid:
                        step_to_trigger = (step_num, True)
                        print(f"[STEP] 步骤 {step_num} ({_step_order.get(step_num, '未知')}) 确认通过！连续{min_stable_frames}帧验证成功")
            user_id = _state.user_id
        except Exception as e:
            print(f"步骤检查异常: {e}")
            import traceback
            traceback.print_exc()

    if step_to_trigger:
        step_num, result = step_to_trigger
        trigger_step(step_num, frame, result, user_id)
        time.sleep(0.05)


def trigger_step(step_num, frame, result, user_id):
    """触发步骤完成，更新状态+保存验证图片+回调客户端"""
    step_name = _step_order[step_num]
    # return_false步骤：内部通过（推进步骤），但回调客户端时返回False
    req = _step_validator.step_requirements.get(step_num, {})
    client_result = False if req.get('return_false', False) else result
    with _state.lock:
        if step_name in _state.step_results and _state.step_results[step_name] is not None:
            return
        _state.step_results[step_name] = client_result
        _state.current_step = step_num
        _state.step_completed[step_name] = {
            'time': time.time(),
            'order': len(_state.step_completed) + 1
        }
        img_path = get_image_save_path(step_name) if (result and frame is not None) else None

    if img_path and frame is not None:
        try:
            cv2.imwrite(img_path, frame)
            print(f"[OK] 步骤{step_num}验证图片已保存: {img_path}")
        except Exception as e:
            print(f"保存步骤图片失败: {e}")
            img_path = None

    _callback_sender.send_step_result(step_name, client_result, img_path, user_id)
    print(f"[OK] 步骤 {step_num} ({step_name}) 完成: 内部={'通过' if result else '未完成'}, 客户端={'通过' if client_result else '未完成'}")


# ==================== 客户端回调 ====================

def send_to_client(step_name, result, img_path, user_id):
    """将步骤完成结果回调到客户端服务"""
    try:
        img_url = img_path if img_path else ""
        data = {
            "userid": user_id,
            "wearfanghu": False, "checkhuanjing": False, "weilanopen": False, "toolbag": False,
            "checkjiaolun": False, "setjiaolun": False, "setjiaocha": False, "setjiaoshouban": False,
            "setpati": False, "setanquandai": False, "setfanghulan": False, "chaichu": False,
            "safetyopt": False, "weilanclose": False, "img_url": img_url
        }
        if step_name in data:
            data[step_name] = result
        if _config.ENABLE_CLIENT_CALLBACK:
            try:
                import requests
                response = requests.post(_config.STEP_CALLBACK_URL, json=data, timeout=3)
                print(f"[ICON] 客户端步骤回调成功: {response.status_code}")
            except Exception as e:
                print(f"[ICON] 客户端步骤回调失败: {e}")
        else:
            print(f"[ICON] 模拟回调客户端: {step_name} = {result}, 图片路径={img_path}")
    except Exception as e:
        print(f"构造回调数据失败: {e}")


def send_coordinates_to_client(coordinate_data, user_id):
    """将检测目标坐标数据实时回调到客户端"""
    if _state.exam_state != ExamState.RUNNING:
        return
    try:
        coordinate_data["userid"] = user_id
        headers = {"Content-Type": "application/json; charset=utf-8"}
        import requests
        requests.post(
            url=_config.COORDINATES_CALLBACK_URL,
            json=coordinate_data,
            headers=headers,
            timeout=1
        )
    except Exception:
        pass  # 连接错误/超时静默处理


class CallbackSender:
    """异步回调发送器：专用线程+队列，解耦HTTP I/O与推理消费循环"""
    def __init__(self, maxsize=100):
        self._queue = queue.Queue(maxsize=maxsize)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        while True:
            try:
                task = self._queue.get(timeout=0.5)
                task_type = task[0]
                if task_type == 'coord':
                    _, data, user_id = task
                    send_coordinates_to_client(data, user_id)
                elif task_type == 'step':
                    _, step_name, result, img_path, user_id = task
                    send_to_client(step_name, result, img_path, user_id)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CallbackSender] 执行异常: {e}")

    def _enqueue(self, task):
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
        self._queue.put_nowait(task)

    def send_coordinates(self, coordinate_data, user_id):
        self._enqueue(('coord', coordinate_data, user_id))

    def send_step_result(self, step_name, result, img_path, user_id):
        self._enqueue(('step', step_name, result, img_path, user_id))


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
        "userid": ""
    }
    if recog_area:
        result["recog_area"] = recog_area
    return result


# ==================== 推理 Pipeline ====================

def _inference_process_main(frame_queue, result_queue, running_event, reset_event, _unused=False):
    """YOLOv8子进程：全流程推理+后处理+跟踪（spawn模式，绕过GIL）"""
    try:
        # 子进程重新加载配置（spawn模式不继承父进程模块变量）
        from config_loader import load_config
        proc_config = load_config()
        backend_type = proc_config["inference"]["backend"].lower()
        if backend_type == "yolov8":
            model_path = proc_config["inference"]["yolov8_model_path"]
        elif backend_type == "ascend":
            model_path = proc_config["inference"]["ascend_om_model_path"]
        elif backend_type == "rockchip":
            model_path = proc_config["inference"]["rockchip_model_path"]
        else:
            model_path = proc_config["inference"]["yolov8_model_path"]
        label_map = {
            0: "安全帽", 1: "安全带", 2: "围栏", 3: "标识牌", 4: "手套",
            5: "滑动脚轮", 6: "门架", 7: "交叉杆", 8: "脚手板", 9: "爬梯",
            10: "防护栏", 11: "安全带挂钩", 12: "扫把", 13: "人"
        }
        backend = create_inference_engine(
            backend=backend_type,
            model_path=model_path,
            conf_threshold=proc_config["detection"].get("conf_threshold", 0.5),
            recog_area=proc_config["detection"].get("recog_area"),
            names=label_map,
            device_id=proc_config["inference"]["device_id"],
            target=proc_config["inference"].get("rockchip_target") if backend_type == "rockchip" else None,
            preprocess_mode=proc_config["inference"].get("rockchip_preprocess_mode") if backend_type == "rockchip" else None
        )
        from run_app import ObjectTracker
        local_tracker = ObjectTracker(max_disappeared=3)
        orig_size = None  # YOLOv8 子进程不使用 DVPP，orig_size 由 Ascend 主线程设置
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
            boxes_list, vis_image, _ = backend.infer(frame, orig_size=orig_size)
            for box in boxes_list:
                class_id = int(box["class"]) if isinstance(box["class"], (int, np.integer)) else 0
                box["class_id"] = class_id
                box["class"] = label_map.get(class_id, f"类别_{class_id}")
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
        self._is_thread_mode = _config.INFERENCE_BACKEND.lower() in ("ascend", "rockchip")
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

    def submit_frame(self, frame, step_frame=None):
        try:
            while self.frame_queue.full():
                try: self.frame_queue.get_nowait()
                except: break
            if self._is_thread_mode:
                # AIPP 模式下 frame 是 device buffer dict，不需要 copy
                if isinstance(frame, dict):
                    # The inference engine returns a black visualization for NV12 input.
                    # Keep the matching host BGR frame for step evidence and screenshots.
                    evidence_frame = step_frame.copy() if step_frame is not None else None
                    self.frame_queue.put_nowait((frame, evidence_frame))
                else:
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
        """Ascend：主进程中做推理+后处理+跟踪

        AIPP 模式下，frame_queue 中的 item 可能是：
        - numpy array (BGR 帧，非 AIPP 模式)
        - dict {'buffer': dev_ptr, 'size': int} (NV12 device buffer，AIPP 零拷贝)
        """
        global _tracker
        while self._running_flag:
            if self._reset_flag:
                if _tracker is not None:
                    _tracker.objects.clear()
                    _tracker.next_id = 0
                self._reset_flag = False
            try:
                task = self.frame_queue.get(timeout=0.1)
            except:
                continue
            try:
                if isinstance(task, tuple):
                    frame, evidence_frame = task
                else:
                    frame, evidence_frame = task, None
                # frame 可能是 numpy array 或 AIPP device buffer dict，直接传给推理引擎
                detections, vis_frame, det_texts = analyze_frame_with_tracking(frame)
                _state.latest_det_texts = det_texts
                try:
                    while self.result_queue.full():
                        try: self.result_queue.get_nowait()
                        except: break
                    self.result_queue.put_nowait((detections, vis_frame, evidence_frame))
                except:
                    pass
            except:
                continue

    def _result_worker(self):
        use_aipp = getattr(_config, 'ASCEND_AIPP', False)
        while True:
            try:
                result = self.result_queue.get(timeout=0.1)
                detections, vis_frame, evidence_frame = unpack_inference_result(result)
                for det in detections:
                    track_id = det.get('track_id')
                    if track_id is not None:
                        _state.detection_buffer[track_id] = {
                            'box': [det['x1'], det['y1'], det['x2'], det['y2']],
                            'class': det['class'],
                            'stable_count': det.get('stable_count', 0)
                        }
                # AIPP 模式下 vis_frame 是黑色零数组，不覆盖 latest_vis_frame
                # generate_stream() 会从 frame_queue 拿 BGR 帧并自己画框
                if not use_aipp:
                    _state.latest_vis_frame = vis_frame
                _state.latest_detections = detections
                if _state.user_id and _config.ENABLE_CLIENT_CALLBACK:
                    csharp_data = build_csharp_coordinate_data(detections, recog_area=_config.RECOG_AREA)
                    _callback_sender.send_coordinates(csharp_data, _state.user_id)
                step_frame = select_step_frame(vis_frame, evidence_frame, use_aipp)
                check_step_logic_enhanced(detections, step_frame)
            except:
                continue


def get_inference_pipeline():
    """获取全局推理管线实例"""
    return _inference_pipeline


def set_inference_pipeline(pipeline):
    """设置全局推理管线实例"""
    global _inference_pipeline
    _inference_pipeline = pipeline


def probe_health(decoder, use_aipp):
    """AIPP 健康探测：拉一帧验证流可解，维护 _state.is_healthy。

    IDLE 时由 reader 线程周期性调用，/start 非阻塞读 _state.is_healthy 快照。
    read_frame_aipp 内部自释放 VDEC buffer 并返回预分配复用 NV12 buffer，
    调用方无需 dvpp_free，不会泄漏 NPU 内存。
    """
    if not (decoder and decoder.is_started and use_aipp):
        return _state.is_healthy
    try:
        nv12_info, _ = decoder.read_frame_aipp()
        if nv12_info is not None:
            _state.is_healthy = True
            return True
    except Exception:
        pass
    _state.is_healthy = False
    return False


# ==================== 视频流处理 ====================

def _stream_processor_cv():
    """视频流处理 - CPU软解码路径（cv2.VideoCapture，原有逻辑）"""
    cap = None
    last_inference_frame_id = -1
    TARGET_FPS = _config.VIDEO_WRITE_FPS
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    last_frame_time = 0

    while True:
        try:
            if cap is None or not cap.isOpened():
                print(f"[VIDEO] 连接视频源: {_config.RTSP_URL}")
                cap = cv2.VideoCapture(_config.RTSP_URL, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    print("[ERR] 视频源连接失败，5秒后重试...")
                    time.sleep(5)
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                _state.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                _state.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                TARGET_FPS = _config.VIDEO_WRITE_FPS
                FRAME_INTERVAL = 1.0 / TARGET_FPS
                print(f"[OK] 视频流连接成功: {_state.frame_width}x{_state.frame_height} | 源帧率: {source_fps:.1f}fps | 处理帧率: {TARGET_FPS:.1f}fps")
                last_frame_time = time.time()

            current_time = time.time()
            if current_time - last_frame_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - (current_time - last_frame_time))
                continue
            last_frame_time = time.time()

            ret, frame = cap.read()
            if not ret or not is_valid_frame(frame):
                if os.path.isfile(_config.RTSP_URL):
                    print("[VIDEO] 视频文件播放完毕，重新播放...")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    print(f"[WARN]  获取无效帧/断帧，重新连接视频源...")
                    cap.release()
                    cap = None
                    time.sleep(2)
                    continue

            _dispatch_frame(frame)

        except Exception as e:
            print(f"视频流处理错误: {e}")
            import traceback
            traceback.print_exc()
            if cap:
                cap.release()
            cap = None
            time.sleep(5)


def _stream_processor_dvpp():
    """视频流处理 - DVPP 硬解码路径（Ascend 环境，失败持续重试，不降级 cv2）

    AIPP 模式下使用 read_frame_device_nv12() 零拷贝推理，
    非 AIPP 模式使用 read_frame() BGR 帧推理。
    """
    from dvpp_decoder import create_dvpp_decoder
    decoder = None
    TARGET_FPS = _config.VIDEO_WRITE_FPS
    FRAME_INTERVAL = 1.0 / TARGET_FPS
    last_frame_time = 0
    reconnect_count = 0
    # Ascend 环境不允许回退 cv2，max_reconnect 提升至 10，递增等待 2/5/8/15/30/30/.../30s
    max_reconnect = getattr(_config, 'DVPP_MAX_RECONNECT', 10)
    use_aipp = getattr(_config, 'ASCEND_AIPP', False)
    _last_probe_time = 0.0
    _probe_fail_count = 0

    while True:
        try:
            if decoder is None or not decoder.is_started:
                if reconnect_count >= max_reconnect:
                    # Ascend 环境不回退 cv2，长睡后重置计数继续重试硬解码
                    print(f"[DVPP] 连续 {max_reconnect} 次连接失败，等待 60s 后继续重试硬解码（不降级 cv2）")
                    time.sleep(60)
                    reconnect_count = 0
                    continue

                print(f"[DVPP] 连接视频源: {_config.RTSP_URL} (尝试 {reconnect_count + 1}/{max_reconnect})")
                try:
                    decoder = create_dvpp_decoder(
                        rtsp_url=_config.RTSP_URL,
                        device_id=_config.DVPP_DEVICE_ID,
                        channel_id=getattr(_config, 'DVPP_CHANNEL_ID', None),
                        en_type=_config.DVPP_EN_TYPE,
                        auto_detect_codec=getattr(_config, 'DVPP_AUTO_DETECT_CODEC', True),
                    )
                except RuntimeError as e:
                    print(f"[ERR] DVPP 硬解码创建失败: {e}")
                    decoder = None
                    reconnect_count += 1
                    wait_time = min(5 * reconnect_count, 30)
                    print(f"[DVPP] {wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                if decoder is None:
                    reconnect_count += 1
                    wait_time = min(5 * reconnect_count, 30)
                    print(f"[ERR] DVPP 解码器创建失败，{wait_time}秒后重试...")
                    time.sleep(wait_time)
                    continue
                reconnect_count = 0
                # 优先 src_width/src_height（DVPP 源分辨率 1920×1080），回退 width/height（cv2 模式）
                # 检测框经 orig_size 反推后已回到源分辨率空间，frame_width/height 必须与之一致
                _state.frame_width = getattr(decoder, 'src_width', None) or decoder.width
                _state.frame_height = getattr(decoder, 'src_height', None) or decoder.height
                _state.dvpp_decoder = decoder
                if hasattr(decoder, 'src_height'):
                    _state.orig_size = (decoder.src_height, decoder.src_width)
                else:
                    _state.orig_size = None
                TARGET_FPS = _config.VIDEO_WRITE_FPS
                FRAME_INTERVAL = 1.0 / TARGET_FPS
                decoder_type = type(decoder).__name__
                aipp_str = " | AIPP零拷贝" if use_aipp else ""
                print(f"[OK] DVPP 硬解码连接成功: {decoder.width}x{decoder.height} | "
                      f"解码器: {decoder_type}{aipp_str} | 处理帧率: {TARGET_FPS:.1f}fps")
                last_frame_time = time.time()

            current_time = time.time()
            if current_time - last_frame_time < FRAME_INTERVAL:
                time.sleep(FRAME_INTERVAL - (current_time - last_frame_time))
                continue
            last_frame_time = time.time()

            # 待考状态（IDLE）：AIPP 模式做健康探测维护 is_healthy，断流自愈；不读帧/不推理
            # demux 线程仍在后台拉流保活，RTSP 不会断；/start 时 exam_state=STARTING 立即唤醒
            if _state.exam_state == ExamState.IDLE:
                if use_aipp and decoder is not None and decoder.is_started:
                    _now = time.time()
                    if _now - _last_probe_time >= 2.0:
                        _last_probe_time = _now
                        _was_healthy = _state.is_healthy
                        probe_health(decoder, use_aipp)
                        if not _state.is_healthy:
                            _probe_fail_count += 1
                            # 连续 3 次探测失败才重连（避免偶发抖动），之后每 3 次再试
                            if _probe_fail_count >= 3 and _probe_fail_count % 3 == 0:
                                print(f"[DVPP] IDLE 健康探测失败 {_probe_fail_count} 次，触发软恢复（保留 VDEC 通道）")
                                try:
                                    if hasattr(decoder, '_reconnect_demux'):
                                        decoder._reconnect_demux()
                                    else:
                                        decoder.soft_reset()
                                except Exception as e:
                                    print(f"[DVPP] 软恢复异常: {e}")
                        else:
                            if not _was_healthy:
                                print("[DVPP] IDLE 健康探测恢复")
                            _probe_fail_count = 0
                time.sleep(0.5)
                continue

            if use_aipp:
                # AIPP 零拷贝路径：同时获取 NV12 device buffer（推理）和 BGR 帧（显示）
                nv12_info, bgr_frame = decoder.read_frame_aipp()
                if nv12_info is None:
                    # 纯软恢复：_reconnect_demux 重启 demux（保留 VDEC 通道，绝不 release）
                    print("[WARN] DVPP AIPP 读取帧失败，软恢复（保留 VDEC 通道）...")
                    try:
                        if hasattr(decoder, '_reconnect_demux'):
                            decoder._reconnect_demux()
                        else:
                            decoder.soft_reset()
                    except Exception as e:
                        print(f"[DVPP] 软恢复异常: {e}")
                    time.sleep(2)
                    continue

                # AIPP 模式：提交 device buffer 给推理管线（零拷贝）
                if _state.inference_backend and _state.exam_state == ExamState.RUNNING and _state.user_id:
                    if _inference_pipeline is not None:
                        _inference_pipeline.submit_frame(nv12_info, step_frame=bgr_frame)

                # BGR 帧用于显示
                if bgr_frame is not None:
                    try:
                        while _state.frame_queue.full():
                            _state.frame_queue.get_nowait()
                        _state.frame_queue.put_nowait(bgr_frame.copy())
                    except:
                        pass
                    try:
                        while _state.display_queue.full():
                            _state.display_queue.get_nowait()
                        _state.display_queue.put_nowait(bgr_frame.copy())
                    except:
                        pass
            else:
                # 非 AIPP 路径：读取 BGR 帧
                frame = decoder.read_frame()
                if frame is None:
                    # 纯软恢复：_reconnect_demux 重启 demux（保留 VDEC 通道，绝不 release）
                    print("[WARN] DVPP 读取帧失败，软恢复（保留 VDEC 通道）...")
                    try:
                        if hasattr(decoder, '_reconnect_demux'):
                            decoder._reconnect_demux()
                    except Exception as e:
                        print(f"[DVPP] 软恢复异常: {e}")
                    time.sleep(2)
                    continue
                if not is_valid_frame(frame):
                    continue

                _dispatch_frame(frame)

        except Exception as e:
            print(f"DVPP 视频流处理错误: {e}")
            import traceback
            traceback.print_exc()
            _cleanup_dvpp(decoder)
            decoder = None
            _state.dvpp_decoder = None
            reconnect_count += 1
            if reconnect_count >= max_reconnect:
                # Ascend 环境不回退 cv2，长睡后重置计数继续重试硬解码
                print(f"[DVPP] 连续 {max_reconnect} 次异常，等待 60s 后继续重试硬解码（不降级 cv2）")
                time.sleep(60)
                reconnect_count = 0
                continue
            time.sleep(5)


def _cleanup_dvpp(decoder):
    """清理 Ascend-FFmpeg 资源"""
    try:
        if decoder:
            decoder.release()
    except Exception as e:
        print(f"[Ascend-FFmpeg] 清理异常: {e}")


def _dispatch_frame(frame):
    """帧分发：放入推理/显示队列，提交推理"""
    try:
        while _state.frame_queue.full():
            _state.frame_queue.get_nowait()
        _state.frame_queue.put_nowait(frame.copy())
    except:
        pass

    if _state.inference_backend and _state.exam_state == ExamState.RUNNING and _state.user_id:
        if _inference_pipeline is not None:
            _inference_pipeline.submit_frame(frame)

    try:
        while _state.display_queue.full():
            _state.display_queue.get_nowait()
        _state.display_queue.put_nowait(frame.copy())
    except:
        pass


def stream_processor():
    """视频流核心处理线程 - 根据配置选择 CPU软解码 或 DVPP硬解码

    Ascend + DVPP_DECODE_ENABLED：必须使用硬解码，失败持续重试，**不降级 cv2**
    其他环境：使用 cv2 软解码作为兼容方案
    """
    use_dvpp = (_config.DVPP_DECODE_ENABLED and
                _config.INFERENCE_BACKEND.lower() in ("ascend",))
    if use_dvpp:
        print("[OK] 视频流模式: DVPP 硬解码 (Ascend，不降级 cv2)")
        _stream_processor_dvpp()
    else:
        print("[OK] 视频流模式: CPU 软解码 (cv2.VideoCapture)")
        _stream_processor_cv()


def analyze_and_check(frame):
    """后台异步执行：帧推理+步骤检查+坐标回调"""
    try:
        if _state.exam_state != ExamState.RUNNING:
            return
        detections, vis_frame, det_texts = analyze_frame_with_tracking(frame)
        _state.latest_det_texts = det_texts
        _state.latest_vis_frame = vis_frame
        if _state.user_id and _config.ENABLE_CLIENT_CALLBACK:
            csharp_data = build_csharp_coordinate_data(detections, recog_area=_config.RECOG_AREA)
            _callback_sender.send_coordinates(csharp_data, _state.user_id)
        check_step_logic_enhanced(detections, vis_frame)
    except Exception as e:
        print(f"异步检测/步骤检查异常: {e}")


def get_current_frame(max_wait=5):
    """从帧队列获取当前视频帧，支持等待重连"""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        try:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            return _state.frame_queue.get(timeout=min(remaining, 1.0))
        except:
            continue
    return None


def generate_stream():
    """生成浏览器可播放的MJPEG实时视频流

    AIPP 模式下：latest_vis_frame 不更新，从 frame_queue 取 BGR 帧自己画框
    非 AIPP 模式下：用 latest_vis_frame（推理引擎已画好框）
    """
    CONFIG_W, CONFIG_H = _config.CONFIG_WIDTH, _config.CONFIG_HEIGHT
    last_frame_time = 0
    TARGET_FPS = 15
    JPEG_QUALITY = 80
    STREAM_MAX_WIDTH = 960
    last_vis_frame = None
    use_aipp = getattr(_config, 'ASCEND_AIPP', False)

    def get_cached_zones():
        current_time = time.time()
        if _state.zones_cache is None or current_time - _state.zones_cache_timestamp > 5:
            _state.zones_cache = _zone_manager.get_all_zones()
            _state.zones_cache_timestamp = current_time
        return _state.zones_cache

    while True:
        try:
            current_time = time.time()
            if current_time - last_frame_time < 1.0 / TARGET_FPS:
                time.sleep(0.001)
                continue
            last_frame_time = current_time

            if use_aipp:
                # AIPP 模式：从 frame_queue 取 BGR 帧，自己画检测框
                frame = get_current_frame(max_wait=0.5)
                if frame is None:
                    if last_vis_frame is not None:
                        vis_frame = last_vis_frame.copy()
                    else:
                        time.sleep(0.05)
                        continue
                else:
                    vis_frame = frame.copy()
                # 画检测框（坐标基于原始分辨率，需缩放到 vis_frame 尺寸）
                detections = _state.latest_detections
                orig_size = getattr(_state, 'orig_size', None)
                if orig_size is not None:
                    orig_h, orig_w = orig_size
                else:
                    orig_h, orig_w = vis_frame.shape[:2]
                det_texts = []
                for det in detections:
                    x1, y1, x2, y2 = det['x1'], det['y1'], det['x2'], det['y2']
                    # 原始分辨率坐标 → vis_frame 坐标
                    sx = vis_frame.shape[1] / orig_w
                    sy = vis_frame.shape[0] / orig_h
                    fx1, fy1 = int(x1 * sx), int(y1 * sy)
                    fx2, fy2 = int(x2 * sx), int(y2 * sy)
                    class_id = det.get('class_id', 0)
                    color = CLASS_COLORS.get(class_id, (0, 255, 0))
                    cv2.rectangle(vis_frame, (fx1, fy1), (fx2, fy2), color, 2)
                    label = f"{det.get('class', '')} {det.get('confidence', 0):.2f}"
                    det_texts.append((fx1, max(15, fy1-5), label, color))
            else:
                # 非 AIPP 模式：用推理引擎画好框的 vis_frame
                vis_frame = _state.latest_vis_frame
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
                det_texts = None  # 使用 _state.latest_det_texts
            last_vis_frame = vis_frame.copy()

            if vis_frame.shape[1] > STREAM_MAX_WIDTH:
                scale = STREAM_MAX_WIDTH / vis_frame.shape[1]
                vis_frame = cv2.resize(vis_frame, (STREAM_MAX_WIDTH, int(vis_frame.shape[0] * scale)),
                                       interpolation=cv2.INTER_LINEAR)

            all_zones = get_cached_zones()
            frame_h, frame_w = vis_frame.shape[:2]
            scale_x = frame_w / CONFIG_W if CONFIG_W > 0 else 1.0
            scale_y = frame_h / CONFIG_H if CONFIG_H > 0 else 1.0

            all_texts = []

            if use_aipp:
                # AIPP 模式：det_texts 坐标已在 vis_frame 尺寸上，需缩放到显示分辨率
                # vis_frame 已被 resize 到 STREAM_MAX_WIDTH，所以按比例缩放
                if det_texts and last_vis_frame is not None:
                    # vis_frame 已 resize，det_texts 坐标基于 resize 前的 vis_frame
                    pre_scale = vis_frame.shape[1] / last_vis_frame.shape[1] if last_vis_frame.shape[1] != vis_frame.shape[1] else 1.0
                    if abs(pre_scale - 1.0) > 0.01:
                        all_texts.extend([(int(tx * pre_scale), int(ty * pre_scale), label, color)
                                          for tx, ty, label, color in det_texts])
                    else:
                        all_texts.extend(det_texts)
                else:
                    all_texts.extend(det_texts)
            else:
                # 非 AIPP 模式：使用推理引擎的 det_texts
                if vis_frame.shape[1] != 1920 and _state.latest_det_texts:
                    text_scale = vis_frame.shape[1] / 1920
                    for tx, ty, label, color in _state.latest_det_texts:
                        all_texts.append((int(tx * text_scale), int(ty * text_scale), label, color))
                else:
                    all_texts.extend(_state.latest_det_texts)

            for zone_type, zones in all_zones.items():
                for zone in zones:
                    try:
                        coords = zone['coords']
                        color = tuple(zone.get('color', [0, 255, 0]))
                        x1, y1 = int(coords[0]*scale_x), int(coords[1]*scale_y)
                        x2, y2 = int(coords[2]*scale_x), int(coords[3]*scale_y)
                        overlay = vis_frame.copy()
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                        cv2.addWeighted(overlay, 0.1, vis_frame, 0.9, 0, vis_frame)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                        all_texts.append((x1, max(15, y1-5), f"{zone['name']} ({zone_type})", color))
                    except:
                        pass

            for det in _state.latest_detections:
                track_id = det.get('track_id')
                if track_id:
                    x1, y1 = det['x1'], det['y1']
                    # 坐标基于原始分辨率，缩放到当前 vis_frame 尺寸
                    if use_aipp and orig_size is not None:
                        oh, ow = orig_size
                        x1 = int(x1 * vis_frame.shape[1] / ow)
                        y1 = int(y1 * vis_frame.shape[0] / oh)
                    elif vis_frame.shape[1] != 1920:
                        ts = vis_frame.shape[1] / 1920
                        x1, y1 = int(x1*ts), int(y1*ts)
                    cv2.putText(vis_frame, f"ID:{track_id}", (x1, y1-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)

            status_text = f"当前步骤:{_state.current_step} | exam_state:{_state.exam_state} | 已写帧数:{_state.frames_written}"
            all_texts.append((10, 30, status_text, (0, 255, 0)))

            render_chinese_texts(vis_frame, all_texts, _PIL_FONT)

            encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
            _, buffer = cv2.imencode('.jpg', vis_frame, encode_params)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"视频流生成错误: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.5)
