import os
import yaml
import platform
from typing import Dict, Any

# 定义配置默认值（兜底用，防止配置文件缺失项）
DEFAULT_CONFIG = {
    "inference": {
        "backend": "yolov8",
        "device_id": "cuda:0",
        "yolov8_model_path": "weights/best.pt",
        "ascend_om_model_path": "weights/best_optimized.om",
        "rockchip_model_path": "weights/best.rknn",
        "rockchip_target": "rk3567",
        "rockchip_preprocess_mode": "non_quant",
        "bm_model_path": "weights/best.bmodel"
    },
    "video": {
        "rtsp_url": "test.mp4",
        "write_fps": None,
        "codec": "mp4v"
    },
    "storage": {
        "image_base_dir": "/ai/StepImages",
        "image_sub_dir": "YDPT/{date}/{userid}",
        "video_base_dir": "/ai/ExamVideos",
        "video_sub_dir": "YDPT/{date}/{userid}",
        "video_save_dir": "./videos",
        "image_save_dir": "./images",
        "zones_config_file": "./zones_config.json",
        "config_background_image": "./config_background.jpg"
    },
    "feature": {
        "enable_streaming": True,
        "enable_client_callback": True,
        "enable_recording": True,
        "debug_mode": False
    },
    "detection": {
        "interval": 0.3,
        "iou_threshold": 0.45,
        "conf_threshold": 0.5,
        "config_width": 1920,
        "config_height": 1080,
        "recog_area": None
    },
    "step_validation": {
        "detection_zone": None
    },
    "callback": {
        "step_url": "http://localhost:8070/Data_return",
        "coordinates_url": "http://localhost:8070/coordinates_return"
    },
    "ffmpeg": {
        "disable_auto_pts": True,
        "log_level": "error",
        "flush_timeout": 1.0,
        "valid_frame_min_size": 1024
    },
    "service": {
        "port": 5010,
        "host": "0.0.0.0",
        "threaded": True,
        "debug": False,
        "use_reloader": False
    },
    "system": {
        "cpu_cores": None
    }
}

def load_config(config_path: str = "./config.yaml") -> Dict[str, Any]:
    """
    加载配置文件，优先级：环境变量 > 配置文件 > 默认值
    :param config_path: 配置文件路径
    :return: 合并后的完整配置字典
    """
    # 1. 初始化配置为默认值
    config = deepcopy(DEFAULT_CONFIG)
    
    # 2. 读取配置文件（如果存在）
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
            # 递归合并配置文件到默认配置（只覆盖存在的项）
            merge_config(config, file_config)
            print(f"[OK] 成功加载配置文件: {config_path}")
        except Exception as e:
            print(f"[WARN] 配置文件读取失败，使用默认配置: {e}")
    
    # 3. 适配跨平台路径（Windows/Linux）
    adapt_platform_paths(config)
    
    # 4. 环境变量覆盖（支持如 INFERENCE_BACKEND=yolov8 这样的环境变量）
    override_with_env(config)
    
    # 5. 配置校验（必填项检查）
    validate_config(config)
    
    return config

def merge_config(target: Dict, source: Dict):
    """递归合并两个配置字典（source覆盖target，仅覆盖存在的层级）"""
    for k, v in source.items():
        if k in target and isinstance(target[k], dict) and isinstance(v, dict):
            merge_config(target[k], v)
        elif k in target:
            target[k] = v

def adapt_platform_paths(config: Dict):
    """根据操作系统适配存储路径"""
    is_windows = platform.system() == "Windows"
    if is_windows:
        # Windows路径替换
        config["storage"]["image_base_dir"] = config["storage"]["image_base_dir"].replace("/ai/StepImages", "D:\\StepImages")
        config["storage"]["video_base_dir"] = config["storage"]["video_base_dir"].replace("/ai/ExamVideos", "D:\\ExamVideos")
    # 路径标准化（统一分隔符）
    for key in ["image_base_dir", "video_base_dir", "video_save_dir", "image_save_dir", 
                "zones_config_file", "config_background_image"]:
        path = config["storage"][key]
        config["storage"][key] = os.path.normpath(path)

def override_with_env(config: Dict):
    """通过环境变量覆盖配置（支持层级，用下划线分隔，如 INFERENCE_BACKEND）"""
    env_mapping = {
        # 环境变量名: 配置路径（层级用.分隔）
        "INFERENCE_BACKEND": "inference.backend",
        "DEVICE_ID": "inference.device_id",
        "BM_MODEL_PATH": "inference.bm_model_path",
        "RTSP_URL": "video.rtsp_url",
        "ENABLE_DEBUG": "feature.debug_mode",
        "VIDEO_CODEC": "video.codec",
        "SERVICE_PORT": "service.port",
        "SERVICE_HOST": "service.host"
    }
    for env_name, config_path in env_mapping.items():
        env_value = os.getenv(env_name)
        if env_value is not None:
            # 解析配置路径（如 inference.backend）
            keys = config_path.split(".")
            current = config
            for k in keys[:-1]:
                current = current.get(k, {})
            # 类型转换（布尔值/数字）
            target_value = current[keys[-1]]
            if isinstance(target_value, bool):
                env_value = env_value.lower() in ["true", "1", "yes"]
            elif isinstance(target_value, int):
                env_value = int(env_value) if env_value.isdigit() else target_value
            elif isinstance(target_value, float):
                env_value = float(env_value) if is_float(env_value) else target_value
            # 覆盖配置
            current[keys[-1]] = env_value
            print(f"[ENV] 环境变量 {env_name} 覆盖配置 {config_path} = {env_value}")

def is_float(s: str) -> bool:
    """判断字符串是否为浮点数"""
    try:
        float(s)
        return True
    except:
        return False

def validate_config(config: Dict):
    """校验必填配置项"""
    required_items = [
        ("inference.backend", ["yolov8", "ascend", "bm", "rockchip"]),
        ("video.rtsp_url", None),
        ("inference.yolov8_model_path", None) if config["inference"]["backend"] == "yolov8" else None,
        ("inference.ascend_om_model_path", None) if config["inference"]["backend"] == "ascend" else None,
        ("inference.bm_model_path", None) if config["inference"]["backend"] == "bm" else None,
        ("inference.rockchip_model_path", None) if config["inference"]["backend"] == "rockchip" else None,
        ("service.port", None),
        ("service.host", None)
    ]
    for item, allowed_values in filter(None, required_items):
        keys = item.split(".")
        current = config
        try:
            for k in keys:
                current = current[k]
            # 检查端口合法性
            if item == "service.port":
                if not isinstance(current, int) or current < 1 or current > 65535:
                    raise ValueError(f"端口必须是1-65535的整数，当前值：{current}")
            # 检查值是否在允许范围内
            if allowed_values and current not in allowed_values:
                raise ValueError(f"值 {current} 不在允许列表: {allowed_values}")
        except Exception as e:
            raise RuntimeError(f"配置校验失败 - {item}: {e}")

# 辅助函数：深拷贝（避免修改默认配置）
def deepcopy(obj):
    import copy
    return copy.deepcopy(obj)

# 全局配置对象（加载后供其他模块使用）
global_config = load_config()