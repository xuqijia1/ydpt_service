#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
推理引擎模块 - ydpt_service 门架式移动平台检测服务
参照 ChuanDaiJianCa 的处理方式：
- 检测区域用于过滤检测框，而非裁剪图片
- 整图 resize 到 640x640 进行推理
- 检测区域根据图片分辨率自适应缩放
- 使用中心点判断检测框是否在检测区域内
- 返回检测结果和可视化图像（兼容原有代码）
"""

import cv2
import numpy as np
import time
import os

# ===================== 中文字体预加载 =====================

_PIL_FONT = None
_PIL_FONT_SIZE = 20

def _load_chinese_font():
    """在模块级别预加载中文字体，避免每帧重复加载"""
    global _PIL_FONT
    try:
        from PIL import ImageFont
        font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
        if not os.path.exists(font_path):
            for fp in ['C:\\Windows\\Fonts\\msyh.ttc', 'C:\\Windows\\Fonts\\simhei.ttf']:
                if os.path.exists(fp):
                    font_path = fp
                    break
            else:
                font_path = None
        if font_path:
            _PIL_FONT = ImageFont.truetype(font_path, _PIL_FONT_SIZE)
            print(f"[OK] 中文字体加载成功: {font_path}")
        else:
            print("[WARN] 未找到中文字体文件，将使用英文标签")
    except ImportError:
        print("[WARN] Pillow 未安装，将使用英文标签 (cv2.putText)")
    except Exception as e:
        print(f"[WARN] 中文字体加载失败: {e}")

_load_chinese_font()

# ===================== 类别颜色映射 =====================

CLASS_COLORS = {
    0: (0, 255, 0), 1: (0, 165, 255), 2: (128, 0, 128), 3: (255, 255, 0),
    4: (0, 255, 255), 5: (0, 0, 255), 6: (255, 128, 0), 7: (128, 255, 0),
    8: (255, 0, 128), 9: (0, 128, 255), 10: (128, 128, 0), 11: (255, 0, 255),
    12: (128, 0, 0), 13: (255, 0, 0)
}

# ===================== 工具函数 =====================

def get_adaptive_recog_area(recog_area_config, config_resolution, current_resolution):
    """
    根据当前分辨率自适应检测区域

    Args:
        recog_area_config: 配置的检测区域 [x1, y1, x2, y2]
        config_resolution: 配置的分辨率 (width, height)
        current_resolution: 当前图片分辨率 (width, height)

    Returns:
        自适应后的检测区域 [x1, y1, x2, y2]
    """
    config_w, config_h = config_resolution
    current_w, current_h = current_resolution

    scale_x = current_w / config_w
    scale_y = current_h / config_h

    x1 = int(recog_area_config[0] * scale_x)
    y1 = int(recog_area_config[1] * scale_y)
    x2 = int(recog_area_config[2] * scale_x)
    y2 = int(recog_area_config[3] * scale_y)

    return [x1, y1, x2, y2]


def nms(boxes, scores, iou_threshold=0.45):
    """
    非极大值抑制 (NMS)

    Args:
        boxes: 检测框数组 (N, 4) - [x1, y1, x2, y2]
        scores: 置信度数组 (N,)
        iou_threshold: IoU 阈值

    Returns:
        保留的索引列表
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        # 计算 IoU
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        # 保留 IoU 小于阈值的框
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def is_box_in_recog_area(x1, y1, x2, y2, recog_area):
    """
    判断检测框中心点是否在检测区域内

    Args:
        x1, y1, x2, y2: 检测框坐标
        recog_area: 检测区域 [x1, y1, x2, y2]

    Returns:
        True 如果中心点在检测区域内
    """
    recog_x1, recog_y1, recog_x2, recog_y2 = recog_area
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return not (center_x < recog_x1 or center_y < recog_y1 or center_x > recog_x2 or center_y > recog_y2)


def draw_detection_boxes(image, detections, label_map=None):
    """
    在图像上绘制检测框矩形，收集中文标签信息返回（不渲染文字）
    文字统一由调用方批量Pillow渲染，避免多次全图颜色转换

    Args:
        image: BGR 格式图像
        detections: 检测结果列表
        label_map: 类别名称映射

    Returns:
        (image, texts): image上已画矩形框，texts为[(x, y, label, color_bgr), ...]列表
    """
    if label_map is None:
        label_map = {}

    texts = []

    for det in detections:
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        class_id = det.get("class_id", det.get("class", 0))
        color = CLASS_COLORS.get(class_id, (0, 255, 0))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        class_name = label_map.get(class_id, f"class_{class_id}")
        label = f"{class_name} {det['confidence']:.2f}"
        texts.append((x1, max(15, y1-5), label, color))

    return image, texts


def render_chinese_texts(image, texts, font=_PIL_FONT):
    """
    一次性将所有中文文字渲染到图像上（仅做1次BGR↔RGB转换）

    Args:
        image: BGR格式图像（会被原地修改）
        texts: [(x, y, label, color_bgr), ...]列表
        font: PIL字体对象

    Returns:
        渲染后的图像
    """
    if not texts:
        return image

    if font is not None:
        from PIL import Image as PILImage, ImageDraw as PILDraw
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        draw = PILDraw.Draw(pil_img)
        for tx, ty, label, color_bgr in texts:
            pil_color = (color_bgr[2], color_bgr[1], color_bgr[0])
            draw.text((tx, ty), label, fill=pil_color, font=font)
        img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        image[:] = img_bgr
    else:
        for tx, ty, label, color in texts:
            cv2.putText(image, label, (tx, ty),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image


# ===================== 推理引擎基类 =====================

class BaseInferenceEngine:
    """推理引擎基类"""

    # 配置分辨率（检测区域配置基于此分辨率）
    CONFIG_RESOLUTION = (1920, 1080)

    def __init__(self, model_path, conf_threshold=0.5, recog_area=None, names=None):
        """
        初始化推理引擎

        Args:
            model_path: 模型路径
            conf_threshold: 置信度阈值
            recog_area: 检测区域 [x1, y1, x2, y2]（基于配置分辨率）
            names: 类别名称映射 {class_id: class_name}
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.recog_area = recog_area
        self.names = names or {}
        self._infer_count = 0

    def infer(self, image):
        """
        执行推理

        Args:
            image: BGR 格式图片 (numpy array)

        Returns:
            tuple: (boxes_list, vis_image)
                boxes_list: 检测框列表，每个元素为字典:
                    {
                        'x1': int,          # 原图 x1 坐标
                        'y1': int,          # 原图 y1 坐标
                        'x2': int,          # 原图 x2 坐标
                        'y2': int,          # 原图 y2 坐标
                        'class': int,       # 类别 ID
                        'confidence': float # 置信度
                    }
                vis_image: 可视化图像
        """
        raise NotImplementedError

    def set_recog_area(self, recog_area):
        """设置检测区域"""
        self.recog_area = recog_area

    def set_conf_threshold(self, conf_threshold):
        """设置置信度阈值"""
        self.conf_threshold = conf_threshold

    def _get_adaptive_recog_area(self, image_shape):
        """获取自适应检测区域"""
        h, w = image_shape[:2]
        current_resolution = (w, h)
        if self.recog_area:
            return get_adaptive_recog_area(self.recog_area, self.CONFIG_RESOLUTION, current_resolution)
        return [0, 0, w, h]

    def _filter_by_recog_area(self, boxes, recog_area):
        """按检测区域过滤检测框"""
        filtered = []
        for box in boxes:
            if is_box_in_recog_area(box['x1'], box['y1'], box['x2'], box['y2'], recog_area):
                filtered.append(box)
        return filtered

    def get_model_info(self):
        """获取模型信息"""
        return {
            "model_path": self.model_path,
            "conf_threshold": self.conf_threshold,
            "names": self.names
        }

    def release(self):
        """释放资源"""
        pass


# ===================== CUDA 推理引擎 =====================

class CUDAInferenceEngine(BaseInferenceEngine):
    """CUDA/CPU 推理引擎（使用 YOLOv8）"""

    def __init__(self, model_path, conf_threshold=0.5, recog_area=None, names=None, device='cuda'):
        """
        初始化 CUDA 推理引擎

        Args:
            model_path: YOLO 模型路径 (.pt)
            conf_threshold: 置信度阈值
            recog_area: 检测区域
            names: 类别名称映射
            device: 设备 ('cuda' 或 'cpu')
        """
        super().__init__(model_path, conf_threshold, recog_area, names)
        self.device = device
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        try:
            import torch
            from ultralytics import YOLO

            self.model = YOLO(self.model_path)

            # 检测 CUDA 可用性
            if self.device == 'cuda' and not torch.cuda.is_available():
                print(f"[CUDAInferenceEngine] CUDA 不可用，切换到 CPU")
                self.device = 'cpu'

            # 预热模型
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self.model.predict(dummy, imgsz=640, conf=self.conf_threshold, device=self.device, verbose=False)

            # 如果未提供 names，使用模型的 names
            if not self.names and hasattr(self.model, 'names'):
                self.names = self.model.names

            print(f"[CUDAInferenceEngine] 模型加载成功: {self.model_path} | 设备: {self.device}")
        except Exception as e:
            raise RuntimeError(f"[CUDAInferenceEngine] 模型加载失败: {e}")

    def infer(self, image):
        """执行推理，返回检测结果和可视化图像"""
        t0 = time.time()

        # 获取图片尺寸
        orig_h, orig_w = image.shape[:2]

        # 自适应检测区域
        recog_area = self._get_adaptive_recog_area(image.shape)
        recog_x1, recog_y1, recog_x2, recog_y2 = recog_area

        # 推理（YOLO内部会自动处理resize，保持宽高比）
        results = self.model.predict(image, imgsz=640, conf=self.conf_threshold, device=self.device, verbose=False)

        # 解析结果
        boxes = []
        vis_image = image.copy()

        if results and len(results) > 0 and results[0].boxes is not None:
            result = results[0]
            det_boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            labels = result.boxes.cls.cpu().numpy()

            for box, conf, label in zip(det_boxes, confs, labels):
                orig_x1, orig_y1, orig_x2, orig_y2 = map(int, box)

                # 检测区域过滤（中心点判断）
                center_x = (orig_x1 + orig_x2) / 2
                center_y = (orig_y1 + orig_y2) / 2
                if center_x < recog_x1 or center_y < recog_y1 or center_x > recog_x2 or center_y > recog_y2:
                    continue

                boxes.append({
                    'x1': orig_x1,
                    'y1': orig_y1,
                    'x2': orig_x2,
                    'y2': orig_y2,
                    'class': int(label),
                    'confidence': float(conf)
                })

            # 使用 YOLO 原生可视化
            vis_image = result.plot()

        self._infer_count += 1
        # 仅在检测数量变化时打印关键日志（减少刷屏）
        if self._infer_count % 100 == 0:
            elapsed = (time.time() - t0) * 1000
            # 只在检测到目标时打印
            if len(boxes) > 0:
                print(f"[CUDAInferenceEngine] 推理耗时: {elapsed:.1f}ms | 检测: {len(boxes)}个")

        # CUDA引擎用YOLO原生plot，无额外中文文字列表
        return boxes, vis_image, []

    def get_model_info(self):
        """获取模型信息"""
        return {
            "backend": "YOLOv8",
            "model_path": self.model_path,
            "device": self.device,
            "names": self.names
        }


# ===================== Ascend 推理引擎 =====================

class AscendInferenceEngine(BaseInferenceEngine):
    """Ascend NPU 推理引擎"""

    def __init__(self, model_path, conf_threshold=0.5, recog_area=None, names=None, device_id=0):
        """
        初始化 Ascend 推理引擎

        Args:
            model_path: OM 模型路径
            conf_threshold: 置信度阈值
            recog_area: 检测区域
            names: 类别名称映射
            device_id: Ascend 设备 ID
        """
        super().__init__(model_path, conf_threshold, recog_area, names)
        self.device_id = device_id
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        try:
            from ais_bench.infer.interface import InferSession

            self.session = InferSession(self.device_id, self.model_path)

            # 预分配输入缓冲区
            self._input_buffer = np.zeros((1, 3, 640, 640), dtype=np.float32)

            print(f"[AscendInferenceEngine] 模型加载成功: {self.model_path} | 设备: {self.device_id}")
        except ImportError:
            raise RuntimeError("[AscendInferenceEngine] ais_bench 未安装，无法使用 Ascend 模式")
        except Exception as e:
            raise RuntimeError(f"[AscendInferenceEngine] 模型加载失败: {e}")

    def _preprocess(self, image):
        """预处理：letterbox 保持宽高比 resize 到 640x640 + 归一化"""
        orig_h, orig_w = image.shape[:2]

        # letterbox: 保持宽高比缩放，用灰色填充
        scale = min(640 / orig_w, 640 / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 缩放图像
        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建 640x640 灰色画布，将缩放后的图像放在中心
        img_640 = np.full((640, 640, 3), 114, dtype=np.uint8)  # 114 是 YOLO 默认填充色
        pad_x = (640 - new_w) // 2
        pad_y = (640 - new_h) // 2
        img_640[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized

        # 保存缩放参数供后处理使用
        self._letterbox_scale = scale
        self._letterbox_pad_x = pad_x
        self._letterbox_pad_y = pad_y

        img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)
        img_float = img_rgb.astype(np.float32) * 0.00392156862745098  # 1/255
        self._input_buffer[0] = img_float.transpose(2, 0, 1)
        return self._input_buffer

    def infer(self, image):
        """执行推理，返回检测结果和可视化图像"""
        t0 = time.time()

        # 获取图片尺寸
        orig_h, orig_w = image.shape[:2]

        # 自适应检测区域
        recog_area = self._get_adaptive_recog_area(image.shape)
        recog_x1, recog_y1, recog_x2, recog_y2 = recog_area

        # 预处理
        input_data = self._preprocess(image)

        # 推理
        outputs = self.session.infer([input_data])

        # 后处理
        boxes = self._postprocess(outputs[0], orig_w, orig_h, recog_area)

        # 绘制可视化图像（只画矩形框，文字由stream统一渲染）
        vis_image, det_texts = draw_detection_boxes(image.copy(), boxes, self.names)

        self._infer_count += 1
        # 仅在检测数量变化时打印关键日志（减少刷屏）
        if self._infer_count % 100 == 0:
            elapsed = (time.time() - t0) * 1000
            # 只在检测到目标时打印
            if len(boxes) > 0:
                print(f"[AscendInferenceEngine] 推理耗时: {elapsed:.1f}ms | 检测: {len(boxes)}个")

        return boxes, vis_image, det_texts

    def _postprocess(self, output, orig_w, orig_h, recog_area):
        """后处理：坐标还原 + NMS + 检测区域过滤"""
        predictions = output[0]

        # YOLOv8 输出格式: (num_classes+4, num_boxes) 或 (batch, num_boxes, num_classes+4)
        if len(predictions.shape) == 3:
            predictions = predictions[0]

        # 判断是否需要转置
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose(1, 0)

        det_boxes = predictions[:, :4]
        scores = predictions[:, 4:]

        # sigmoid（如果需要）
        if scores.max() > 1.0:
            scores = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # 阈值过滤
        mask = max_scores >= self.conf_threshold
        if not np.any(mask):
            return []

        filtered_boxes = det_boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = class_ids[mask]

        # xywh -> xyxy（在 640x640 坐标系）
        cx, cy, bw, bh = filtered_boxes.T
        x1_640 = cx - bw / 2
        y1_640 = cy - bh / 2
        x2_640 = cx + bw / 2
        y2_640 = cy + bh / 2

        # 坐标从 640x640 letterbox 还原到原图
        # 先去除 padding，再按 scale 还原
        scale = self._letterbox_scale if hasattr(self, '_letterbox_scale') else min(640 / orig_w, 640 / orig_h)
        pad_x = self._letterbox_pad_x if hasattr(self, '_letterbox_pad_x') else 0
        pad_y = self._letterbox_pad_y if hasattr(self, '_letterbox_pad_y') else 0

        x1_orig = (x1_640 - pad_x) / scale
        y1_orig = (y1_640 - pad_y) / scale
        x2_orig = (x2_640 - pad_x) / scale
        y2_orig = (y2_640 - pad_y) / scale

        # NMS：按类别分组
        unique_classes = np.unique(filtered_classes)
        keep_indices = []

        for cls_id in unique_classes:
            cls_mask = filtered_classes == cls_id
            cls_boxes = np.column_stack((x1_orig[cls_mask], y1_orig[cls_mask], x2_orig[cls_mask], y2_orig[cls_mask]))
            cls_scores = filtered_scores[cls_mask]

            cls_keep = nms(cls_boxes, cls_scores, iou_threshold=0.45)
            global_indices = np.where(cls_mask)[0][cls_keep]
            keep_indices.extend(global_indices)

        if not keep_indices:
            return []

        # 构建结果
        recog_x1, recog_y1, recog_x2, recog_y2 = recog_area
        boxes = []

        for idx in keep_indices:
            orig_x1 = int(x1_orig[idx])
            orig_y1 = int(y1_orig[idx])
            orig_x2 = int(x2_orig[idx])
            orig_y2 = int(y2_orig[idx])

            # 检测区域过滤（中心点判断）
            center_x = (orig_x1 + orig_x2) / 2
            center_y = (orig_y1 + orig_y2) / 2
            if center_x < recog_x1 or center_y < recog_y1 or center_x > recog_x2 or center_y > recog_y2:
                continue

            boxes.append({
                'x1': orig_x1,
                'y1': orig_y1,
                'x2': orig_x2,
                'y2': orig_y2,
                'class': int(filtered_classes[idx]),
                'confidence': float(filtered_scores[idx])
            })

        return boxes

    def get_model_info(self):
        """获取模型信息"""
        return {
            "backend": "Ascend",
            "model_path": self.model_path,
            "device": f"ascend:{self.device_id}",
            "names": self.names
        }

    def release(self):
        """释放资源"""
        if self.session is not None:
            del self.session
            self.session = None
        print("[AscendInferenceEngine] 资源已释放")


# ===================== Rockchip 推理引擎 =====================

class RockchipInferenceEngine(BaseInferenceEngine):
    """Rockchip RKNN 推理引擎 (rknn-toolkit-lite2)"""

    def __init__(self, model_path, conf_threshold=0.5, recog_area=None, names=None, device_id=0, target=None,preprocess_mode="non_quant"):
        """
        初始化 Rockchip 推理引擎

        Args:
            model_path: RKNN 模型路径 (.rknn)
            conf_threshold: 置信度阈值
            recog_area: 检测区域
            names: 类别名称映射
            device_id: NPU 核心 ID (0, 1, 2, ...)
            target: 目标芯片型号 (如 'rk3567')，若为 None 则自动检测
        """
        super().__init__(model_path, conf_threshold, recog_area, names)
        self.device_id = device_id
        self.target = target
        self.preprocess_mode = preprocess_mode
        self._init_model()

    def _init_model(self):
        """初始化模型"""
        try:
            from rknnlite.api import RKNNLite

            self.rknn = RKNNLite()

            # 加载 RKNN 模型
            ret = self.rknn.load_rknn(self.model_path)
            if ret != 0:
                raise RuntimeError(f"加载 RKNN 模型失败: {ret}")

            # 初始化运行时
            # 板端 aarch64: target=None 自动识别本地 NPU
            # core_mask: RKNNLite.NPU_CORE_0_1=3 使用双核, NPU_CORE_0=1 单核
            # RK3576 有2个核心，双核并行可提升吞吐
            core_mask = RKNNLite.NPU_CORE_0_1  # 使用双核提升吞吐
            if isinstance(self.device_id, int) and self.device_id >= 1:
                core_mask = self.device_id
            ret = self.rknn.init_runtime(target=None, core_mask=core_mask)
            if ret != 0:
                raise RuntimeError(f"初始化 RKNN 运行时失败: {ret}")

            # 关键：根据模式设置输入缓冲区类型
            if self.preprocess_mode == "quant":
            # 量化模型：需要 float32 类型
                self._input_buffer = np.zeros((1, 640, 640, 3), dtype=np.float32)
            else:
            # 非量化模型：保持 uint8 类型
                self._input_buffer = np.zeros((1, 640, 640, 3), dtype=np.uint8)

            print(f"[RockchipInferenceEngine] 模型加载成功: {self.model_path} | 目标: {self.target or 'auto'} | 核心: {self.device_id}")
        except ImportError:
            raise RuntimeError("[RockchipInferenceEngine] rknn-toolkit-lite2 未安装，无法使用 Rockchip 模式")
        except Exception as e:
            raise RuntimeError(f"[RockchipInferenceEngine] 模型加载失败: {e}")

    def _preprocess(self, image):
        """
        预处理：letterbox 保持宽高比 resize 到 640x640
        - non_quant 模式：输出 uint8 0~255，模型内做归一化
        - quant 模式：输出 float32 0~1，模型内不做归一化
        """
        orig_h, orig_w = image.shape[:2]

        # letterbox: 保持宽高比缩放，用灰色填充
        scale = min(640 / orig_w, 640 / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 缩放图像
        img_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # 创建 640x640 灰色画布，将缩放后的图像放在中心
        img_640 = np.full((640, 640, 3), 114, dtype=np.uint8)
        pad_x = (640 - new_w) // 2
        pad_y = (640 - new_h) // 2
        img_640[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = img_resized

        # BGR -> RGB (RKNN 模型通常需要 RGB 输入)
        img_rgb = cv2.cvtColor(img_640, cv2.COLOR_BGR2RGB)

        # 保存缩放参数供后处理使用
        self._letterbox_scale = scale
        self._letterbox_pad_x = pad_x
        self._letterbox_pad_y = pad_y

        if self.preprocess_mode == "quant":
        # 量化模型：转为 float32 并归一化到 0~1
            img_rgb = img_rgb.astype(np.float32) / 255.0

        self._input_buffer[0] = img_rgb
        return self._input_buffer

    def infer(self, image):
        """执行推理，返回检测结果和可视化图像"""
        t0 = time.time()

        # 获取图片尺寸
        orig_h, orig_w = image.shape[:2]

        # 自适应检测区域
        recog_area = self._get_adaptive_recog_area(image.shape)
        recog_x1, recog_y1, recog_x2, recog_y2 = recog_area

        # 预处理
        input_data = self._preprocess(image)

        # 推理
        outputs = self.rknn.inference(inputs=[input_data])

        # 后处理
        boxes = self._postprocess(outputs, orig_w, orig_h, recog_area)

        # 绘制可视化图像（只画矩形框，文字由stream统一渲染）
        vis_image, det_texts = draw_detection_boxes(image.copy(), boxes, self.names)

        self._infer_count += 1
        if self._infer_count % 100 == 0:
            elapsed = (time.time() - t0) * 1000
            if len(boxes) > 0:
                print(f"[RockchipInferenceEngine] 推理耗时: {elapsed:.1f}ms | 检测: {len(boxes)}个")

        return boxes, vis_image, det_texts

    def _postprocess(self, outputs, orig_w, orig_h, recog_area):
        """
        后处理：解析 RKNN 输出 + 坐标还原 + NMS + 检测区域过滤

        YOLOv8 RKNN 输出格式: (1, 4+num_classes, 8400)
        前4行: xywh (像素坐标，已缩放)
        后num_classes行: 类别分数 (已 sigmoid)
        不需要额外 sigmoid 或缩放
        """
        if outputs is None or len(outputs) == 0:
            return []

        output = outputs[0]

        # (1, 4+C, 8400) -> (8400, 4+C)
        predictions = output[0].transpose(1, 0)

        # 分离边界框和类别分数
        det_boxes = predictions[:, :4]   # (8400, 4) - xywh 像素坐标
        scores = predictions[:, 4:]       # (8400, 14) - sigmoid 后的类别分数

        max_scores = np.max(scores, axis=1)
        class_ids = np.argmax(scores, axis=1)

        # 阈值过滤
        mask = max_scores >= self.conf_threshold
        if not np.any(mask):
            return []

        filtered_boxes = det_boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_classes = class_ids[mask]

        # xywh -> xyxy（在 640x640 坐标系）
        cx, cy, bw, bh = filtered_boxes.T
        x1_640 = cx - bw / 2
        y1_640 = cy - bh / 2
        x2_640 = cx + bw / 2
        y2_640 = cy + bh / 2

        # 坐标从 640x640 letterbox 还原到原图
        scale = self._letterbox_scale if hasattr(self, '_letterbox_scale') else min(640 / orig_w, 640 / orig_h)
        pad_x = self._letterbox_pad_x if hasattr(self, '_letterbox_pad_x') else 0
        pad_y = self._letterbox_pad_y if hasattr(self, '_letterbox_pad_y') else 0

        x1_orig = (x1_640 - pad_x) / scale
        y1_orig = (y1_640 - pad_y) / scale
        x2_orig = (x2_640 - pad_x) / scale
        y2_orig = (y2_640 - pad_y) / scale

        # NMS：按类别分组
        unique_classes = np.unique(filtered_classes)
        keep_indices = []

        for cls_id in unique_classes:
            cls_mask = filtered_classes == cls_id
            cls_boxes = np.column_stack((x1_orig[cls_mask], y1_orig[cls_mask], x2_orig[cls_mask], y2_orig[cls_mask]))
            cls_scores = filtered_scores[cls_mask]

            cls_keep = nms(cls_boxes, cls_scores, iou_threshold=0.45)
            global_indices = np.where(cls_mask)[0][cls_keep]
            keep_indices.extend(global_indices)

        if not keep_indices:
            return []

        # 构建结果
        recog_x1, recog_y1, recog_x2, recog_y2 = recog_area
        boxes = []

        for idx in keep_indices:
            orig_x1 = int(x1_orig[idx])
            orig_y1 = int(y1_orig[idx])
            orig_x2 = int(x2_orig[idx])
            orig_y2 = int(y2_orig[idx])

            # 检测区域过滤（中心点判断）
            center_x = (orig_x1 + orig_x2) / 2
            center_y = (orig_y1 + orig_y2) / 2
            if center_x < recog_x1 or center_y < recog_y1 or center_x > recog_x2 or center_y > recog_y2:
                continue

            boxes.append({
                'x1': orig_x1,
                'y1': orig_y1,
                'x2': orig_x2,
                'y2': orig_y2,
                'class': int(filtered_classes[idx]),
                'confidence': float(filtered_scores[idx])
            })

        return boxes

    def get_model_info(self):
        """获取模型信息"""
        return {
            "backend": "Rockchip",
            "model_path": self.model_path,
            "target": self.target or "auto",
            "device_id": self.device_id,
            "names": self.names
        }

    def release(self):
        """释放资源"""
        if self.rknn is not None:
            self.rknn.release()
            self.rknn = None
        print("[RockchipInferenceEngine] 资源已释放")


# ===================== 工厂函数 =====================

def create_inference_engine(backend, model_path, conf_threshold=0.5, recog_area=None, names=None, device_id=0, target=None, preprocess_mode=None):
    """
    创建推理引擎

    Args:
        backend: 后端类型 ('yolov8'/'cuda', 'ascend', 'rockchip')
        model_path: 模型路径
        conf_threshold: 置信度阈值
        recog_area: 检测区域
        names: 类别名称映射
        device_id: 设备 ID
        target: 目标芯片型号 (仅 Rockchip 使用，如 'rk3567')
        preprocess_mode: 预处理模式 (仅 Rockchip 使用: 'non_quant'/'quant')

    Returns:
        推理引擎实例
    """
    backend_lower = backend.lower()
    if backend_lower == 'ascend':
        return AscendInferenceEngine(model_path, conf_threshold, recog_area, names, device_id)
    elif backend_lower == 'rockchip':
        return RockchipInferenceEngine(model_path, conf_threshold, recog_area, names, device_id, target, preprocess_mode)
    else:
        return CUDAInferenceEngine(model_path, conf_threshold, recog_area, names, device='cuda')