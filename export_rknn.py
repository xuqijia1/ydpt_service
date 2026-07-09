#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RKNN 模型转换脚本 - 将 ONNX 模型转换为 RKNN 格式 (用于 Rockchip NPU)
适用于 RK3567 / RK3588 / RK3576 等 Rockchip 芯片

使用方法:
    1. 确保 ONNX 模型已通过 export_onnx.py 导出
    2. 准备量化数据集文件 dataset.txt (每行一个图片路径，用于 INT8 量化)
    3. 运行: python export_rknn.py

依赖:
    PC 端安装 rknn-toolkit2: pip install rknn-toolkit2

注意:
    - 此脚本需要在 PC 端运行，生成的 .rknn 文件复制到板端使用
    - 板端推理使用 rknn-toolkit-lite2 (轻量版)
"""

import os
import sys

# ===================== 配置区域 =====================
# ONNX 模型路径
ONNX_MODEL_PATH = "weights/best.onnx"

# 输出 RKNN 模型路径
RKNN_MODEL_PATH = "weights/best.rknn"

# 目标芯片型号 (rk3567 / rk3588 / rk3576 / rk3568 / rv1126 等)
TARGET_PLATFORM = "rk3576"

# 模型输入尺寸
IMG_SIZE = 640

# 是否启用 INT8 量化 (推荐开启，可大幅提升推理速度)
DO_QUANTIZATION = True

# 量化数据集文件路径 (每行一个图片路径)
# 如果不量化，可以设为 None
DATASET_FILE = "dataset.txt"

# 类别数量 (根据实际模型调整)
NUM_CLASSES = 14

# 归一化参数 (需要与训练时一致)
# YOLOv8 默认: mean=[0, 0, 0], std=[255, 255, 255] (即除以255)
MEAN_VALUES = [[0, 0, 0]]
STD_VALUES = [[255, 255, 255]]

# 量化输出类型
# "fp16": 输出层强制半精度浮点，保护sigmoid精度不损失（推荐）
# "i8": 全INT8量化，sigmoid精度损失大，需后处理手动sigmoid
QUANT_OUTPUT_TYPE = "fp16"

# ===================== 转换函数 =====================

def generate_dataset_file(image_dir, output_file="dataset.txt", num_images=200):
    """
    从图片目录生成量化数据集文件

    Args:
        image_dir: 图片目录路径
        output_file: 输出的数据集文件路径
        num_images: 最多使用的图片数量
    """
    import glob

    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(image_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(image_dir, ext.upper())))

    if len(image_paths) == 0:
        print(f"[ERROR] 在 {image_dir} 中未找到图片文件")
        return False

    # 随机选择指定数量的图片
    import random
    if len(image_paths) > num_images:
        image_paths = random.sample(image_paths, num_images)

    with open(output_file, "w") as f:
        for path in image_paths:
            f.write(path + "\n")

    print(f"[OK] 已生成量化数据集文件: {output_file} (共 {len(image_paths)} 张图片)")
    return True


def export_rknn():
    """执行 ONNX -> RKNN 转换"""
    try:
        from rknn.api import RKNN
    except ImportError:
        print("[ERROR] 未安装 rknn-toolkit2，请执行: pip install rknn-toolkit2")
        return False

    # 检查 ONNX 模型是否存在
    if not os.path.exists(ONNX_MODEL_PATH):
        print(f"[ERROR] ONNX 模型不存在: {ONNX_MODEL_PATH}")
        print("[TIP] 请先运行 export_onnx.py 导出 ONNX 模型")
        return False

    # 检查量化数据集
    if DO_QUANTIZATION and not os.path.exists(DATASET_FILE):
        print(f"[WARN] 量化数据集文件不存在: {DATASET_FILE}")
        print("[INFO] 请准备量化数据集或设置 DO_QUANTIZATION=False")
        return False

    # 创建 RKNN 对象
    rknn = RKNN(verbose=True)

    # 配置模型参数
    print(f"\n[INFO] 开始配置 RKNN 模型...")
    print(f"  - 目标平台: {TARGET_PLATFORM}")
    print(f"  - 输入尺寸: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  - INT8 量化: {'启用' if DO_QUANTIZATION else '禁用'}")

    # 设置配置
    # mean_values 和 std_values 用于归一化: (x - mean) / std
    # YOLOv8: x / 255，所以 mean=[0,0,0], std=[255,255,255]
    # quant_output_type="fp16": 输出层用半精度浮点，保护sigmoid精度
    rknn.config(
        mean_values=MEAN_VALUES,
        std_values=STD_VALUES,
        target_platform=TARGET_PLATFORM,
        quantization_algorithm="normal",
        quant_input_type="u8",
        quant_output_type=QUANT_OUTPUT_TYPE
    )

    # 加载 ONNX 模型
    print(f"\n[INFO] 加载 ONNX 模型: {ONNX_MODEL_PATH}")
    ret = rknn.load_onnx(model=ONNX_MODEL_PATH)
    if ret != 0:
        print(f"[ERROR] 加载 ONNX 模型失败: {ret}")
        return False
    print("[OK] ONNX 模型加载成功")

    # 构建 RKNN 模型
    print(f"\n[INFO] 构建 RKNN 模型...")
    if DO_QUANTIZATION:
        print(f"  - 使用量化数据集: {DATASET_FILE}")
        ret = rknn.build(do_quantization=True, dataset=DATASET_FILE)
    else:
        ret = rknn.build(do_quantization=False)

    if ret != 0:
        print(f"[ERROR] 构建 RKNN 模型失败: {ret}")
        return False
    print("[OK] RKNN 模型构建成功")

    # 导出 RKNN 模型
    print(f"\n[INFO] 导出 RKNN 模型: {RKNN_MODEL_PATH}")
    ret = rknn.export_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print(f"[ERROR] 导出 RKNN 模型失败: {ret}")
        return False
    print("[OK] RKNN 模型导出成功")

    # 获取模型信息
    print(f"\n[INFO] 模型信息:")
    print(f"  - 输出路径: {os.path.abspath(RKNN_MODEL_PATH)}")
    print(f"  - 文件大小: {os.path.getsize(RKNN_MODEL_PATH) / 1024 / 1024:.2f} MB")

    # 释放资源
    rknn.release()

    print(f"\n{'='*50}")
    print("[SUCCESS] RKNN 模型转换完成！")
    print(f"{'='*50}")
    print(f"\n[NEXT] 请将 {RKNN_MODEL_PATH} 复制到板端使用")
    print(f"[NEXT] 修改 config.yaml: backend='rockchip', rockchip_model_path='weights/best.rknn'")

    return True


def verify_rknn_on_pc():
    """
    在 PC 端验证 RKNN 模型 (模拟推理)
    注意: PC 端模拟结果可能与板端实际结果有差异
    """
    try:
        from rknn.api import RKNN
        import numpy as np
    except ImportError:
        print("[ERROR] 未安装 rknn-toolkit2")
        return False

    if not os.path.exists(RKNN_MODEL_PATH):
        print(f"[ERROR] RKNN 模型不存在: {RKNN_MODEL_PATH}")
        return False

    rknn = RKNN()

    # 加载 RKNN 模型
    ret = rknn.load_rknn(RKNN_MODEL_PATH)
    if ret != 0:
        print(f"[ERROR] 加载 RKNN 模型失败: {ret}")
        return False

    # 初始化运行时 (PC 模拟器)
    ret = rknn.init_runtime(target=None)  # None 表示使用模拟器
    if ret != 0:
        print(f"[ERROR] 初始化运行时失败: {ret}")
        return False

    # 创建测试输入
    test_input = np.random.randint(0, 255, (1, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    # 执行推理
    outputs = rknn.inference(inputs=[test_input])

    if outputs is not None:
        print(f"[OK] 模拟推理成功，输出形状: {[o.shape for o in outputs]}")
    else:
        print("[ERROR] 模拟推理失败")
        return False

    rknn.release()
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ONNX -> RKNN 模型转换工具")
    parser.add_argument("--onnx", type=str, default=ONNX_MODEL_PATH, help="ONNX 模型路径")
    parser.add_argument("--output", type=str, default=RKNN_MODEL_PATH, help="输出 RKNN 模型路径")
    parser.add_argument("--target", type=str, default=TARGET_PLATFORM, help="目标芯片型号")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE, help="模型输入尺寸")
    parser.add_argument("--no-quant", action="store_true", help="禁用 INT8 量化")
    parser.add_argument("--dataset", type=str, default=DATASET_FILE, help="量化数据集文件路径")
    parser.add_argument("--gen-dataset", type=str, metavar="IMAGE_DIR", help="从图片目录生成量化数据集")
    parser.add_argument("--verify", action="store_true", help="在 PC 端验证 RKNN 模型")

    args = parser.parse_args()

    # 更新全局配置
    ONNX_MODEL_PATH = args.onnx
    RKNN_MODEL_PATH = args.output
    TARGET_PLATFORM = args.target
    IMG_SIZE = args.img_size
    DO_QUANTIZATION = not args.no_quant
    DATASET_FILE = args.dataset

    if args.gen_dataset:
        # 生成量化数据集
        generate_dataset_file(args.gen_dataset, DATASET_FILE)
    elif args.verify:
        # 验证 RKNN 模型
        verify_rknn_on_pc()
    else:
        # 执行转换
        export_rknn()
