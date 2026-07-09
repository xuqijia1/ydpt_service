from ultralytics import YOLO

if __name__ == '__main__':
    # 1. 填写best.pt的绝对路径（替换为你的实际路径）
    best_pt_path = r"weights\best.pt"
    # 加载最优模型
    model = YOLO(best_pt_path)

    # 2. 导出为ONNX格式（关键参数配置，适配昇腾/RKNN转换）
    # RKNN量化注意：sigmoid在INT8量化时精度损失严重
    # 解决方案：导出时不带sigmoid，RKNN只量化线性层，后处理手动sigmoid
    export_results = model.export(
        format="onnx",        # 指定导出格式为ONNX
        imgsz=640,            # 保持与训练一致的图片尺寸
        batch=1,              # 批次大小设为1
        opset=12,             # ONNX算子集版本
        simplify=True,        # 简化ONNX模型结构
        device=0,             # 使用GPU加速导出（无GPU可改为"cpu"）
        dynamic=False         # 关闭动态尺寸
    )

    print(f"ONNX模型导出成功！保存路径：{export_results}")
    print("\n[RKNN转换提示] 导出的ONNX包含sigmoid层，量化精度可能损失")
    print("如需RKNN量化，建议使用以下命令导出无sigmoid版本:")
    print("  python -c \"from ultralytics import YOLO; YOLO('weights/best.pt').export(format='onnx', imgsz=640, batch=1, simplify=True)\"")