#!/bin/bash
set -e

# ATC基础参数固定
SOC_VERSION="Ascend310P3"
INPUT_SHAPE="images:1,3,640,640"
AIPP_CFG_PATH="./weights/aipp.cfg"
FRAMEWORK=5

# 检查aipp配置文件是否存在
if [ ! -f "${AIPP_CFG_PATH}" ]; then
    echo "ERROR: 找不到AIPP配置文件 ${AIPP_CFG_PATH}"
    exit 1
fi

echo "===================== 递归批量 ONNX → AIPP OM 开始 ====================="
# 递归查找所有*.onnx文件
find . -type f -name "*.onnx" | while read -r onnx_file; do
    # 分离目录、文件名、不带后缀名称
    dir=$(dirname "${onnx_file}")
    filename=$(basename "${onnx_file}")
    name="${filename%.onnx}"
    out_file="${dir}/${name}_aipp"

    echo ""
    echo "-------------------------------------------------"
    echo "正在转换: ${onnx_file}"
    echo "输出OM前缀: ${out_file}"
    echo "ATC命令:"
    echo "atc --model=${onnx_file} --output=${out_file} --framework=${FRAMEWORK} --input_shape=${INPUT_SHAPE} --soc_version=${SOC_VERSION} --insert_op_conf=${AIPP_CFG_PATH}"

    # 执行atc转换
    atc \
        --model="${onnx_file}" \
        --output="${out_file}" \
        --framework="${FRAMEWORK}" \
        --input_shape="${INPUT_SHAPE}" \
        --soc_version="${SOC_VERSION}" \
        --insert_op_conf="${AIPP_CFG_PATH}"

    # 判断转换结果
    if [ $? -eq 0 ]; then
        echo "✅ ${filename} 转换成功"
    else
        echo "❌ ${filename} 转换失败"
    fi
done

echo ""
echo "===================== 全部转换任务执行完毕 ====================="
