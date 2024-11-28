#!/bin/bash

# 定义 Miniconda 安装路径
CONDA_PATH="/workspace/SometingElse/Conda/miniconda3"

# 检查 Miniconda 目录是否存在
if [ ! -d "$CONDA_PATH" ]; then
    echo "错误: Miniconda 目录不存在: $CONDA_PATH"
    exit 1
fi

# 检查 activate 文件是否存在
if [ ! -f "$CONDA_PATH/bin/activate" ]; then
    echo "错误: activate 文件不存在: $CONDA_PATH/bin/activate"
    exit 1
fi

# 激活 conda 环境
source "$CONDA_PATH/bin/activate"

# 验证 conda 是否可用
if command -v conda >/dev/null 2>&1; then
    echo "Conda 环境已成功激活"
    conda --version
else
    echo "错误: Conda 激活失败"
    exit 1
fi