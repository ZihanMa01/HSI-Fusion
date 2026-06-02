#!/bin/bash
# 训练入口
# 用法: bash scripts/train.sh [method_name] [config_path]

METHOD=${1:-baseline_cnn}
CONFIG=${2:-config/train.yaml}

echo "=== 训练 ${METHOD} ==="
python run.py --mode train --method ${METHOD} --config ${CONFIG}
