#!/bin/bash
# 评估入口
# 用法: bash scripts/eval.sh [method_name] [checkpoint_path]

METHOD=${1:-baseline_cnn}
CKPT=${2:-experiments/baseline_cnn/best_model.pth}

echo "=== 评估 ${METHOD} ==="
python run.py --mode eval --method ${METHOD} --checkpoint ${CKPT}
