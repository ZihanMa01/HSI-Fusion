#!/bin/bash
# 批量对比实验
# 用法: bash scripts/benchmark.sh "method1,method2,method3"

METHODS=${1:-"baseline_cnn"}

echo "=== 批量对比: ${METHODS} ==="
python run.py --mode benchmark --methods ${METHODS}
