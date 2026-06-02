# 快速开始

## 安装

```bash
# 1. 克隆仓库
git clone <repo-url>
cd HSI-Fusion

# 2. 创建虚拟环境
conda create -n hsi-fusion python=3.9
conda activate hsi-fusion

# 3. 安装依赖
pip install -r requirements.txt

# 4. （可选）安装为包
pip install -e .
```

## 下载数据集

```bash
bash scripts/download_data.sh
```

或手动下载并放入 `data/` 目录：
- [CAVE](http://www.cs.columbia.edu/CAVE/databases/multispectral/)
- [Harvard](http://vision.seas.harvard.edu/hyperspec/)
- [Chikusei](https://www.eurac.edu/en/data)

## 训练

```bash
# 使用默认配置训练 baseline_cnn
python run.py --mode train --method baseline_cnn

# 指定配置文件
python run.py --mode train --method fusionmamba --config config/method/fusionmamba.yaml

# 使用脚本
bash scripts/train.sh baseline_cnn
```

## 评估

```bash
python run.py --mode eval --method baseline_cnn --checkpoint experiments/baseline_cnn/best_model.pth
bash scripts/eval.sh baseline_cnn
```

## 批量对比

```bash
python run.py --mode benchmark --methods "baseline_cnn,hypertransformer,umsft"
bash scripts/benchmark.sh "baseline_cnn,hypertransformer,fusionmamba"
```

## 运行测试

```bash
pytest tests/
```
