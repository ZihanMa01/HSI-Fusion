# HSI-MSI Fusion Benchmark Framework

> 一个通用、可扩展的 HSI-MSI（高光谱-多光谱）图像融合方法测试与对比实验框架。

## 设计目标

- **标准化流程**：统一的数据预处理 → 训练/推理 → 评估 → 可视化流程
- **多方法对比**：轻松集成不同融合方法进行公平对比
- **完整评估体系**：全参考 + 无参考 + 效率 + 可视化评估
- **可复现实验**：配置驱动 + 随机种子控制 + 实验日志
- **易于扩展**：插件式模型注册，新方法只需实现统一接口

---

## 目录结构

```
HSI-Fusion/
├── config/                      # 配置文件（YAML）
│   ├── dataset/                 #   数据集配置
│   │   ├── cave.yaml
│   │   ├── harvard.yaml
│   │   ├── chikusei.yaml
│   │   ├── pavia.yaml
│   │   └── houston.yaml
│   ├── method/                  #   融合方法配置
│   │   ├── baseline_cnn.yaml
│   │   ├── fusionmamba.yaml
│   │   └── ...
│   ├── train.yaml               #   训练参数
│   ├── eval.yaml                #   评估参数
│   └── default.yaml             #   默认主配置
│
├── data/                        # 数据集存放（.gitignore 忽略）
│   ├── cave/
│   ├── harvard/
│   ├── chikusei/
│   ├── pavia/
│   ├── houston/
│   └── simulated/               #   合成退化数据缓存
│
├── src/                         # 核心源代码
│   ├── __init__.py
│   │
│   ├── data/                    # 数据模块
│   │   ├── __init__.py
│   │   ├── base_dataset.py      #   数据集基类
│   │   ├── degradation.py       #   退化模型（PSF / SRF / 噪声）
│   │   ├── transforms.py        #   数据增强与预处理
│   │   ├── dataloader.py        #   DataLoader 构建
│   │   └── registries.py        #   数据集注册表
│   │
│   ├── models/                  # 模型模块
│   │   ├── __init__.py
│   │   ├── base_model.py        #   模型基类（统一接口）
│   │   ├── registry.py          #   模型注册机制
│   │   ├── blocks/              #   通用网络组件
│   │   │   ├── __init__.py
│   │   │   ├── conv.py
│   │   │   ├── attention.py
│   │   │   ├── transformer.py
│   │   │   └── mamba.py
│   │   └── methods/             # ★ 各融合方法实现
│   │       ├── __init__.py
│   │       ├── baseline_cnn/
│   │       │   ├── __init__.py
│   │       │   ├── model.py
│   │       │   └── config.py
│   │       ├── fusionmamba/
│   │       │   ├── __init__.py
│   │       │   ├── model.py
│   │       │   └── config.py
│   │       ├── hypertransformer/
│   │       ├── dstf/
│   │       ├── umsft/
│   │       ├── csakd/
│   │       ├── hsr-kan/
│   │       └── template/        #   新方法模板
│   │           ├── __init__.py
│   │           ├── model.py
│   │           └── config.py
│   │
│   ├── training/                # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py           #   训练器
│   │   ├── losses.py            #   损失函数集
│   │   ├── optimizers.py        #   优化器 & 调度器
│   │   └── callbacks.py         #   训练回调（日志、早停、checkpoint）
│   │
│   ├── evaluation/              # 评估模块
│   │   ├── __init__.py
│   │   ├── metrics.py           #   全参考指标（PSNR, SSIM, SAM, ERGAS, UIQI）
│   │   ├── nr_metrics.py        #   无参考指标（D_lambda, D_s, HQNR, NQE）
│   │   ├── efficiency.py        #   效率指标（FLOPs, Params, 推理时间）
│   │   ├── evaluator.py         #   评估调度器
│   │   └── statistical.py       #   统计分析工具
│   │
│   ├── visualization/           # 可视化模块
│   │   ├── __init__.py
│   │   ├── plot_spectral.py     #   光谱曲线对比
│   │   ├── plot_spatial.py      #   空间/伪彩色图对比
│   │   ├── plot_metrics.py      #   指标对比图（柱状图、雷达图）
│   │   ├── plot_error.py        #   误差图（差值图、热力图）
│   │   ├── plot_ablation.py     #   消融实验图
│   │   └── dashboard.py         #   综合结果看板
│   │
│   └── utils/                   # 工具模块
│       ├── __init__.py
│       ├── logger.py            #   日志记录
│       ├── config.py            #   配置加载与合并
│       ├── reproducibility.py   #   随机种子 & 确定性控制
│       ├── checkpoint.py        #   模型保存/加载
│       ├── tensorboard.py       #   TensorBoard 集成
│       └── file_io.py           #   文件读写工具
│
├── experiments/                 # 实验输出（自动生成）
│   └── {method_name}/
│       └── {timestamp}/
│           ├── config.yaml      #   实验配置（存档）
│           ├── train.log        #   训练日志
│           ├── checkpoints/     #   模型权重
│           ├── tensorboard/     #   TensorBoard 日志
│           ├── results/         #   融合结果（.mat / .png）
│           ├── metrics.json     #   评估指标
│           └── figures/         #   可视化图表
│
├── scripts/                     # 运行脚本
│   ├── train.sh                 #   训练入口
│   ├── eval.sh                  #   评估入口
│   ├── benchmark.sh             #   批量对比实验
│   ├── visualize.sh             #   可视化入口
│   └── download_data.sh         #   数据集下载
│
├── notebooks/                   # Jupyter Notebooks
│   ├── 00_data_exploration.ipynb
│   ├── 01_training_demo.ipynb
│   ├── 02_evaluation_demo.ipynb
│   ├── 03_comparison_analysis.ipynb
│   └── 04_visualization.ipynb
│
├── tests/                       # 单元测试
│   ├── test_metrics.py
│   ├── test_degradation.py
│   ├── test_models.py
│   ├── test_trainer.py
│   └── test_integration.py
│
├── docs/                        # 文档
│   ├── quickstart.md            #   快速开始
│   ├── adding_new_method.md     #   添加新方法指南
│   ├── dataset_preparation.md   #   数据集准备
│   ├── configuration.md         #   配置说明
│   ├── api_reference.md         #   API 参考
│   └── references.md            #   参考文献
│
├── run.py                       # 统一入口
├── setup.py                     # 安装配置
├── requirements.txt             # 依赖列表
├── .gitignore
└── README.md                    # 项目说明
```

---

## 核心设计

### 1. 统一模型接口

```python
class BaseFusionModel(nn.Module):
    """所有融合方法必须继承的基类"""

    @abstractmethod
    def forward(self, lr_hsi: Tensor, hr_msi: Tensor) -> Tensor:
        """输入 LR-HSI 和 HR-MSI，输出融合后的 HR-HSI"""
        pass

    def get_loss(self, pred: Tensor, target: Tensor,
                 lr_hsi: Tensor, hr_msi: Tensor) -> dict:
        """返回损失字典，默认 L1 + Spectral angle loss"""
        pass
```

### 2. 模型注册机制

```python
@register_model("fusionmamba")
class FusionMamba(BaseFusionModel):
    ...

# 自动发现
models = ModelRegistry.list_models()
model  = ModelRegistry.get("fusionmamba")(**config)
```

### 3. 配置驱动

所有实验参数通过 YAML 配置文件管理，保证实验可复现。

### 4. 评估体系

| 类别 | 指标 |
|------|------|
| **全参考** | PSNR, SSIM, SAM, ERGAS, UIQI |
| **无参考** | D_lambda, D_s, HQNR, NQE |
| **效率** | FLOPs, Params, Runtime (ms) |

---

## 使用流程

```bash
# 1. 训练
python run.py --mode train --method fusionmamba --config config/train.yaml

# 2. 评估
python run.py --mode eval --method fusionmamba --checkpoint path/to/ckpt

# 3. 批量对比
python run.py --mode benchmark --methods fusionmamba,hypertransformer,umsft

# 4. 可视化
python run.py --mode visualize --experiment experiments/*
```

---

## 参考项目

- [HyperBench](https://github.com/ritikgshah/HyperBench) — 标准化 HSI 融合基准框架
- [Efficient-MIF](https://github.com/294coder/Efficient-MIF) — 多源图像融合框架
- [FusionMamba](https://github.com/PSRben/FusionMamba) — Mamba 融合（TGRS 2024）
- [HIFToolBox](https://blog.csdn.net/Syuhen/article/details/139836202) — HSI 融合工具箱
- [CSAKD](https://github.com/ming053l/CSAKD) — 知识蒸馏融合（TIP 2024）
- [HyperTransformer](https://github.com/wgcban/HyperTransformer) — Transformer 融合（CVPR 2022）
- [UMSFT](https://github.com/Caoxuheng/UMSFT) — 无监督多层级融合
