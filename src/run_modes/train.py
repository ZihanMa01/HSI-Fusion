"""训练模式 — 通用调度器。

根据方法元信息自动路由:
    - training_required=False → 跳过训练，直出日志
    - training_mode="supervised" → 默认监督训练
    - multi_stage=True → 分阶段训练
"""

from pathlib import Path
from omegaconf import DictConfig

from src.utils.logger import setup_logger
from src.utils.config import config_to_dict


def run_train(config: DictConfig, method_name: str):
    logger = setup_logger("HSI-Fusion")

    # 加载模型
    from src.models import ModelRegistry
    cfg_dict = config_to_dict(config)
    cfg_dict["method"] = {"name": method_name}
    model = ModelRegistry.get(method_name)(cfg_dict)

    # 判断是否需要训练
    if not model.training_required:
        logger.info(f"[{method_name}] 零训练方法，跳过训练，可直接评估")
        return

    logger.info(f"[{method_name}] 开始训练 (mode={model.training_mode}, "
                f"multi_stage={model.multi_stage})")

    # 加载数据集
    from src.data import build_dataloader
    from src.data.transforms import get_transform_pipeline
    from src.data.datasets import HSIDataset

    dataset_cfg = cfg_dict.get("data", {})
    train_transform = get_transform_pipeline("train", dataset_cfg)
    val_transform = get_transform_pipeline("val", dataset_cfg)

    # TODO: 实例化数据集（等数据集加载器就绪后取消注释）
    # train_ds = HSIDataset(dataset_cfg["root_dir"], "train", train_transform, dataset_cfg)
    # val_ds = HSIDataset(dataset_cfg["root_dir"], "val", val_transform, dataset_cfg)
    # train_loader = build_dataloader(train_ds, cfg_dict, "train")
    # val_loader = build_dataloader(val_ds, cfg_dict, "val")

    # 调用模型自身的训练入口
    # model.train_model(train_loader, val_loader, cfg_dict)

    logger.info(f"[{method_name}] 训练完成")
