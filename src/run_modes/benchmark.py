"""批量对比模式 — 统一调度多种类型方法。

自动处理:
    - 零训练方法（CNMF 等）→ 直接推理
    - 监督方法 → 检查 checkpoint → 推理
    - 混合类型 → 统一输出对比表
"""

from pathlib import Path
from typing import List
from omegaconf import DictConfig

from src.utils.logger import setup_logger
from src.utils.config import config_to_dict


def run_benchmark(config: DictConfig, methods: List[str]):
    logger = setup_logger("HSI-Fusion")
    cfg_dict = config_to_dict(config)

    from src.models import ModelRegistry

    models = {}
    for name in methods:
        model_cls = ModelRegistry.get(name)
        model = model_cls(cfg_dict)

        need_train = model.training_required
        logger.info(f"[{name}] 已加载 (training_required={need_train})")

        # 如果需要训练的，尝试加载 checkpoint
        if need_train:
            ckpt_path = Path(f"experiments/{name}/best_model.pth")
            if ckpt_path.exists():
                from src.utils.checkpoint import load_checkpoint
                load_checkpoint(str(ckpt_path), model=model)
                logger.info(f"  已加载 checkpoint: {ckpt_path}")
            else:
                logger.warning(f"  未找到 checkpoint: {ckpt_path}，使用未训练权重")

        models[name] = model

    # TODO: 统一加载测试数据
    # from src.evaluation import BenchmarkRunner
    # runner = BenchmarkRunner(test_loader, cfg_dict)
    # results = runner.run(models)
    # runner.print_comparison_table(results)

    logger.info(f"批量对比完成: {', '.join(methods)}")
