"""评估模式：自动加载配置 → 模型 → 数据 → 评估 → 记录。"""

from pathlib import Path


def run_eval(config_path="config/default.yaml", method_name="cnmf",
             dataset="paviau_h5py", checkpoint_path=None, seed=None):
    # 1. 配置 (default → dataset → method)
    from omegaconf import OmegaConf
    from src.utils.config import load_config, config_to_dict

    config = load_config(config_path)
    ds_path = f"config/dataset/{dataset}.yaml"
    if Path(ds_path).exists():
        config.data = OmegaConf.merge(config.data, load_config(ds_path))
    config.data.dataset = dataset
    m_path = f"config/method/{method_name}.yaml"
    if Path(m_path).exists():
        config = OmegaConf.merge(config, load_config(m_path))

    cfg_dict = config_to_dict(config)

    # 2. 种子
    from src.utils.reproducibility import set_seed, set_deterministic
    set_seed(seed or cfg_dict.get("experiment", {}).get("seed", 42))
    set_deterministic(cfg_dict.get("experiment", {}).get("deterministic", True))

    # 3. 模型
    from src.models import ModelRegistry
    model = ModelRegistry.get(method_name)(cfg_dict)
    if checkpoint_path:
        from src.utils.checkpoint import load_checkpoint
        load_checkpoint(checkpoint_path, model=model)

    # 4. 数据
    from src.data.datasets import HSIDataset
    from src.data.transforms import ToTensor
    from torchvision import transforms as T
    from src.data.dataloader import build_dataloader

    data_cfg = cfg_dict["data"]
    ds = HSIDataset(data_cfg["root_dir"], "val", T.Compose([ToTensor()]), data_cfg)
    loader = build_dataloader(ds, cfg_dict, "test")

    # 5. 评估
    from src.evaluation import Evaluator
    metrics = Evaluator(model, cfg_dict).evaluate(loader)

    # 6. 记录
    from src.utils.experiment import ExperimentRecorder
    rec = ExperimentRecorder(f"experiments/{method_name}", cfg_dict)
    rec.save_metrics(metrics)

    # 7. 打印
    for k, v in metrics.items():
        v = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"  {k}: {v}")
    print(f"→ {rec.path}")
