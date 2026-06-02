"""实验记录 — 每次运行自动存档配置、结果、日志。"""

import json, shutil, logging
from pathlib import Path
from datetime import datetime


class ExperimentRecorder:
    """实验记录器：每次 run 自动创建记录目录，存档关键信息。"""

    def __init__(self, experiment_dir: str, config: dict = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = Path(experiment_dir) / timestamp
        self.path.mkdir(parents=True, exist_ok=True)

        # 保存配置
        if config:
            self.save_config(config)

    def save_config(self, config: dict):
        import yaml
        with open(self.path / "config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False)

    def save_metrics(self, metrics: dict, name: str = "metrics"):
        with open(self.path / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    def save_results_summary(self, results: dict):
        """保存 benchmark 对比结果。"""
        with open(self.path / "comparison.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    @property
    def log_path(self):
        return str(self.path / "run.log")
