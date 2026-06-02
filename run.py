#!/usr/bin/env python
"""HSI-MSI Fusion Benchmark — 统一入口。"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def parse_args():
    p = argparse.ArgumentParser(description="HSI-MSI Fusion Benchmark")
    p.add_argument("--mode", "-m", choices=["train", "eval", "benchmark", "visualize"],
                   default="eval")
    p.add_argument("--method", "-M", default="cnmf")
    p.add_argument("--config", "-c", default="config/default.yaml")
    p.add_argument("--dataset", default="paviau_h5py")
    p.add_argument("--checkpoint", "--ckpt", default=None)
    p.add_argument("--methods", default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    kw = dict(config_path=args.config, method_name=args.method,
              dataset=args.dataset, seed=args.seed)

    if args.mode == "eval":
        from src.run_modes.eval import run_eval
        run_eval(**kw, checkpoint_path=args.checkpoint)

    elif args.mode == "train":
        from src.run_modes.train import run_train
        run_train(**kw)

    elif args.mode == "benchmark":
        from src.run_modes.benchmark import run_benchmark
        run_benchmark(**kw, methods=args.methods.split(",") if args.methods else [args.method])

    elif args.mode == "visualize":
        from src.run_modes.visualize import run_visualize
        run_visualize(**kw)


if __name__ == "__main__":
    main()
