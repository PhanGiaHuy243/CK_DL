import argparse
from .common import load_config


def experiment_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--budget", choices=["matched", "same_epochs"], default="matched")
    parser.add_argument("--device", help="cpu or one GPU index, e.g. 0; defaults to config/auto")
    parser.add_argument("--smoke", action="store_true", help="Only use artifacts/subsets from smoke_test.py")
    args = parser.parse_args()
    return load_config(args.config), args
