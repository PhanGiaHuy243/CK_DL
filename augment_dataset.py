import argparse
from fabric_pipeline.common import guarded_main, load_config
from fabric_pipeline.augmentation import audit_augmentation, build_augmentation


def main():
    parser = argparse.ArgumentParser(description="Build/audit exact offline augmentation; never overwrite source")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    (audit_augmentation if args.audit_only else build_augmentation)(cfg)


if __name__ == "__main__":
    guarded_main(main)
