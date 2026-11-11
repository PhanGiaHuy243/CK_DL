import argparse
from fabric_pipeline.common import guarded_main, load_config
from fabric_pipeline.experiments import check_environment


def main():
    parser = argparse.ArgumentParser(description="Check dependencies, CUDA, torchvision NMS; optionally fetch pretrained weights")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--download-weights", action="store_true")
    args = parser.parse_args()
    check_environment(load_config(args.config), args.download_weights)


if __name__ == "__main__":
    guarded_main(main)
