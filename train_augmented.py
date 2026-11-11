from fabric_pipeline.cli import experiment_args
from fabric_pipeline.common import guarded_main
from fabric_pipeline.experiments import train_experiment


def main():
    cfg, args = experiment_args("Train independent YOLOv8n on offline augmented Train")
    train_experiment(cfg, "augmented", args.budget, args.device, args.smoke)


if __name__ == "__main__":
    guarded_main(main)
