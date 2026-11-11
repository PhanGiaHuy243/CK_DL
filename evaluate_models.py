from fabric_pipeline.cli import experiment_args
from fabric_pipeline.common import guarded_main
from fabric_pipeline.evaluation import evaluate_one, prediction_gallery


def main():
    cfg, args = experiment_args("Evaluate both best checkpoints on exactly the same Validation")
    for experiment in ("baseline", "augmented"):
        evaluate_one(cfg, experiment, args.budget, args.device, args.smoke)
    prediction_gallery(cfg, args.budget, args.device, args.smoke)


if __name__ == "__main__":
    guarded_main(main)
