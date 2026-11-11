from fabric_pipeline.cli import experiment_args
from fabric_pipeline.common import guarded_main
from fabric_pipeline.comparison import compare


def main():
    cfg, args = experiment_args("Compare measured Validation results and actual training budgets")
    compare(cfg, args.budget, args.smoke)


if __name__ == "__main__":
    guarded_main(main)
