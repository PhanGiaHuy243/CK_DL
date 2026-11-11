import argparse
import time

from fabric_pipeline.common import LOG, guarded_main, load_config, output, read_json, write_json
from fabric_pipeline.experiments import check_environment, create_smoke_data, run_directory, train_experiment


def main():
    parser = argparse.ArgumentParser(description="Short end-to-end CPU/GPU check; never creates research results")
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--reuse-completed", action="store_true", help="Use completed smoke runs after checking their metadata")
    args = parser.parse_args()
    cfg = load_config(args.config)
    check_environment(cfg)
    n, nv = create_smoke_data(cfg)
    started = time.perf_counter()
    for experiment in ("baseline", "augmented"):
        run = run_directory(cfg, experiment, smoke=True)
        if args.reuse_completed and (run / "run_summary.json").is_file():
            summary = read_json(run / "run_summary.json")
            if summary["seed"] != cfg["seed"] or summary["batch"] != cfg["smoke"]["batch"] or summary["resolved_args"]["imgsz"] != cfg["smoke"]["imgsz"]:
                raise ValueError("Existing smoke configuration differs; use a new output directory")
            if not (run / "validation_metrics.json").is_file():
                from fabric_pipeline.evaluation import evaluate_one
                evaluate_one(cfg, experiment, device=args.device, smoke=True)
            LOG.info("Reusing completed smoke run: %s", run)
        else:
            train_experiment(cfg, experiment, device=args.device, smoke=True)
    from fabric_pipeline.evaluation import prediction_gallery
    from fabric_pipeline.comparison import compare
    prediction_gallery(cfg, device=args.device, smoke=True)
    compare(cfg, smoke=True)
    write_json(output(cfg, "smoke", "smoke_status.json"), {"status": "passed", "scientific_result": False,
                                                          "baseline_train": n, "augmented_train": 3*n,
                                                          "validation": nv, "seconds_this_invocation": time.perf_counter()-started})
    LOG.info("End-to-end smoke test PASSED. No official training has been run.")


if __name__ == "__main__":
    guarded_main(main)
