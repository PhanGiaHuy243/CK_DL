"""Run official experiments sequentially, with persistent status and per-stage logs."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from fabric_pipeline.common import guarded_main, load_config, output, sha256


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--budget", choices=["matched", "same_epochs"], default="matched")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["_root"])
    directory = output(cfg, args.budget)
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / "pipeline_status.json"
    lock_path = directory / "pipeline.lock"
    if status_path.exists():
        raise FileExistsError(f"Pipeline status already exists: {status_path}. Inspect it before starting another pipeline.")
    for experiment in ("baseline", "augmented"):
        if (directory / "runs" / f"yolov8n_{experiment}").exists():
            raise FileExistsError(f"Existing official run: {experiment}; refusing to overwrite.")
    lock = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(lock, str(os.getpid()).encode("ascii"))
    os.close(lock)
    config_path = Path(args.config).resolve()
    initial_config_hash = sha256(config_path)
    status = {"status": "running", "pid": os.getpid(), "started_utc": timestamp(),
              "config": str(config_path), "config_sha256": initial_config_hash, "device": args.device,
              "budget": args.budget, "training_batch": cfg["training"]["batch"], "imgsz": cfg["training"]["imgsz"],
              "epochs": cfg["budgets"][args.budget], "stages": []}

    def save_status():
        temporary = status_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(status_path)

    started = time.perf_counter()
    # Prevent sleep while this explicit long-running experiment is active, without changing power settings.
    sleep_guard = None
    try:
        if os.name == "nt":
            import ctypes
            sleep_guard = ctypes.windll.kernel32.SetThreadExecutionState
            sleep_guard(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED; display may turn off normally.
        save_status()
        for stage in ("check_environment", "train_baseline", "train_augmented", "evaluate_models", "compare_experiments"):
            if sha256(config_path) != initial_config_hash:
                raise RuntimeError("Configuration changed during full training; stopping to preserve comparison fairness.")
            command = [sys.executable, "-u", str(root / f"{stage}.py"), "--config", str(config_path)]
            if stage != "check_environment":
                command += ["--budget", args.budget]
            if stage in ("train_baseline", "train_augmented", "evaluate_models"):
                command += ["--device", args.device]
            log_path = directory / f"{stage}.log"
            record = {"stage": stage, "status": "running", "started_utc": timestamp(), "log": str(log_path), "command": command}
            status["stage"] = stage
            status["stages"].append(record)
            save_status()
            with log_path.open("w", encoding="utf-8") as log:
                env = os.environ.copy()
                env.update(PYTHONUNBUFFERED="1", PYTHONUTF8="1", MPLBACKEND="Agg")
                process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, env=env)
                record["pid"] = process.pid
                save_status()
                return_code = process.wait()
            record.update(return_code=return_code, finished_utc=timestamp(), status="passed" if return_code == 0 else "failed")
            save_status()
            if return_code:
                raise RuntimeError(f"{stage} failed (exit {return_code}). See {log_path}")
        status.update(status="completed", finished_utc=timestamp(), seconds=time.perf_counter()-started,
                      comparison=str(directory / "comparison" / "comparison.md"))
        save_status()
    except BaseException as exc:
        status.update(status="failed", finished_utc=timestamp(), error=str(exc), traceback=traceback.format_exc())
        save_status()
        raise
    finally:
        if sleep_guard is not None:
            sleep_guard(0x80000000)  # Release only this process's sleep request.
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    guarded_main(main)
