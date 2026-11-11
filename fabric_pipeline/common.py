from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG = logging.getLogger("chenab")


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_config(path="configs/experiment.yaml"):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["_root"] = str(path.parent.parent)
    cfg["_config"] = str(path)
    cfg["output"] = str(resolve(cfg, cfg["output"]))
    data_yaml = resolve(cfg, cfg["dataset"]["yaml"])
    if Path(cfg["output"]).is_relative_to(data_yaml.parent):
        raise ValueError("Output must be outside the original dataset directory")
    logs = output(cfg, "logs")
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    handler = logging.FileHandler(logs / f"{Path(sys.argv[0]).stem}_{stamp}.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    return cfg


def resolve(cfg, path):
    p = Path(path)
    return p.resolve() if p.is_absolute() else (Path(cfg["_root"]) / p).resolve()


def output(cfg, *parts):
    return Path(cfg["output"]).joinpath(*parts)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_yaml(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_csv(path, rows, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def environment():
    packages = {}
    for name in ["torch", "torchvision", "ultralytics", "opencv-python", "numpy", "Pillow", "PyYAML", "matplotlib", "pandas", "albumentations"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    info = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": packages}
    try:
        import torch
        info.update(cuda=torch.cuda.is_available(), torch_cuda=torch.version.cuda,
                    gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                    cpu_threads=torch.get_num_threads())
    except ImportError:
        info["cuda"] = False
    return info


def setup_ultralytics(cfg):
    # Keep library settings, downloads, and caches inside the workspace.
    output(cfg, "settings", "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(output(cfg, "settings"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.setdefault("MPLBACKEND", "Agg")
    import ultralytics
    if ultralytics.__version__ != "8.3.221":
        raise RuntimeError(f"Expected ultralytics==8.3.221, found {ultralytics.__version__}. Install requirements.txt.")
    from ultralytics import settings
    settings.update({key: False for key in ["sync", "hub", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune", "tensorboard", "wandb"] if key in settings})


def device_for(cfg, override=None):
    import torch
    device = override if override is not None else cfg["training"]["device"]
    if device == "auto":
        return "0" if torch.cuda.is_available() else "cpu"
    if str(device) != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or install a CUDA torch build.")
    return str(device)


def guarded_main(fn):
    setup_logging()
    try:
        fn()
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, AssertionError) as exc:
        LOG.error("%s", exc)
        raise SystemExit(1) from exc
