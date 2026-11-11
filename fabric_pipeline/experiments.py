from __future__ import annotations

import math
import random
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

import numpy as np

from .common import (LOG, device_for, environment, output, read_json, resolve,
                     setup_ultralytics, sha256, write_json, write_yaml)
from .data import checked_copy, discover, pairs, prepare_original, read_boxes, verify_source
from .augmentation import augmented_root, select_preview_indices

ZERO_AUGMENTATION = {"hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "degrees": 0.0,
                     "translate": 0.0, "scale": 0.0, "shear": 0.0, "perspective": 0.0,
                     "flipud": 0.0, "fliplr": 0.0, "mosaic": 0.0, "mixup": 0.0,
                     "cutmix": 0.0, "copy_paste": 0.0, "bgr": 0.0, "erasing": 0.0,
                     "auto_augment": None}
PRETRAINED_SHA256 = "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"


def checkpoint_path(cfg):
    value = cfg["training"]["checkpoint"]
    p = resolve(cfg, value)
    return p if p.is_file() else output(cfg, "weights", Path(value).name)


def download_weights(cfg):
    if Path(cfg["training"]["checkpoint"]).name != "yolov8n.pt":
        raise ValueError("This pipeline requires yolov8n.pt pretrained initialization")
    path = checkpoint_path(cfg)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".download")
        url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"
        LOG.info("Downloading official pretrained weights: %s", url)
        try:
            urllib.request.urlretrieve(url, partial)
        except (OSError, urllib.error.URLError) as exc:
            raise RuntimeError(f"Cannot download pretrained weights: {exc}. Download {url} to {path} and rerun this command.") from exc
        if sha256(partial) != PRETRAINED_SHA256:
            raise RuntimeError("Downloaded checkpoint does not match the pinned official yolov8n.pt checksum")
        partial.replace(path)
    digest = sha256(path)
    if digest != PRETRAINED_SHA256:
        raise RuntimeError("Checkpoint is not the pinned official pretrained yolov8n.pt. Download from the documented release URL.")
    lock = path.with_suffix(".sha256.json")
    if lock.exists() and read_json(lock)["sha256"] != digest:
        raise RuntimeError("Pretrained checkpoint changed since previous run")
    write_json(lock, {"sha256": digest, "filename": path.name,
                      "source": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt"})
    LOG.info("Pretrained checkpoint SHA256: %s", digest)
    return path


def check_environment(cfg, download=False):
    info = environment()
    write_json(output(cfg, "environment.json"), info)
    LOG.info("Python %s; torch=%s; CUDA=%s; GPUs=%s", sys.version.split()[0], info["packages"]["torch"], info["cuda"], info.get("gpus"))
    setup_ultralytics(cfg)
    import torch
    import torchvision
    # Detect a mismatched torch/torchvision build before spending time on training.
    kept = torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.]]), torch.tensor([0.9]), 0.5)
    if kept.tolist() != [0]:
        raise RuntimeError("torchvision NMS check failed")
    requested_device = device_for(cfg)
    if requested_device != "cpu":
        cuda_device = torch.device(f"cuda:{requested_device}")
        matrix = torch.ones((64, 64), device=cuda_device)
        product = matrix @ matrix
        kept_cuda = torchvision.ops.nms(torch.tensor([[0., 0., 10., 10.]], device=cuda_device),
                                         torch.tensor([0.9], device=cuda_device), 0.5)
        torch.cuda.synchronize(cuda_device)
        if product[0, 0].item() != 64 or kept_cuda.tolist() != [0]:
            raise RuntimeError("CUDA matrix multiplication/torchvision NMS check failed")
        info["cuda_runtime_check"] = {"passed": True, "device": str(product.device),
                                      "cudnn_version": torch.backends.cudnn.version()}
        write_json(output(cfg, "environment.json"), info)
        LOG.info("CUDA compute and torchvision CUDA NMS PASSED on %s", torch.cuda.get_device_name(cuda_device))
    if download:
        download_weights(cfg)
    return info


def run_directory(cfg, experiment, budget="matched", smoke=False):
    group = "smoke" if smoke else budget
    return output(cfg, group, "runs", f"yolov8n_{experiment}")


def create_smoke_data(cfg, baseline_only=False):
    baseline_yaml = prepare_original(cfg)
    root = baseline_yaml.parent
    aug = augmented_root(cfg)
    if not baseline_only and not (aug / "augmentation_audit.json").is_file():
        raise FileNotFoundError("Generate and audit augmentation before smoke test")
    _, names, _ = discover(cfg)
    train_items = pairs(root / "train/images", root / "train/labels")
    all_boxes = [read_boxes(lab, tolerance=2e-12)[0] for _, lab in train_items]
    selected = select_preview_indices(all_boxes, cfg["smoke"]["images"], cfg["seed"])
    val_items = pairs(root / "val/images", root / "val/labels")
    val_boxes = [read_boxes(lab, tolerance=cfg["dataset"]["rounding_tolerance"])[0] for _, lab in val_items]
    selected_val = select_preview_indices(val_boxes, cfg["smoke"]["val_images"], cfg["seed"])
    import csv
    manifest = []
    if not baseline_only:
        with (aug / "augmentation_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
            manifest = list(csv.DictReader(f))
    source_names = {train_items[i][0].relative_to(root / "train/images").as_posix() for i in selected}
    for experiment in (("baseline",) if baseline_only else ("baseline", "augmented")):
        dst = output(cfg, "smoke", "data", experiment)
        subset = [train_items[i] for i in selected] if experiment == "baseline" else [
            (aug / row["new_image"], aug / row["new_label"]) for row in manifest if row["source_image"] in source_names]
        for split, items in [("train", subset), ("val", [val_items[i] for i in selected_val])]:
            for image, lab in items:
                checked_copy(image, dst / split / "images" / image.name)
                checked_copy(lab, dst / split / "labels" / lab.name)
        write_yaml(dst / "data.yaml", {"path": str(dst), "train": "train/images", "val": "val/images", "nc": len(names), "names": names})
    return len(selected), len(selected_val)


def train_experiment(cfg, experiment, budget="matched", device=None, smoke=False):
    if experiment not in ("baseline", "augmented"):
        raise ValueError(experiment)
    setup_ultralytics(cfg)
    from ultralytics import YOLO
    from .trainer import PreprocessingOnlyTrainer
    import torch
    if "," in device_for(cfg, device):
        raise ValueError("Use one GPU for this measured-budget adapter; DDP is not supported")
    if cfg["training"]["optimizer"] == "auto":
        raise ValueError("Specify optimizer explicitly; auto can change behavior with dataset size")
    if cfg["training"]["batch"] <= 0 or cfg["training"]["warmup_epochs"] != 0 or cfg["training"]["patience"] != 0:
        raise ValueError("Use fixed positive batch, warmup_epochs=0, patience=0 for measured budgets")
    if cfg["training"].get("amp"):
        raise ValueError("Use amp=false so skipped scaler updates cannot change the matched budget")
    if cfg["training"].get("fraction", 1) != 1 or cfg["training"].get("multi_scale") or cfg["training"].get("rect"):
        raise ValueError("Use fraction=1, multi_scale=false, rect=false for this protocol")
    run_dir = run_directory(cfg, experiment, budget, smoke)
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}. Use a new output directory; completed artifacts are never overwritten.")
    baseline_yaml = prepare_original(cfg)
    if experiment == "augmented":
        audit = read_json(augmented_root(cfg) / "augmentation_audit.json")
        if audit["status"] != "passed" or not audit["pixel_check"]:
            raise RuntimeError("Augmentation must pass the full pixel and geometry audit")
        # Check current files against manifest before training, not only a historic audit status.
        from .augmentation import audit_augmentation
        if not smoke:
            audit_augmentation(cfg)
    data_yaml = baseline_yaml if experiment == "baseline" else augmented_root(cfg) / "data.yaml"
    if smoke:
        data_yaml = output(cfg, "smoke", "data", experiment, "data.yaml")
    checkpoint = checkpoint_path(cfg)
    if not checkpoint.is_file():
        raise FileNotFoundError("Missing yolov8n.pt. Run python check_environment.py --download-weights first.")
    if checkpoint.name != "yolov8n.pt":
        raise ValueError("Both experiments must start from yolov8n.pt")
    digest = sha256(checkpoint)
    if digest != PRETRAINED_SHA256:
        raise RuntimeError("Both experiments require the pinned official pretrained yolov8n.pt checksum")
    lock = checkpoint.with_suffix(".sha256.json")
    if not lock.exists() or read_json(lock)["sha256"] != digest:
        raise RuntimeError("Missing/mismatched pretrained checksum. Run check_environment.py --download-weights.")
    epochs = cfg["budgets"][budget][experiment]
    args = {**cfg["training"], **ZERO_AUGMENTATION}
    args.pop("checkpoint")
    if smoke:
        epochs = cfg["smoke"]["epochs"] * (3 if experiment == "baseline" else 1)
        args.update(imgsz=cfg["smoke"]["imgsz"], batch=cfg["smoke"]["batch"])
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    args.update(data=str(data_yaml), epochs=epochs, seed=cfg["seed"], device=device_for(cfg, device),
                project=str(run_dir.parent), name=run_dir.name, exist_ok=False, nbs=args["batch"],
                resume=False, pretrained=True, time=None, conf=cfg["evaluation"]["conf"],
                iou=cfg["evaluation"]["iou"], max_det=cfg["evaluation"]["max_det"])
    # A single run owns its directory. The library creates it; metadata is saved after trainer init.
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    model = YOLO(str(checkpoint))  # fresh pretrained object on every call, never Baseline best.pt
    started = time.perf_counter()
    model.train(trainer=PreprocessingOnlyTrainer, **args)
    elapsed = time.perf_counter()-started
    trainer = model.trainer
    if Path(trainer.save_dir).resolve() != run_dir.resolve():
        raise RuntimeError("Unexpected output directory; refusing to mix experiment artifacts")
    expected_images = (cfg["smoke"]["images"] if smoke else cfg["dataset"]["expected"]["train"])*(3 if experiment == "augmented" else 1)
    expected_val = cfg["smoke"]["val_images"] if smoke else cfg["dataset"]["expected"]["val"]
    if len(trainer.train_loader.dataset) != expected_images or len(trainer.test_loader.dataset) != expected_val:
        raise RuntimeError("Unexpected Train/Validation counts in actual dataloaders")
    expected_steps = math.ceil(expected_images/args["batch"])*epochs
    if trainer.actual_steps != expected_steps or trainer.image_presentations != expected_images*epochs or trainer.epoch+1 != epochs:
        raise RuntimeError("Actual training budget differs from planned budget")
    for required in ["weights/best.pt", "weights/last.pt", "results.csv", "results.png"]:
        if not (run_dir / required).is_file():
            raise RuntimeError(f"Missing training artifact: {required}")
    summary = {"experiment": experiment, "budget": "smoke" if smoke else budget, "scientific_result": not smoke,
               "seed": cfg["seed"], "checkpoint_sha256": digest, "checkpoint": str(checkpoint),
               "best_sha256": sha256(run_dir / "weights/best.pt"), "epochs": epochs,
               "train_images": expected_images, "val_images": expected_val, "batch": args["batch"],
               "batches_per_epoch": len(trainer.train_loader), "last_batch_size": expected_images % args["batch"] or args["batch"],
               "optimizer_steps": trainer.actual_steps, "planned_optimizer_steps": expected_steps,
               "image_presentations": trainer.image_presentations, "training_seconds": elapsed,
               "fit_seconds": time.time()-trainer.train_time_start, "data_yaml": str(data_yaml),
               "environment": environment(), "resolved_args": vars(trainer.args),
               "scheduler": "cosine_per_optimizer_step" if args["cos_lr"] else "linear_per_optimizer_step",
               "warmup_steps": 0, "gradient_accumulation": trainer.accumulate,
               "source_hashes_sha256": sha256(output(cfg, "audit", "source_hashes.json")),
               "validation_hashes": read_json(output(cfg, "audit", "source_hashes.json"))["val"]}
    write_json(run_dir / "run_summary.json", summary)
    write_yaml(run_dir / "experiment_config.yaml", cfg)
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
    (run_dir / "requirements.lock.txt").write_text(freeze.stdout, encoding="utf-8")
    verify_source(cfg)
    LOG.info("%s finished: %d optimizer steps, %d image presentations, %.1f seconds", experiment, trainer.actual_steps, trainer.image_presentations, elapsed)
    from .evaluation import evaluate_one
    evaluate_one(cfg, experiment, budget, device, smoke)
    return summary
