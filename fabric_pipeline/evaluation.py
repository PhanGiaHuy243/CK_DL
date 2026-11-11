from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .common import (LOG, device_for, output, read_json, setup_ultralytics,
                     sha256, write_csv, write_json)
from .data import discover, pairs, read_boxes, verify_source
from .augmentation import draw_boxes, select_preview_indices
from .experiments import run_directory


def finite(value):
    number = float(value)
    return number if math.isfinite(number) else None


def evaluate_one(cfg, experiment, budget="matched", device=None, smoke=False):
    setup_ultralytics(cfg)
    from ultralytics import YOLO
    import yaml
    run = run_directory(cfg, experiment, budget, smoke)
    summary = read_json(run / "run_summary.json")
    best = run / "weights/best.pt"
    if sha256(best) != summary["best_sha256"]:
        raise RuntimeError("Best checkpoint changed after training")
    _, names, _ = discover(cfg)
    verify_source(cfg)
    data_yaml = Path(summary["data_yaml"])
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if "test" in data:
        raise ValueError("Evaluation configuration exposes Test")
    val = Path(data["val"])
    if not val.is_absolute():
        val = Path(data["path"]) / val
    val_items = pairs(val, val.parent / "labels")
    expected = cfg["smoke"]["val_images"] if smoke else cfg["dataset"]["expected"]["val"]
    if len(val_items) != expected:
        raise ValueError(f"Expected {expected} Validation images, got {len(val_items)}")
    val_hashes = {"images": {im.relative_to(val).as_posix(): sha256(im) for im, _ in val_items},
                  "labels": {lab.relative_to(val.parent / 'labels').as_posix(): sha256(lab) for _, lab in val_items}}
    if not smoke and val_hashes != summary["validation_hashes"]:
        raise ValueError("Validation differs from training selection set")
    settings = {"split": "val", "conf": cfg["evaluation"]["conf"], "iou": cfg["evaluation"]["iou"],
                "max_det": cfg["evaluation"]["max_det"], "imgsz": summary["resolved_args"]["imgsz"],
                "batch": cfg["smoke"]["batch"] if smoke else cfg["evaluation"]["batch"],
                "augment": False, "half": False, "rect": True, "seed": cfg["seed"],
                "workers": cfg["training"]["workers"], "device": device_for(cfg, device)}
    model = YOLO(str(best))
    actual_names = {int(k): str(v) for k, v in model.names.items()}
    if actual_names != names:
        raise ValueError(f"Checkpoint class names differ from data.yaml: {actual_names}")
    metrics = model.val(data=str(data_yaml), project=str(run), name="validation", exist_ok=True,
                        plots=True, save_json=False, **settings)
    present = {int(c): i for i, c in enumerate(metrics.box.ap_class_index)}
    per_class = []
    for class_id, name in names.items():
        index = present.get(class_id)
        per_class.append({"class_id": class_id, "class": name,
                          "AP50": finite(metrics.box.ap50[index]) if index is not None else None,
                          "AP50_95": finite(metrics.box.ap[index]) if index is not None else None,
                          "precision": finite(metrics.box.p[index]) if index is not None else None,
                          "recall": finite(metrics.box.r[index]) if index is not None else None})
    result = {"scientific_result": not smoke, "seed": summary["seed"], "evaluator": "ultralytics==8.3.221 DetectionValidator",
              "checkpoint_sha256": sha256(best), "validation_hashes": val_hashes, "settings": settings,
              "ap_matching_iou_thresholds": np.linspace(0.5, 0.95, 10).tolist(),
              "metrics": {"mAP50_95": finite(metrics.box.map), "mAP50": finite(metrics.box.map50),
                          "precision": finite(metrics.box.mp), "recall": finite(metrics.box.mr)},
              "per_class": per_class, "validation_images": len(val_items)}
    write_json(run / "validation_metrics.json", result)
    write_csv(run / "per_class_ap.csv", per_class)
    LOG.info("Validation metrics saved: %s", run / "validation_metrics.json")
    return result


def prediction_gallery(cfg, budget="matched", device=None, smoke=False):
    setup_ultralytics(cfg)
    from ultralytics import YOLO
    import yaml
    base_run = run_directory(cfg, "baseline", budget, smoke)
    summary = read_json(base_run / "run_summary.json")
    data = yaml.safe_load(Path(summary["data_yaml"]).read_text(encoding="utf-8"))
    val = Path(data["val"])
    if not val.is_absolute():
        val = Path(data["path"]) / val
    items = pairs(val, val.parent / "labels")
    _, names, _ = discover(cfg)
    boxes = [read_boxes(lab, len(names), cfg["dataset"]["rounding_tolerance"])[0] for _, lab in items]
    indices = select_preview_indices(boxes, min(len(items), cfg["evaluation"]["gallery_count"]), cfg["seed"])
    paths = [str(items[i][0]) for i in indices]
    gallery = output(cfg, "smoke" if smoke else budget, "comparison", "gallery")
    gallery.mkdir(parents=True, exist_ok=True)
    predictions = {}
    for experiment in ("baseline", "augmented"):
        model = YOLO(str(run_directory(cfg, experiment, budget, smoke) / "weights/best.pt"))
        predictions[experiment] = model.predict(paths, imgsz=summary["resolved_args"]["imgsz"],
                                                conf=cfg["evaluation"]["gallery_conf"], iou=cfg["evaluation"]["iou"],
                                                max_det=cfg["evaluation"]["max_det"], device=device_for(cfg, device),
                                                augment=False, half=False, save=False, verbose=False)
    gallery_rows = []
    for j, index in enumerate(indices):
        panels = [draw_boxes(items[index][0], boxes[index], names, "Ground truth")]
        for experiment in ("baseline", "augmented"):
            pixels = predictions[experiment][j].plot(line_width=2)
            panel = Image.fromarray(pixels[:, :, ::-1])
            panel.thumbnail((700, 700))
            ImageDraw.Draw(panel).text((4, 4), experiment, fill="white", stroke_width=1, stroke_fill="black")
            panels.append(panel)
        canvas = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), (30, 30, 30))
        x = 0
        for panel in panels:
            canvas.paste(panel, (x, 0)); x += panel.width
        name = f"{j:02d}_{items[index][0].stem}.jpg"
        canvas.save(gallery / name, quality=92)
        gallery_rows.append({"validation_image": str(items[index][0]), "gallery": name,
                             "ground_truth_boxes": len(boxes[index]),
                             "baseline_detections": len(predictions["baseline"][j].boxes),
                             "augmented_detections": len(predictions["augmented"][j].boxes)})
    write_csv(gallery / "manifest.csv", gallery_rows)
    write_json(gallery / "settings.json", {"confidence": cfg["evaluation"]["gallery_conf"], "nms_iou": cfg["evaluation"]["iou"],
                                           "note": "Display confidence only; AP evaluation uses the lower configured evaluation confidence."})
