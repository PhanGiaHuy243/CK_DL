from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from .common import LOG, output, read_json, resolve, sha256, write_csv, write_json, write_yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def discover(cfg):
    yaml_path = resolve(cfg, cfg["dataset"]["yaml"])
    raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = raw["names"]
    if isinstance(names, list):
        names = dict(enumerate(names))
    names = {int(k): str(v) for k, v in names.items()}
    n = cfg["dataset"]["classes"]
    if set(names) != set(range(n)) or raw.get("nc", n) != n:
        raise ValueError(f"Expected contiguous class IDs 0..{n-1}; found {names}")
    declared = Path(raw.get("path", "."))
    root = declared if declared.is_absolute() else yaml_path.parent / declared
    if not root.exists():
        LOG.warning("Dataset path in YAML does not exist: %s; using YAML directory %s", root, yaml_path.parent)
        root = yaml_path.parent
    splits = {}
    for split in ("train", "val", "test"):
        entry = raw.get(split)
        if not isinstance(entry, str):
            raise ValueError(f"{split} must point to one image directory in {yaml_path}")
        images = Path(entry)
        images = images if images.is_absolute() else root / images
        images = images.resolve()
        if images.name != "images":
            raise ValueError(f"Expected images/labels sibling layout, found {images}")
        labels = images.parent / "labels"
        if not images.is_dir() or not labels.is_dir():
            raise FileNotFoundError(f"Missing images/labels for {split}: {images}")
        splits[split] = {"images": images, "labels": labels}
    destination = Path(cfg["output"]).resolve()
    if destination.is_relative_to(yaml_path.parent) or any(destination.is_relative_to(d["images"].parent) for d in splits.values()):
        raise ValueError("Output directory must be outside the original dataset")
    return yaml_path, names, splits


def pairs(images, labels):
    imgs = sorted(p for p in Path(images).rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    labs = sorted(Path(labels).rglob("*.txt"))
    # Preserve nested directories, reject ambiguous image stems.
    mapping = {}
    for im in imgs:
        key = im.relative_to(images).with_suffix("").as_posix()
        if key in mapping:
            raise ValueError(f"Duplicate image stem: {im}")
        mapping[key] = (im, Path(labels) / im.relative_to(images).with_suffix(".txt"))
    if set(labs) != {lab for _, lab in mapping.values()}:
        raise ValueError(f"Images and labels are not one-to-one: {images}")
    return list(mapping.values())


def read_boxes(path, nc=7, tolerance=0.0):
    rows, rounding = [], []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = np.array([float(v) for v in line.split()], dtype=np.float64)
        except ValueError as exc:
            raise ValueError(f"Invalid number {path}:{line_number}") from exc
        if row.shape != (5,) or not np.isfinite(row).all():
            raise ValueError(f"Expected finite class xc yc w h: {path}:{line_number}")
        c, x, y, w, h = row
        if c != int(c) or not 0 <= c < nc or min(w, h) <= 0 or min(x, y, w, h) < 0 or max(x, y, w, h) > 1:
            raise ValueError(f"Invalid class/YOLO coordinates: {path}:{line_number}: {row}")
        xyxy = np.array([x-w/2, y-h/2, x+w/2, y+h/2])
        excess = float(max(0, -xyxy.min(), xyxy.max()-1))
        if excess > tolerance + 1e-12:
            raise ValueError(f"Box outside image by {excess}: {path}:{line_number}")
        if excess > 1e-12:
            rounding.append({"label": str(path), "line": line_number, "excess": excess, "original": row.tolist()})
        rows.append(row)
    return np.asarray(rows, dtype=np.float64).reshape(-1, 5), rounding


def canonical_boxes(boxes):
    b = boxes.copy()
    if len(b):
        lo = np.maximum(0, b[:, 1:3] - b[:, 3:5]/2)
        hi = np.minimum(1, b[:, 1:3] + b[:, 3:5]/2)
        b[:, 1:3] = (lo+hi)/2
        b[:, 3:5] = hi-lo
    return b


def write_boxes(path, boxes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{int(b[0])} " + " ".join(f"{x:.12f}" for x in b[1:]) + "\n" for b in boxes), encoding="utf-8")
    read_boxes(path, tolerance=2e-12)


def statistics(items, nc, tolerance=0):
    box_counts = np.zeros(nc, dtype=int)
    image_counts = np.zeros(nc, dtype=int)
    warnings = []
    multi = 0
    for im, lab in items:
        with Image.open(im) as image:
            image.verify()
        boxes, rounding = read_boxes(lab, nc, tolerance)
        warnings.extend(rounding)
        cs = boxes[:, 0].astype(int)
        box_counts += np.bincount(cs, minlength=nc)
        image_counts[np.unique(cs)] += 1
        multi += len(boxes) > 1
    return {"images": len(items), "labels": len(items), "boxes_per_class": box_counts.tolist(),
            "images_per_class": image_counts.tolist(), "multi_box_images": multi, "rounding_warnings": warnings}


def snapshot(splits):
    result = {}
    for split, dirs in splits.items():
        result[split] = {kind: {p.relative_to(folder).as_posix(): sha256(p)
                              for p in sorted(folder.rglob("*")) if p.is_file()}
                         for kind, folder in dirs.items()}
    return result


def verify_source(cfg):
    _, _, splits = discover(cfg)
    expected = read_json(output(cfg, "audit", "source_hashes.json"))
    if snapshot(splits) != expected:
        raise RuntimeError("Original dataset changed since audit. Inspect changes before creating a new output directory.")
    return splits


def audit_source(cfg):
    yaml_path, names, splits = discover(cfg)
    report = {"yaml": str(yaml_path), "names": names, "splits": {}}
    for split, dirs in splits.items():
        items = pairs(**dirs)
        if len(items) != cfg["dataset"]["expected"][split]:
            raise ValueError(f"Wrong {split} count: {len(items)}")
        report["splits"][split] = statistics(items, len(names), cfg["dataset"]["rounding_tolerance"])
        LOG.info("%s: %d image/label pairs, %d boxes", split, len(items), sum(report["splits"][split]["boxes_per_class"]))
    hashes = snapshot(splits)
    seen = defaultdict(list)
    for split, kinds in hashes.items():
        for path, digest in kinds["images"].items():
            seen[digest].append((split, path))
    leakage = [v for v in seen.values() if len({s for s, _ in v}) > 1]
    report["cross_split_exact_duplicates"] = leakage
    old = output(cfg, "audit", "source_hashes.json")
    if old.exists() and read_json(old) != hashes:
        raise RuntimeError("Source hashes differ from previous audit. Preserve previous artifacts and inspect source changes.")
    write_json(old, hashes)
    write_json(output(cfg, "audit", "dataset_audit.json"), report)
    rows = [{"split": split, "class_id": c, "class": name, "images": info["images_per_class"][c], "boxes": info["boxes_per_class"][c]}
            for split, info in report["splits"].items() for c, name in names.items()]
    write_csv(output(cfg, "audit", "class_counts.csv"), rows)
    text = ["# Dataset audit", "", f"Source YAML: `{yaml_path}`", "", "Original split assignments preserved. Test inspected for integrity only.", "",
            "|Split|Images|Boxes|Multi-box images|Rounding warnings|", "|---|---:|---:|---:|---:|"]
    for s, info in report["splits"].items():
        text.append(f"|{s}|{info['images']}|{sum(info['boxes_per_class'])}|{info['multi_box_images']}|{len(info['rounding_warnings'])}|")
    text += ["", f"Cross-split byte-identical image groups: {len(leakage)}.",
             "", "Small boundary rounding is logged in JSON. Only staged Train labels are canonicalized; Validation/Test bytes are unchanged."]
    output(cfg, "audit", "dataset_audit.md").write_text("\n".join(text), encoding="utf-8")
    if leakage:
        raise RuntimeError("Cross-split exact duplicates found; see dataset_audit.json. Do not resplit automatically.")
    return report


def checked_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise RuntimeError(f"Existing copy changed: {target}")
    else:
        shutil.copy2(source, target)


def prepared_yaml(cfg):
    return output(cfg, "prepared", "baseline", "data.yaml")


def prepare_original(cfg):
    if not output(cfg, "audit", "source_hashes.json").exists():
        audit_source(cfg)
    splits = verify_source(cfg)
    _, names, _ = discover(cfg)
    root = output(cfg, "prepared", "baseline")
    if any(root.resolve().is_relative_to(d["images"].parent) for d in splits.values()):
        raise ValueError("Output must be outside source dataset.")
    corrections = []
    for split in ("train", "val"):
        dirs = splits[split]
        for im, lab in pairs(**dirs):
            dst_im = root / split / "images" / im.relative_to(dirs["images"])
            dst_lab = root / split / "labels" / lab.relative_to(dirs["labels"])
            checked_copy(im, dst_im)
            if split == "train":
                boxes, rounding = read_boxes(lab, len(names), cfg["dataset"]["rounding_tolerance"])
                corrections.extend(rounding)
                if rounding:
                    write_boxes(dst_lab, canonical_boxes(boxes))
                else:
                    checked_copy(lab, dst_lab)
            else:
                checked_copy(lab, dst_lab)
    write_json(root / "train_rounding_corrections.json", corrections)
    write_yaml(root / "data.yaml", {"path": str(root), "train": "train/images", "val": "val/images", "nc": len(names), "names": names})
    return root / "data.yaml"
