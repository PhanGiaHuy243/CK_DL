from __future__ import annotations

import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .common import LOG, output, read_json, sha256, write_csv, write_json, write_yaml
from .data import (canonical_boxes, checked_copy, discover, pairs, prepare_original,
                   read_boxes, snapshot, statistics, verify_source, write_boxes)


def rotation_matrix(width, height, angle):
    # Transform all four image corners, translating the whole image into expanded canvas.
    matrix = cv2.getRotationMatrix2D((width/2, height/2), float(angle), 1.0)
    corners = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]], dtype=float)
    transformed = corners @ matrix.T
    lower = np.floor(transformed.min(axis=0))
    upper = np.ceil(transformed.max(axis=0))
    matrix[:, 2] -= lower
    size = tuple(int(v) for v in upper-lower)
    return matrix, size


def transform_boxes(boxes, width, height, matrix, size):
    result = boxes.copy()
    if not len(boxes):
        return result
    lo = (boxes[:, 1:3]-boxes[:, 3:5]/2)*[width, height]
    hi = (boxes[:, 1:3]+boxes[:, 3:5]/2)*[width, height]
    corners = np.stack([lo, np.column_stack([hi[:, 0], lo[:, 1]]), hi, np.column_stack([lo[:, 0], hi[:, 1]])], axis=1)
    corners = np.concatenate([corners, np.ones((*corners.shape[:2], 1))], axis=2)
    transformed = corners @ matrix.T
    new_lo, new_hi = transformed.min(axis=1), transformed.max(axis=1)
    if np.any(new_lo < -1e-7) or np.any(new_hi > np.array(size)+1e-7):
        raise ValueError("Expanded rotation clipped a box corner")
    result[:, 1:3] = (new_lo+new_hi)/2/size
    result[:, 3:5] = (new_hi-new_lo)/size
    if np.any(result[:, 3:5] <= 0):
        raise ValueError("Degenerate rotated bounding box")
    return result


def apply_transform(image, boxes, transform, params):
    height, width = image.shape[:2]
    b = boxes.copy()
    if transform == "rotation":
        matrix, size = rotation_matrix(width, height, params["angle"])
        b = transform_boxes(boxes, width, height, matrix, size)
        # Neutral YOLO padding, never reflected or synthesized fabric/defects.
        image = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
        return image, b, {"matrix": matrix.tolist(), "canvas": list(size), "border": [114, 114, 114]}
    if transform == "horizontal_flip":
        b[:, 1] = 1-b[:, 1]
        return cv2.flip(image, 1), b, {}
    if transform == "vertical_flip":
        b[:, 2] = 1-b[:, 2]
        return cv2.flip(image, 0), b, {}
    raise ValueError(f"Unknown transform: {transform}")


def allocate_flips(boxes, seed, candidates):
    n = len(boxes)
    if n % 4:
        raise ValueError("Train size must be divisible by four for exact flip groups")
    counts = np.array([np.bincount(b[:, 0].astype(int), minlength=7) for b in boxes])
    features = np.column_stack([counts > 0, counts]).astype(float)
    target = features.sum(axis=0)/4
    rng = np.random.default_rng(seed)
    best, best_score = None, float("inf")
    # Candidate search uses every class/box in multi-label images and exact group capacities.
    # Inverse-frequency normalization gives rare classes influence over allocation.
    for _ in range(candidates):
        permutation = rng.permutation(n)
        groups = permutation.reshape(4, n//4)
        totals = np.stack([features[g].sum(axis=0) for g in groups])
        score = float(np.square((totals-target)/np.maximum(target, 1)).sum())
        if score < best_score:
            best, best_score = groups.copy(), score
    result = {}
    for group, indices in zip(["H1", "H2", "V1", "V2"], best):
        for i in indices:
            result[int(i)] = group
    return result, best_score


def draw_boxes(image_path, boxes, names, title=""):
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((700, 700))
    draw = ImageDraw.Draw(image)
    w, h = image.size
    for c, x, y, bw, bh in boxes:
        color = ["#ff5050", "#00ccff", "#ffee00", "#00ee66", "#ff77ff", "#ff9900", "#aaaaFF"][int(c)]
        xyxy = [(x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h]
        draw.rectangle(xyxy, outline=color, width=3)
        draw.text((max(0, xyxy[0]), max(0, xyxy[1])), names[int(c)], fill=color, stroke_width=1, stroke_fill="black")
    if title:
        draw.text((4, 4), title, fill="white", stroke_width=1, stroke_fill="black")
    return image


def select_preview_indices(all_boxes, count, seed):
    chosen = []
    # Cover every available class and a multi-box source before random samples.
    for c in range(7):
        candidates = [i for i, b in enumerate(all_boxes) if c in b[:, 0]]
        if candidates:
            chosen.append(max(candidates, key=lambda i: len(all_boxes[i])))
    multi = [i for i, b in enumerate(all_boxes) if len(b) > 1]
    chosen.extend(multi[:3])
    chosen.extend(np.random.default_rng(seed).permutation(len(all_boxes)).tolist())
    return list(dict.fromkeys(chosen))[:count]


def augmented_root(cfg):
    return output(cfg, "prepared", "augmented")


def build_augmentation(cfg):
    baseline_yaml = prepare_original(cfg)
    _, names, splits = discover(cfg)
    source = baseline_yaml.parent
    items = pairs(source / "train/images", source / "train/labels")
    root = augmented_root(cfg)
    manifest = root / "augmentation_manifest.csv"
    if root.exists():
        if manifest.exists() and (root / "build_complete.json").exists():
            LOG.info("Augmentation exists; verifying rather than overwriting")
            audit_augmentation(cfg)
            return root / "data.yaml"
        raise FileExistsError(f"Incomplete augmentation directory: {root}. Preserve it and select a new output in config.")
    min_angle, max_angle = cfg["augmentation"]["min_degrees"], cfg["augmentation"]["max_degrees"]
    if not 0 < min_angle <= max_angle <= 5:
        raise ValueError("Rotation must satisfy 0 < min_degrees <= max_degrees <= 5")
    root.mkdir(parents=True)
    all_boxes = [read_boxes(lab, len(names), 2e-12)[0] for _, lab in items]
    allocation, score = allocate_flips(all_boxes, cfg["seed"], cfg["augmentation"]["allocation_candidates"])
    rng = np.random.default_rng(cfg["seed"])
    preview = set(select_preview_indices(all_boxes, cfg["augmentation"]["preview_count"], cfg["seed"]))
    rows = []
    for i, ((im, lab), boxes) in enumerate(zip(items, all_boxes)):
        # Hash-based prefix avoids stem collisions without depending on machine paths.
        relative = im.relative_to(source / "train/images")
        key = f"{i:04d}_{relative.stem}"
        original_source = splits["train"]["images"] / relative
        original_label = splits["train"]["labels"] / relative.with_suffix(".txt")
        new_im, new_lab = root / "train/images" / (key+im.suffix), root / "train/labels" / (key+".txt")
        checked_copy(im, new_im)
        checked_copy(lab, new_lab)
        variants = [("original", "original", {}, new_im, new_lab)]
        image = cv2.imread(str(im), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {im}")
        for transform in ("rotation", "horizontal_flip" if allocation[i].startswith("H") else "vertical_flip"):
            params = {}
            last_error = None
            for attempt in range(cfg["augmentation"]["retries"]):
                if transform == "rotation":
                    params = {"angle": float(rng.uniform(min_angle, max_angle)*rng.choice([-1, 1]))}
                try:
                    transformed, new_boxes, geometry = apply_transform(image, boxes, transform, params)
                    p_im = root / "train/images" / f"{key}__{transform}.png"
                    p_lab = root / "train/labels" / f"{key}__{transform}.txt"
                    write_boxes(p_lab, new_boxes)
                    if not cv2.imwrite(str(p_im), transformed, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                        raise RuntimeError(f"Failed writing {p_im}")
                    params.update(geometry)
                    variants.append((transform, "rotation" if transform == "rotation" else allocation[i], params, p_im, p_lab))
                    break
                except ValueError as exc:
                    last_error = exc
            else:
                raise RuntimeError(f"No valid {transform} for {im}: {last_error}")
        for transform, group, params, dst_im, dst_lab in variants:
            rows.append({"new_image": dst_im.relative_to(root).as_posix(), "new_label": dst_lab.relative_to(root).as_posix(),
                         "source_image": original_source.relative_to(splits["train"]["images"]).as_posix(),
                         "source_label": original_label.relative_to(splits["train"]["labels"]).as_posix(),
                         "transform": transform, "group": group, "parameters": json.dumps(params, sort_keys=True),
                         "source_sha256": sha256(original_source), "image_sha256": sha256(dst_im), "label_sha256": sha256(dst_lab)})
        if i in preview:
            panels = []
            for transform, _, params, dst_im, dst_lab in variants:
                b, _ = read_boxes(dst_lab, tolerance=2e-12)
                panel = draw_boxes(dst_im, b, names, f"{transform} {params.get('angle', '')}")
                panels.append(panel)
            canvas = Image.new("RGB", (sum(p.width for p in panels), max(p.height for p in panels)), (30, 30, 30))
            x = 0
            for panel in panels:
                canvas.paste(panel, (x, 0)); x += panel.width
            preview_dir = root / "previews"
            preview_dir.mkdir(exist_ok=True)
            canvas.save(preview_dir / f"{i:04d}.jpg", quality=92)
        if (i+1) % 50 == 0:
            LOG.info("Augmented %d/%d original images", i+1, len(items))
    write_csv(manifest, rows)
    # Independent Train; shared, byte-preserved Validation copy. Test is absent.
    write_yaml(root / "data.yaml", {"path": str(root), "train": "train/images", "val": str(source / "val/images"), "nc": len(names), "names": names})
    write_json(root / "build_complete.json", {"seed": cfg["seed"], "augmentation": cfg["augmentation"], "allocation_score": score, "images": len(rows)})
    audit_augmentation(cfg)
    return root / "data.yaml"


def audit_augmentation(cfg, pixel_check=True):
    _, names, splits = discover(cfg)
    root = augmented_root(cfg)
    verify_source(cfg)
    built = read_json(root / "build_complete.json")
    if built["seed"] != cfg["seed"] or built["augmentation"] != cfg["augmentation"]:
        raise RuntimeError("Augmentation config differs from existing build; use a new output directory.")
    with (root / "augmentation_manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    n = cfg["dataset"]["expected"]["train"]
    counts = Counter(r["transform"] for r in rows)
    expected = {"original": n, "rotation": n, "horizontal_flip": n//2, "vertical_flip": n//2}
    if counts != expected or Counter(r["group"] for r in rows) != {"original": n, "rotation": n, "H1": n//4, "H2": n//4, "V1": n//4, "V2": n//4}:
        raise ValueError(f"Wrong augmentation counts: {counts}")
    items = pairs(root / "train/images", root / "train/labels")
    if len(items) != 3*n or len(set(r["new_image"] for r in rows)) != 3*n or len(set(r["new_label"] for r in rows)) != 3*n:
        raise ValueError("Expected exactly 3156 unique image/label pairs")
    actual_images = {im.relative_to(root).as_posix() for im, _ in items}
    actual_labels = {lab.relative_to(root).as_posix() for _, lab in items}
    if actual_images != {r["new_image"] for r in rows} or actual_labels != {r["new_label"] for r in rows}:
        raise ValueError("Manifest does not match generated files")
    grouped = defaultdict(list)
    stats = statistics(items, len(names), 2e-12)
    before = statistics(pairs(**splits["train"]), len(names), cfg["dataset"]["rounding_tolerance"])
    if stats["boxes_per_class"] != [3*x for x in before["boxes_per_class"]] or stats["images_per_class"] != [3*x for x in before["images_per_class"]]:
        raise ValueError("Class boxes/images were lost or duplicated incorrectly")
    group_counts = defaultdict(lambda: np.zeros((2, len(names)), dtype=int))
    for index, row in enumerate(rows):
        if (row["transform"] == "horizontal_flip" and row["group"] not in ("H1", "H2")) or (row["transform"] == "vertical_flip" and row["group"] not in ("V1", "V2")):
            raise ValueError("Manifest flip group does not match the recorded transform")
        grouped[row["source_image"]].append(row["transform"])
        src_im = splits["train"]["images"] / row["source_image"]
        src_lab = splits["train"]["labels"] / row["source_label"]
        dst_im, dst_lab = root / row["new_image"], root / row["new_label"]
        if not src_im.resolve().is_relative_to(splits["train"]["images"]) or not src_lab.resolve().is_relative_to(splits["train"]["labels"]):
            raise ValueError("Manifest references a source outside Train")
        if not dst_im.resolve().is_relative_to(root.resolve()) or not dst_lab.resolve().is_relative_to(root.resolve()):
            raise ValueError("Manifest generated path is outside the augmentation directory")
        if sha256(src_im) != row["source_sha256"] or sha256(dst_im) != row["image_sha256"] or sha256(dst_lab) != row["label_sha256"]:
            raise ValueError(f"Manifest hash mismatch: {dst_im}")
        b, _ = read_boxes(src_lab, len(names), cfg["dataset"]["rounding_tolerance"])
        b = canonical_boxes(b)
        actual, _ = read_boxes(dst_lab, len(names), 2e-12)
        params = json.loads(row["parameters"])
        if row["transform"] == "original":
            expected_boxes = b
            if sha256(src_im) != sha256(dst_im):
                raise ValueError("Original image copy changed")
        else:
            with Image.open(src_im) as im:
                w, h = im.size
            if row["transform"] == "rotation":
                angle = abs(params["angle"])
                if not cfg["augmentation"]["min_degrees"] <= angle <= cfg["augmentation"]["max_degrees"]:
                    raise ValueError("Rotation outside configured range")
                matrix, size = rotation_matrix(w, h, params["angle"])
                if not np.allclose(matrix, params["matrix"], atol=1e-12) or list(size) != params["canvas"]:
                    raise ValueError("Rotation matrix/canvas mismatch")
                expected_boxes = transform_boxes(b, w, h, matrix, size)
            else:
                expected_boxes = b.copy()
                axis = 1 if row["transform"] == "horizontal_flip" else 2
                expected_boxes[:, axis] = 1-expected_boxes[:, axis]
            if pixel_check:
                source_image = cv2.imread(str(src_im))
                expected_image, _, _ = apply_transform(source_image, b, row["transform"], params)
                actual_image = cv2.imread(str(dst_im))
                if actual_image is None or not np.array_equal(expected_image, actual_image):
                    raise ValueError(f"Transformed pixels do not match recorded transform: {dst_im}")
        if expected_boxes.shape != actual.shape or not np.allclose(expected_boxes, actual, rtol=0, atol=2e-11):
            raise ValueError(f"Transformed labels do not match recorded geometry: {dst_lab}")
        cs = actual[:, 0].astype(int)
        group_counts[row["group"]][0] += np.bincount(cs, minlength=len(names))
        group_counts[row["group"]][1, np.unique(cs)] += 1
        if (index+1) % 300 == 0:
            LOG.info("Audited %d/%d generated pairs", index+1, len(rows))
    if set(grouped) != set(s.relative_to(splits["train"]["images"]).as_posix() for s, _ in pairs(**splits["train"])):
        raise ValueError("Missing or unexpected Train sources")
    if any(len(v) != 3 or v.count("original") != 1 or v.count("rotation") != 1 or sum(t.endswith("flip") for t in v) != 1 for v in grouped.values()):
        raise ValueError("Each source must have exactly one original, one rotation, one flip")
    baseline = output(cfg, "prepared", "baseline")
    for kind in ("images", "labels"):
        folder = baseline / "val" / kind
        val_hashes = {p.relative_to(folder).as_posix(): sha256(p) for p in folder.rglob("*") if p.is_file() and p.suffix != ".cache"}
        if val_hashes != read_json(output(cfg, "audit", "source_hashes.json"))["val"][kind]:
            raise ValueError("Validation copy differs from original")
    for yaml_path in [baseline / "data.yaml", root / "data.yaml"]:
        import yaml
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if "test" in data:
            raise ValueError("Training YAML must not expose Test")
    if (root / "val").exists() or (root / "test").exists():
        raise ValueError("Unexpected val/test files in augmentation dataset")
    verify_source(cfg)
    result = {"status": "passed", "pixel_check": pixel_check, "counts": dict(counts), "before": before, "after": stats,
              "groups": {k: {"boxes": v[0].tolist(), "images": v[1].tolist()} for k, v in group_counts.items()},
              "val_test_source_unchanged": True, "validation_copy_byte_identical": True}
    write_json(root / "augmentation_audit.json", result)
    lines = ["# Augmentation audit", "", "Status: PASSED", "", f"Pixel replay check: {pixel_check}",
             "", "|Transform|Images|", "|---|---:|"]
    lines += [f"|{k}|{v}|" for k, v in counts.items()]
    lines += ["", "|Class ID|Class|Original images|Augmented images|Original boxes|Augmented boxes|", "|---|---|---:|---:|---:|---:|"]
    for c, name in names.items():
        lines.append(f"|{c}|{name}|{before['images_per_class'][c]}|{stats['images_per_class'][c]}|{before['boxes_per_class'][c]}|{stats['boxes_per_class'][c]}|")
    lines += ["", f"Multi-box images: {before['multi_box_images']} -> {stats['multi_box_images']}.",
              "", "Every source has exactly one rotation and one flip. All source image corners and bounding-box corners remain within the expanded rotation canvas. No boxes are dropped. Labels are replayed from source geometry; PNG pixels are replayed when enabled.",
              "", "Validation is the same byte-identical copy for both runs. Validation/Test source hashes are unchanged. Test is absent from training YAMLs.",
              "", "Rotation uses neutral (114,114,114) padding matching YOLO LetterBox, with no reflected/synthesized texture. A rotated axis-aligned box can include more background. Pixel/geometry checks cannot establish semantic annotation correctness or rule out model sensitivity to padding; visually review previews/ before research training.",
              "", "Original Train annotation rounding is canonicalized identically for both experiments; see baseline/train_rounding_corrections.json."]
    (root / "augmentation_audit.md").write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Augmentation audit PASSED: %s", dict(counts))
    return result
