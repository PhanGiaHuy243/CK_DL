from __future__ import annotations

import math
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import LOG, output, read_json, write_csv, write_json
from .experiments import run_directory


def format_value(value):
    return "N/A" if value is None else f"{value:.6f}"


def compare(cfg, budget="matched", smoke=False):
    summaries, evaluated, traces = {}, {}, {}
    for experiment in ("baseline", "augmented"):
        run = run_directory(cfg, experiment, budget, smoke)
        summaries[experiment] = read_json(run / "run_summary.json")
        evaluated[experiment] = read_json(run / "validation_metrics.json")
        with (run / "optimizer_steps.csv").open(encoding="utf-8-sig", newline="") as f:
            traces[experiment] = list(csv.DictReader(f))
        trace = traces[experiment]
        if len(trace) != summaries[experiment]["optimizer_steps"] or [int(r["step"]) for r in trace] != list(range(1, len(trace)+1)):
            raise ValueError(f"Optimizer step trace differs from summary: {experiment}")
        if int(trace[-1]["image_presentations"]) != summaries[experiment]["image_presentations"]:
            raise ValueError(f"Image presentation trace differs from summary: {experiment}")
        if evaluated[experiment]["checkpoint_sha256"] != summaries[experiment]["best_sha256"]:
            raise ValueError(f"Evaluation does not belong to saved best checkpoint: {experiment}")
    b, a = summaries["baseline"], summaries["augmented"]
    eb, ea = evaluated["baseline"], evaluated["augmented"]
    for key in ["checkpoint_sha256", "seed", "batch", "scheduler", "warmup_steps", "gradient_accumulation", "source_hashes_sha256"]:
        if b[key] != a[key]:
            raise ValueError(f"Incomparable experiment metadata: {key}")
    # Compare every library hyperparameter except dataset identity and planned epoch count.
    allowed_differences = {"data", "epochs", "name", "save_dir"}
    for key in set(b["resolved_args"]) | set(a["resolved_args"]):
        if key not in allowed_differences and b["resolved_args"].get(key) != a["resolved_args"].get(key):
            raise ValueError(f"Training hyperparameters differ unexpectedly: {key}")
    for key in ["settings", "validation_hashes", "ap_matching_iou_thresholds", "evaluator", "seed", "validation_images"]:
        if eb[key] != ea[key]:
            raise ValueError(f"Evaluation settings/Validation mismatch: {key}")
    if b["environment"]["packages"] != a["environment"]["packages"]:
        raise ValueError("Library versions differ between runs")
    if b["scientific_result"] != a["scientific_result"] or b["scientific_result"] == smoke:
        raise ValueError("Smoke/official experiment artifacts must not be mixed")
    relative_steps = abs(a["optimizer_steps"]-b["optimizer_steps"])/max(b["optimizer_steps"], 1)
    exposure_match = b["image_presentations"] == a["image_presentations"]
    steps_match = relative_steps <= 0.01
    lr_sequence_equal = None
    if b["optimizer_steps"] == a["optimizer_steps"]:
        lr_sequence_equal = bool(np.allclose([float(r["lr"]) for r in traces["baseline"]],
                                             [float(r["lr"]) for r in traces["augmented"]], rtol=0, atol=1e-12))
        if not lr_sequence_equal:
            raise ValueError("Actual learning-rate sequences differ for equal-step runs")
    if (budget == "matched" or smoke) and not (steps_match or exposure_match):
        raise ValueError("Neither image presentations nor optimizer steps match the main training budget")
    rows = []
    for label, key in [("mAP@50:95", "mAP50_95"), ("mAP@50", "mAP50"), ("Precision", "precision"), ("Recall", "recall")]:
        bv, av = eb["metrics"][key], ea["metrics"][key]
        rows.append({"Metric": label, "Baseline": bv, "Augmented": av, "Difference": av-bv if bv is not None and av is not None else None})
    class_rows = []
    for bc, ac in zip(eb["per_class"], ea["per_class"]):
        if bc["class_id"] != ac["class_id"] or bc["class"] != ac["class"]:
            raise ValueError("Class mapping differs")
        bv, av = bc["AP50_95"], ac["AP50_95"]
        delta = av-bv if bv is not None and av is not None else None
        rows.append({"Metric": f"AP@50:95 {bc['class']}", "Baseline": bv, "Augmented": av, "Difference": delta})
        class_rows.append({"class_id": bc["class_id"], "class": bc["class"], "baseline_AP50_95": bv,
                           "augmented_AP50_95": av, "delta_AP50_95": delta,
                           "baseline_AP50": bc["AP50"], "augmented_AP50": ac["AP50"],
                           "change": "unavailable" if delta is None else "improved" if delta > 0 else "decreased" if delta < 0 else "unchanged"})
    for label, key in [("Training Time (seconds)", "training_seconds"), ("Optimizer steps", "optimizer_steps"), ("Image presentations", "image_presentations")]:
        rows.append({"Metric": label, "Baseline": b[key], "Augmented": a[key], "Difference": a[key]-b[key]})
    root = output(cfg, "smoke" if smoke else budget, "comparison")
    write_csv(root / "comparison.csv", rows)
    write_csv(root / "per_class_comparison.csv", class_rows)
    write_json(root / "budget_check.json", {"optimizer_steps_relative_difference": relative_steps, "steps_within_one_percent": steps_match,
                                            "image_presentations_equal": exposure_match, "actual_lr_sequences_equal": lr_sequence_equal,
                                            "baseline": b["optimizer_steps"], "augmented": a["optimizer_steps"]})
    for filename, subset, title in [("metrics_comparison.png", rows[:4], "Validation metrics"),
                                     ("per_class_ap.png", rows[4:4+len(class_rows)], "Per-class AP@50:95")]:
        x = np.arange(len(subset))
        fig, ax = plt.subplots(figsize=(max(8, len(subset)*1.6), 5))
        ax.bar(x-0.2, [r["Baseline"] if r["Baseline"] is not None else np.nan for r in subset], .4, label="Baseline")
        ax.bar(x+0.2, [r["Augmented"] if r["Augmented"] is not None else np.nan for r in subset], .4, label="Augmented")
        ax.set_xticks(x, [r["Metric"].replace("AP@50:95 ", "") for r in subset], rotation=20, ha="right")
        ax.set_ylim(0, 1); ax.set_ylabel("Score (0-1)"); ax.legend()
        ax.set_title(("SMOKE ONLY - " if smoke else "")+title)
        fig.tight_layout(); fig.savefig(root / filename, dpi=180); plt.close(fig)
    lines = ["# " + ("SMOKE TEST ONLY - not research evidence" if smoke else f"Validation comparison ({budget})"), "",
             "|Metric|Baseline|Augmented|Difference (Augmented - Baseline)|", "|---|---:|---:|---:|"]
    for r in rows:
        lines.append(f"|{r['Metric']}|{format_value(r['Baseline'])}|{format_value(r['Augmented'])}|{format_value(r['Difference'])}|")
    lines += ["", f"Budget: actual steps {b['optimizer_steps']} / {a['optimizer_steps']}; image presentations {b['image_presentations']} / {a['image_presentations']}.",
              f"Last batch sizes: {b['last_batch_size']} / {a['last_batch_size']}. No dropped final batch; no gradient accumulation; warmup is disabled in both runs.",
              "LR is cosine/linear over actual optimizer updates, including endpoints; epoch scheduler is held constant. Equal total steps therefore have the same LR sequence. Final partial batches can give different sample weighting.",
              f"Validation checkpoint selection opportunities: {b['epochs']} / {a['epochs']} (one per epoch). Best checkpoint selection uses only Validation; differing checkpoint-selection frequency is a remaining limitation.",
              "Training Time includes initialization, training, in-training Validation, checkpoint saving and the framework final best-checkpoint Validation. The separate common evaluation and prediction gallery are excluded.",
              "AP values use IoU matching thresholds 0.50:0.05:0.95; evaluator iou is the separate NMS IoU. Precision/Recall follow the framework's maximum mean-F1 operating point. All scores use the 0-1 scale.",
              "", "Per-class changes:"]
    lines += [f"- {r['class']}: {r['change']} (delta AP@50:95 {format_value(r['delta_AP50_95'])})." for r in class_rows]
    delta = rows[0]["Difference"]
    if smoke:
        lines += ["", "No research conclusion: these are tiny-subset smoke-test measurements only."]
    elif delta is None:
        lines += ["", "No conclusion: mAP@50:95 was unavailable."]
    else:
        change = "increased" if delta > 0 else "decreased" if delta < 0 else "did not change"
        lines += ["", f"Observed Validation mAP@50:95 {change}; delta={delta:+.6f}."]
    if budget == "same_epochs" and not smoke:
        lines += ["", "Same-epoch comparison has a larger training budget for Augmented and cannot isolate the augmentation effect."]
    lines += ["", f"Single seed ({b['seed']}): this experiment does not establish statistical stability. Repeat with multiple seeds in independent output directories before stronger claims.",
              "", "Confusion matrices and PR curves: each run's validation/ directory. Matched prediction images: gallery/. Ground truth accompanies both predictions; the gallery supports manual error analysis."]
    (root / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Comparison saved to %s", root)
    return rows
