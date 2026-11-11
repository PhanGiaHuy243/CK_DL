# Validation comparison (matched)

|Metric|Baseline|Augmented|Difference (Augmented - Baseline)|
|---|---:|---:|---:|
|mAP@50:95|0.034968|0.044501|0.009533|
|mAP@50|0.087282|0.117491|0.030209|
|Precision|0.182027|0.612275|0.430248|
|Recall|0.099955|0.104770|0.004815|
|AP@50:95 baekra|0.000000|0.004342|0.004342|
|AP@50:95 color issues|0.000762|0.010788|0.010026|
|AP@50:95 contamination|0.033815|0.007029|-0.026786|
|AP@50:95 cut|0.047993|0.018339|-0.029654|
|AP@50:95 gray stitch|0.013585|0.051647|0.038062|
|AP@50:95 selvet|0.002309|0.025436|0.023127|
|AP@50:95 stain|0.146309|0.193925|0.047616|
|Training Time (seconds)|5272.368870|4721.984214|-550.384656|
|Optimizer steps|5940.000000|5940.000000|0.000000|
|Image presentations|94680.000000|94680.000000|0.000000|

Budget: actual steps 5940 / 5940; image presentations 94680 / 94680.
Last batch sizes: 12 / 4. No dropped final batch; no gradient accumulation; warmup is disabled in both runs.
LR is cosine/linear over actual optimizer updates, including endpoints; epoch scheduler is held constant. Equal total steps therefore have the same LR sequence. Final partial batches can give different sample weighting.
Validation checkpoint selection opportunities: 90 / 30 (one per epoch). Best checkpoint selection uses only Validation; differing checkpoint-selection frequency is a remaining limitation.
Training Time includes initialization, training, in-training Validation, checkpoint saving and the framework final best-checkpoint Validation. The separate common evaluation and prediction gallery are excluded.
AP values use IoU matching thresholds 0.50:0.05:0.95; evaluator iou is the separate NMS IoU. Precision/Recall follow the framework's maximum mean-F1 operating point. All scores use the 0-1 scale.

Per-class changes:
- baekra: improved (delta AP@50:95 0.004342).
- color issues: improved (delta AP@50:95 0.010026).
- contamination: decreased (delta AP@50:95 -0.026786).
- cut: decreased (delta AP@50:95 -0.029654).
- gray stitch: improved (delta AP@50:95 0.038062).
- selvet: improved (delta AP@50:95 0.023127).
- stain: improved (delta AP@50:95 0.047616).

Observed Validation mAP@50:95 increased; delta=+0.009533.

Single seed (42): this experiment does not establish statistical stability. Repeat with multiple seeds in independent output directories before stronger claims.

Confusion matrices and PR curves: each run's validation/ directory. Matched prediction images: gallery/. Ground truth accompanies both predictions; the gallery supports manual error analysis.