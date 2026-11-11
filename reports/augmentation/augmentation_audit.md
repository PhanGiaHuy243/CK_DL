# Augmentation audit

Status: PASSED

Pixel replay check: True

|Transform|Images|
|---|---:|
|original|1052|
|rotation|1052|
|vertical_flip|526|
|horizontal_flip|526|

|Class ID|Class|Original images|Augmented images|Original boxes|Augmented boxes|
|---|---|---:|---:|---:|---:|
|0|baekra|184|552|303|909|
|1|color issues|70|210|147|441|
|2|contamination|107|321|121|363|
|3|cut|122|366|147|441|
|4|gray stitch|221|663|288|864|
|5|selvet|101|303|161|483|
|6|stain|249|747|353|1059|

Multi-box images: 260 -> 780.

Every source has exactly one rotation and one flip. All source image corners and bounding-box corners remain within the expanded rotation canvas. No boxes are dropped. Labels are replayed from source geometry; PNG pixels are replayed when enabled.

Validation is the same byte-identical copy for both runs. Validation/Test source hashes are unchanged. Test is absent from training YAMLs.

Rotation uses neutral (114,114,114) padding matching YOLO LetterBox, with no reflected/synthesized texture. A rotated axis-aligned box can include more background. Pixel/geometry checks cannot establish semantic annotation correctness or rule out model sensitivity to padding; visually review previews/ before research training.

Original Train annotation rounding is canonicalized identically for both experiments; see baseline/train_rounding_corrections.json.