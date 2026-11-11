# Dataset audit

Source YAML: `C:\phangiahuy\DL\DL_ck\CHENAB\data.yaml`

Original split assignments preserved. Test inspected for integrity only.

|Split|Images|Boxes|Multi-box images|Rounding warnings|
|---|---:|---:|---:|---:|
|train|1052|1520|260|18|
|val|225|326|49|5|
|test|225|327|55|6|

Cross-split byte-identical image groups: 0.

Small boundary rounding is logged in JSON. Only staged Train labels are canonicalized; Validation/Test bytes are unchanged.