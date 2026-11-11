# Báo cáo triển khai — 11/11/2026

## Cập nhật kết quả cuối cùng

Toàn bộ pipeline GPU đã hoàn tất thành công lúc **05:06:23 ngày 11/11/2026, giờ Việt Nam**, sau **2 giờ 52 phút 50 giây**. Baseline chạy đủ 90 epoch, Augmented đủ 30 epoch; mỗi mô hình đúng 5940 optimizer step và 94680 lượt xem ảnh. Đã đánh giá chung, xuất biểu đồ/gallery và phát thông báo Windows.

mAP@50:95 trên Validation tăng từ **3,50% lên 4,45%** (+0,95 điểm phần trăm); mAP@50 tăng từ 8,73% lên 11,75%. AP tăng ở 5/7 lớp, giảm ở contamination và cut. Recall Augmented chỉ khoảng 10,48%; chất lượng tuyệt đối còn thấp. Một seed chưa chứng minh ổn định thống kê. Thống kê đầy đủ: [outputs/matched/comparison/tong_ket_vi.md](outputs/matched/comparison/tong_ket_vi.md).

Các đoạn bên dưới ghi lại diễn biến triển khai và khởi động trước khi hoàn tất; trạng thái cuối cùng là **completed**.

Đã hoàn thành pipeline YOLOv8n Baseline → offline augmentation → YOLOv8n Augmented → đánh giá và so sánh Validation. Đã tạo dữ liệu và chạy smoke test thật trên CPU. **Lượt full CPU đã được dừng theo yêu cầu chuyển sang GPU; chưa có kết luận về hiệu quả augmentation.** Log/checkpoint CPU được giữ trong `outputs/diagnostics/matched_cpu_before_gpu_20261111/`. Trạng thái lượt GPU mới sẽ được ghi tại `outputs/matched/pipeline_status.json`.

Máy thực tế có RTX 3070 Laptop 8 GiB, driver 566.07. Bản PyTorch CPU ban đầu không thấy CUDA; việc suy diễn rằng máy không có GPU từ kết quả đó là chưa đầy đủ. Đã tạo môi trường CUDA riêng và cấu hình `configs/full_gpu.yaml`: 640, batch 16, seed 42, Baseline 90/Augmented 30 epoch; dự kiến 5940 optimizer step và 94680 lượt xem ảnh cho mỗi mô hình. Hai run GPU khởi tạo lại từ pretrained và không trộn kết quả với CPU.

Đã cài torch 2.7.1+cu126 và torchvision 0.22.1+cu126 trong `.venv-gpu`. Kiểm tra CUDA thực tế đã qua cả phép nhân ma trận và torchvision NMS trên RTX 3070. `outputs/environment.json` lưu `cuda=true` và `cuda_runtime_check.passed=true`; danh sách phiên bản nằm trong `outputs/gpu_requirements.lock.txt`. Pipeline GPU đã khởi động nền bằng `run_full_experiment.py --config configs/full_gpu.yaml --device 0` và đang train Baseline; kết quả cuối cùng chưa có.

Đã xác nhận ít nhất 2 epoch GPU hoàn tất. Epoch đầu mất khoảng **39,0 giây**, đúng **66 optimizer step / 1052 ảnh**; epoch 2 đạt tổng 132 step / 2104 ảnh trong 80,1 giây. `best.pt` và `last.pt` đã lưu. Mức VRAM quan sát từ nvidia-smi là 4753 MiB trên 8192 MiB. So với epoch CPU đầu ở batch 2 (359,4 giây), epoch đầu của cấu hình GPU batch 16 nhanh hơn khoảng 9,2 lần; đây là so sánh thời gian của hai cấu hình chạy, không phải đo tác động riêng của GPU hay kết quả chất lượng mô hình. Snapshot GPU ở `outputs/matched/launch_verification.json`; pipeline vẫn tiếp tục tự động.

Lượt CPU trước đây dùng `configs/full_cpu.yaml` (640, batch 2, seed 42, 90/30 epoch), dự kiến 47340 optimizer step và 94680 lượt xem ảnh mỗi run. Lượt này được lưu để chẩn đoán và không dùng làm một phía của phép so sánh GPU. `run_full_experiment.py` tự chạy lần lượt hai thí nghiệm rồi đánh giá/so sánh, dừng nếu lỗi.

Trong lượt CPU, đã xác nhận epoch Baseline đầu tiên hoàn tất khoảng 359,4 giây, đúng 526 optimizer step và 1052 ảnh; checkpoint và CSV đã lưu. Snapshot cũ hiện nằm trong thư mục CPU được lưu ở diagnostics; đây không phải báo cáo hoàn tất training hoặc kết luận về metric cuối cùng.

## File đã tạo

- `configs/experiment.yaml`: cấu hình dữ liệu, seed, augmentation, huấn luyện, ngân sách, evaluator và smoke.
- `requirements.txt`, `.gitignore`, `README.md`, `IMPLEMENTATION_REPORT.md`.
- `audit_data.py`, `augment_dataset.py`, `check_environment.py`.
- `train_baseline.py`, `train_augmented.py`, `evaluate_models.py`, `compare_experiments.py`, `smoke_test.py`.
- `fabric_pipeline/__init__.py`, `common.py`, `data.py`, `augmentation.py`, `trainer.py`, `experiments.py`, `evaluation.py`, `comparison.py`, `cli.py`.
- `tests/test_geometry.py`: 6 kiểm tra hình học, phân bổ, định dạng manifest và từ chối nhãn sai.
- `.venv/`: môi trường dùng lại PyTorch hệ thống; Ultralytics/OpenCV được cài vào venv.
- `outputs/`: bản sao dữ liệu, augmentation, báo cáo audit, checkpoint pretrained, log và artifact smoke.

Không sửa file nào trong `CHENAB/` hoặc `CHENAB.zip`. Không chia lại dữ liệu.

## Môi trường đã chạy

|Thành phần|Giá trị|
|---|---|
|Python|3.10.0|
|PyTorch|2.14.0+cpu|
|torchvision|0.29.0|
|Ultralytics|8.3.221, được pin và kiểm tra runtime|
|OpenCV|4.11.0.86|
|NumPy|1.26.4|
|Thiết bị|Intel Core i7-12650H, CPU; CUDA không khả dụng|
|Seed|42|

Kiểm tra toán tử torchvision NMS đã qua. Checkpoint chính thức `yolov8n.pt` được tải từ release `v8.3.0` của `ultralytics/assets`, SHA256 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`; cả hai smoke run dùng cùng checksum này.

Thông tin máy và phiên bản: `outputs/environment.json`. Mỗi run lưu `args.yaml`, `experiment_config.yaml`, `requirements.lock.txt` và `run_summary.json`.

## Kết quả kiểm tra dataset

|Split|Ảnh|Nhãn|Box|Ảnh có nhiều box|
|---|---:|---:|---:|---:|
|Train|1052|1052|1520|260|
|Validation|225|225|326|49|
|Test|225|225|327|55|

Tất cả ảnh gốc là JPG, nhãn là TXT YOLO. Mapping theo YAML: 0 baekra, 1 color issues, 2 contamination, 3 cut, 4 gray stitch, 5 selvet, 6 stain. Không phát hiện nhóm ảnh byte-identical giữa các split. Kiểm tra thêm trên toàn bộ JPG: không thiếu dấu kết thúc JPEG, không có EXIF orientation khác 1.

Phát hiện sai số làm tròn ở 18 dòng/file nhãn Train, 5 Validation và 6 Test. Vượt biên lớn nhất là khoảng `9.765625e-6` normalized. Dữ liệu gốc được giữ nguyên; chỉ chuẩn hóa biên nhãn Train trong bản sao dùng chung cho hai thí nghiệm. Validation/Test không sửa. Các trường hợp cụ thể nằm trong audit JSON.

Artifact: `outputs/audit/dataset_audit.md`, `dataset_audit.json`, `class_counts.csv`, `source_hashes.json`; log sửa sai số ở `outputs/prepared/baseline/train_rounding_corrections.json`.

## Augmentation — PASSED

|Nhóm|Ảnh|
|---|---:|
|Original|1052|
|Rotation|1052|
|H1 / H2|263 / 263|
|V1 / V2|263 / 263|
|Horizontal tổng|526|
|Vertical tổng|526|
|Tổng Train|3156|

Có đúng **3156 ảnh và 3156 nhãn, 4560 box, 780 ảnh nhiều box**. Mỗi nguồn có đúng một rotation và một flip. Tổng box và số ảnh chứa từng lớp tăng đúng ba lần. Canvas xoay bảo toàn các góc ảnh/box, góc có độ lớn 1–5°, ảnh biến đổi lưu PNG lossless. Bộ ảnh Train augmented chiếm khoảng 3,66 GiB.

Đã kiểm tra toàn bộ nhãn, class ID, biên box, hash manifest, replay hình học và **replay pixel của tất cả biến thể**. Đã xác nhận Validation copy giống nguồn từng byte, hash dữ liệu gốc cả ba split không thay đổi và Test không có trong YAML train/evaluate. Kiểm tra bổ sung mapping H/V với nhóm phân bổ cũng qua.

Artifact: `outputs/prepared/augmented/augmentation_manifest.csv`, `augmentation_audit.md`, `augmentation_audit.json`, `previews/` (21 ảnh ghép có box). Đã xem trực tiếp các preview 0002 (lỗi sát mép), 0145 (10 box baekra), 0642 (9 box selvet): nội dung và box đi theo phép biến đổi.

Kiểm tra pixel/hình học không chứng minh được chất lượng ngữ nghĩa của mọi nhãn. Padding trung tính 114 tránh phản chiếu/tổng hợp thêm vân vải, nhưng có thể tạo tín hiệu biên cho mô hình; cần xem preview trước khi train nghiên cứu. Box song song trục sau xoay chứa thêm nền.

## Smoke test — PASSED

|Thông số thực đo|Baseline|Augmented|
|---|---:|---:|
|Train images|8|24|
|Validation images|4|4|
|Input size|128|128|
|Batch|2|2|
|Epoch|3|1|
|Optimizer steps|12|12|
|Lượt xem ảnh|24|24|
|Thời gian lời gọi train, giây|22,4|10,8|

Hai chuỗi learning rate thực tế khớp nhau (`actual_lr_sequences_equal=true`), không warmup, accumulation=1. Dataloader kiểm tra runtime chỉ chứa `LetterBox` và `Format`, `augment=false`, `bgr=0`, không Albumentations. Có `best.pt`, `last.pt`, `results.csv`, `results.png`, metric/AP từng lớp, confusion matrices, PR/P/R/F1 curves, bảng/biểu đồ so sánh và gallery chung kèm ground truth.

`outputs/smoke/smoke_status.json` ghi `status=passed`, `scientific_result=false`. Đối chiếu ngân sách ở `outputs/smoke/comparison/budget_check.json`; bảng smoke ở `comparison.md`, ảnh dự đoán ở `gallery/`. Các số đo trên subset này chỉ kiểm tra pipeline, không dùng để kết luận augmentation tốt hơn hay kém hơn.

6/6 kiểm tra hình học đã qua; cả 18 file Python được kiểm tra cú pháp. Không chạy full training hoặc inference trên Test.

## Lỗi đã phát hiện và xử lý

|Vấn đề|Xử lý|
|---|---|
|YAML chứa đường dẫn ổ D của máy khác|Tự resolve thư mục YAML hiện tại khi đường dẫn khai báo không tồn tại; tạo YAML chạy riêng, không sửa nguồn.|
|Nhãn vượt biên rất nhỏ do làm tròn|Log chi tiết, chỉ canonicalize Train copy dùng chung; sai số lớn hơn ngưỡng sẽ dừng.|
|Kích thước canvas kiểu NumPy không ghi được JSON|Đổi sang số Python, thêm regression test cho metadata rotation.|
|Sandbox không tải được dependency/checkpoint|Cài/tải bằng quyền mạng phù hợp; dependency nằm trong venv và checkpoint được xác nhận SHA256.|
|Windows sandbox chặn pipe khi Ultralytics quét nhãn|Chạy lại riêng smoke test với quyền thực thi phù hợp; quét nhãn và train/evaluate đã qua.|
|Ultralytics mặc định ghi settings ngoài workspace|Đặt YOLO_CONFIG_DIR bên trong outputs và tạo sẵn thư mục; tắt các integration logging bên ngoài.|

Artifact các lần lỗi được giữ trong `outputs/diagnostics/`, không trộn với dataset hoặc run hoàn tất.

## Các bước tiếp theo để có kết quả nghiên cứu

1. Xem `outputs/prepared/augmented/previews/` và báo cáo audit; chọn CPU/GPU, batch và input size trong config trước khi chạy cả hai thí nghiệm. Với CUDA cần cài cặp torch/torchvision CUDA phù hợp rồi kiểm tra lại môi trường.
2. Chạy lần lượt các lệnh dưới đây từ terminal PowerShell của VS Code tại thư mục dự án.

```powershell
.\.venv\Scripts\python.exe check_environment.py
.\.venv\Scripts\python.exe train_baseline.py --budget matched
.\.venv\Scripts\python.exe train_augmented.py --budget matched
.\.venv\Scripts\python.exe evaluate_models.py --budget matched
.\.venv\Scripts\python.exe compare_experiments.py --budget matched
```

Với cấu hình mặc định, **dự kiến** Baseline 90 và Augmented 30 epoch, batch 16, cùng 5940 optimizer step và 94680 lượt xem ảnh. Script sẽ đếm và kiểm tra lại khi train thật; batch cuối 12/4 khác nhau. LR được chuẩn hóa theo update để tránh khác biệt scheduler theo epoch. Best checkpoint chọn theo Validation mỗi epoch, nên hai run có số cơ hội chọn best khác nhau; báo cáo ghi rõ giới hạn này.

Kết quả chính thức sẽ nằm trong `outputs/matched/`. Cần lặp nhiều seed với output riêng để đánh giá độ ổn định thống kê. Hướng dẫn cài mới, chạy từng bước, chạy lại audit/smoke, chế độ same_epochs và cấu trúc artifact đầy đủ nằm trong `README.md`.
