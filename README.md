# CHENAB Textile: YOLOv8n Baseline và Offline Augmentation

Repo chia sẻ: [PhanGiaHuy243/CK_DL](https://github.com/PhanGiaHuy243/CK_DL). Người tải repo xem [SHARING.md](SHARING.md) để mở kết quả, dùng hai checkpoint có sẵn hoặc tái lập thí nghiệm trên máy khác. Dataset gốc và môi trường Python được cung cấp/cài riêng.

Pipeline so sánh trên **cùng 225 ảnh Validation**, giữ split gốc và không dùng Test để train, chọn checkpoint hoặc chọn tham số. Cấu hình cơ sở là `configs/experiment.yaml`; lượt GPU dùng `configs/full_gpu.yaml`. Dataset gốc chỉ được đọc; cache YOLO và các bản dữ liệu thực nghiệm nằm trong `outputs/`.

Kết quả triển khai, audit và smoke test thực tế: xem [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).

**Lượt GPU đã hoàn tất cả 5 bước lúc 05:06:23 ngày 11/11/2026 (giờ Việt Nam), tổng 2 giờ 52 phút 50 giây.** Xem [báo cáo kết quả tiếng Việt](outputs/matched/comparison/tong_ket_vi.md). Baseline mAP@50:95 = 3,50%; Augmented = 4,45%, tăng 0,95 điểm phần trăm. Đây là một lượt seed 42 trên Validation; mức mAP và Recall tuyệt đối vẫn thấp.

## Full training trên RTX 3070 Laptop

Máy có **NVIDIA GeForce RTX 3070 Laptop GPU, 8 GiB VRAM**, driver 566.07. Lượt CPU trước đó đã được dừng theo yêu cầu chuyển GPU; checkpoint/log được giữ tại `outputs/diagnostics/matched_cpu_before_gpu_20261111/`. Kết luận cũ từ `CUDA=False` chỉ phản ánh bản PyTorch CPU, không phản ánh việc máy không có GPU.

Cấu hình GPU là `configs/full_gpu.yaml`: device 0, input 640, batch **16** cho cả Train và evaluator, seed 42, Baseline 90 epoch và Augmented 30 epoch. Hai mô hình khởi tạo mới từ cùng pretrained checkpoint, cùng **5940 optimizer step dự kiến** và **94680 lượt xem ảnh dự kiến** mỗi mô hình. Không resume từ lượt CPU.

`run_full_experiment.py` chạy tuần tự kiểm tra môi trường → Baseline → Augmented → đánh giá → so sánh. Mỗi bước chạy ở process riêng và có log riêng; nếu lỗi thì dừng chuỗi. `outputs/matched/pipeline_status.json` ghi bước hiện tại, PID, kết quả từng bước và lỗi nếu có. Không chỉnh `configs/full_gpu.yaml` trong lúc pipeline đang chạy; coordinator kiểm tra hash cấu hình giữa các bước. Một yêu cầu Windows ngăn tự Sleep được giữ trong thời gian coordinator chạy, không thay đổi cài đặt nguồn toàn hệ thống.

Đọc trạng thái và log bằng PowerShell:

```powershell
Get-Content -Raw -Encoding UTF8 outputs/matched/pipeline_status.json
Get-Content -Encoding UTF8 outputs/matched/train_baseline.log -Tail 5 -Wait
```

`Ctrl+C` khi đang xem `Get-Content -Wait` chỉ đóng việc xem log; tiến trình train nền vẫn chạy. Khi chuyển sang Augmented, log là `outputs/matched/train_augmented.log`. `results.csv` trong từng run được cập nhật sau mỗi epoch; `weights/last.pt` và `best.pt` được lưu trong run. Giữ máy bật và cắm điện trong suốt lượt chạy.

Lệnh khởi động coordinator là:

```powershell
.\.venv-gpu\Scripts\python.exe run_full_experiment.py --config configs/full_gpu.yaml --device 0
```

Kiểm tra `pipeline_status.json` trước khi chạy lệnh trên để tránh chạy thêm một pipeline đang hoạt động. Coordinator từ chối ghi đè status hoặc các run đã có. Trạng thái `completed` cùng đường dẫn `comparison.md` chỉ xuất hiện sau khi tất cả các bước thành công.

### Thông báo khi hoàn tất

`notify_when_done.py` theo dõi độc lập lượt chạy hiện tại mỗi 15 giây. Khi Baseline, Augmented, đánh giá và so sánh đều hoàn tất và báo cáo tồn tại, một cửa sổ Windows kèm âm báo sẽ xuất hiện. Cửa sổ giữ lại tới khi được đóng; bấm OK mở báo cáo. Khi pipeline lỗi hoặc controller dừng bất thường, thông báo lỗi sẽ xuất hiện thay cho thông báo thành công. Cần giữ máy bật; âm báo phụ thuộc âm lượng hệ thống.

Bộ theo dõi đã được khởi động nền cho lượt GPU hiện tại. Trạng thái của nó ở `outputs/matched/notification_status.json`; nội dung thông báo khi phát sinh được lưu ở `outputs/matched/notification.txt`. Không cần khởi động thêm bản thứ hai. Khi cần gắn vào một lượt mới, dùng:

```powershell
.\.venv-gpu\Scripts\python.exe notify_when_done.py --status-file outputs/matched/pipeline_status.json
```

Thông báo xuất hiện trên desktop Windows đang chạy training. Cách hiển thị dùng [MessageBoxW của Windows](https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-messageboxw) và [winsound của Python](https://docs.python.org/3/library/winsound.html).

## Dữ liệu thực tế

`CHENAB/data.yaml` có `path` cũ của máy khác. Pipeline tự dùng thư mục chứa YAML nếu đường dẫn đó không tồn tại, không sửa YAML gốc. Có thể đổi `dataset.yaml` trong config sang đường dẫn tương đối repo hoặc đường dẫn tuyệt đối.

|Split|Thư mục ảnh / nhãn|Cặp ảnh-nhãn|Box|
|---|---|---:|---:|
|Train|`CHENAB/train/images`, `labels`|1052|1520|
|Validation|`CHENAB/valid/images`, `labels`|225|326|
|Test|`CHENAB/test/images`, `labels`|225|327|

Ảnh gốc là JPG, nhãn TXT YOLO `class xc yc w h`. Mapping đọc từ YAML: `0 baekra`, `1 color issues`, `2 contamination`, `3 cut`, `4 gray stitch`, `5 selvet`, `6 stain`. Không suy diễn mapping từ thứ tự tên trong đề bài. Một số tên tệp nguồn có chữ `augmented`; pipeline giữ nguyên bộ Train đã được cung cấp, không tự coi đó là dữ liệu cần loại bỏ.

## Cài đặt trên Windows / terminal PowerShell của VS Code

Python 3.10 trở lên. Smoke test ban đầu dùng torch 2.14.0 CPU và torchvision 0.29.0. Môi trường GPU riêng `.venv-gpu` dùng torch **2.7.1+cu126**, torchvision **0.22.1+cu126**, Ultralytics 8.3.221 và OpenCV 4.11.0.86. Mỗi lần train lưu phiên bản và `requirements.lock.txt`; muốn tái lập trên máy khác cần cùng cặp torch/torchvision và các phiên bản trong lock.

Cài môi trường GPU riêng:

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --no-cache-dir -r requirements-gpu.txt
.\.venv-gpu\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple
.\.venv-gpu\Scripts\python.exe check_environment.py --config configs/full_gpu.yaml
```

Cặp phiên bản được công bố tại [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/). Driver 566.07 hiện có đáp ứng yêu cầu CUDA 12.6 trên Windows theo [NVIDIA CUDA 12.6 Release Notes](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-toolkit-release-notes/index.html). Không cần cài thêm CUDA Toolkit để chạy các wheel này. Kiểm tra môi trường thực hiện phép nhân ma trận và torchvision NMS ngay trên CUDA; cấu hình GPU sẽ báo lỗi nếu CUDA không khả dụng.

Nếu cần một môi trường CPU riêng:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Nếu máy đã có cặp torch/torchvision phù hợp, có thể tạo môi trường bằng `python -m venv --system-site-packages .venv`; đây là cách môi trường hiện tại được tạo để dùng PyTorch có sẵn. Không cần activate nên không gặp lỗi execution policy của PowerShell.

Với GPU NVIDIA, chọn lệnh cài cặp torch/torchvision CUDA theo [PyTorch Get Started](https://pytorch.org/get-started/locally/), sau đó cài `requirements.txt`. Chỉ dùng một GPU cho adapter đếm ngân sách hiện tại. `device: auto` tự chọn GPU 0 nếu CUDA có, ngược lại CPU. `--device cpu` hoặc `--device 0` ghi đè cấu hình. CPU chạy được nhưng full training ở 640 có thể lâu.

Kiểm tra phiên bản, CUDA và toán tử NMS của torchvision, tải checkpoint chính thức:

```powershell
.\.venv\Scripts\python.exe check_environment.py --download-weights
```

Checkpoint nằm ở `outputs/weights/yolov8n.pt`, kèm SHA256. Nếu tải mạng lỗi, tải từ URL được thông báo vào vị trí đó rồi chạy lại lệnh trên. Cả hai thí nghiệm tạo một đối tượng YOLO mới từ cùng checkpoint này; pipeline kiểm tra checksum, không resume từ Baseline. Checksum được pin cho asset chính thức `ultralytics/assets`, release `v8.3.0`: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.

## Chạy theo đúng thứ tự

Các lệnh dưới đây chạy tại thư mục gốc repo. Mỗi lệnh hỗ trợ `--config configs/experiment.yaml`.

```powershell
# 1. Audit dữ liệu gốc, lưu hash và chuẩn bị bản sao Train/Validation
.\.venv\Scripts\python.exe audit_data.py

# 2. Tạo đúng 3156 ảnh Train, manifest, ảnh minh họa và audit toàn bộ
.\.venv\Scripts\python.exe augment_dataset.py

# 3. Kiểm tra lại dữ liệu đã tạo nếu cần (bao gồm replay pixel)
.\.venv\Scripts\python.exe augment_dataset.py --audit-only

# 4. Smoke test đầu-cuối trên CPU; 8 Train gốc, 24 Train augmented, 4 Validation
.\.venv\Scripts\python.exe smoke_test.py --device cpu

# 5. Chỉ chạy sau khi đã xem audit/previews và môi trường đáp ứng
.\.venv\Scripts\python.exe train_baseline.py --budget matched
.\.venv\Scripts\python.exe train_augmented.py --budget matched

# 6. Đánh giá lại hai best.pt bằng cùng evaluator, tạo gallery chung
.\.venv\Scripts\python.exe evaluate_models.py --budget matched

# 7. Xuất CSV, Markdown, biểu đồ, chênh lệch và kiểm tra ngân sách
.\.venv\Scripts\python.exe compare_experiments.py --budget matched
```

Mỗi script train đã lưu metric Validation của best.pt. Bước đánh giá chung vẫn cần chạy để tạo gallery dự đoán. Smoke test tự đánh giá và tạo bảng/biểu đồ/gallery trong `outputs/smoke/`; các số đó **không phải kết quả nghiên cứu**, không dùng kết luận augmentation có hiệu quả.

Full training không tự chạy khi audit, augmentation hoặc smoke. Hiện chưa có kết quả chính thức Baseline/Augmented nên chưa có kết luận về mAP.

## Tiền xử lý và vô hiệu hóa augmentation mặc định

`PreprocessingOnlyTrainer` tạo `YOLODataset(augment=False)` cho cả Train và Validation. Kiểm tra runtime yêu cầu đúng hai transform `LetterBox` và `Format`, `bgr=0`. Vì nhánh `v8_transforms` không được gọi, các biến đổi Albumentations tự động cũng không được khởi tạo, kể cả khi package Albumentations có trên máy. Chuẩn hóa tensor `/255` vẫn do `DetectionTrainer.preprocess_batch` thực hiện.

Ngoài việc bypass nhánh augmentation, cấu hình train đặt bằng 0 cho HSV, rotate, translate, scale, shear, perspective, hai flip, mosaic, mixup, cutmix, copy_paste, bgr, erasing; `auto_augment=null`, `multi_scale=false`, `close_mosaic=0`. `transform_audit.json` ghi chuỗi transform thật và số ảnh thật trong dataloader. Tệp `args.yaml` của Ultralytics và `run_summary.json` ghi toàn bộ cấu hình đã resolve, bao gồm các giá trị mặc định từ bản thư viện được pin.

Thiết kế dựa trên API chính thức [YOLODataset.build_transforms](https://docs.ultralytics.com/reference/data/dataset/), [Albumentations và LetterBox](https://docs.ultralytics.com/reference/data/augment/) và [training configuration](https://docs.ultralytics.com/usage/cfg/). Adapter kiểm tra đúng Ultralytics 8.3.221 để tránh vô tình thay đổi hành vi khi cập nhật thư viện.

## Augmentation và kiểm định

Mỗi ảnh gốc có ba mục trong manifest: original, rotation, một flip. Nhóm H1/H2/V1/V2 mỗi nhóm 263 nguồn khác nhau, tạo 526 horizontal và 526 vertical. Phân bổ dùng seed 42, thử nhiều hoán vị có kích thước nhóm chính xác rồi chọn phương án cân bằng theo cả số ảnh chứa lớp và số box của **tất cả** lớp trong mỗi ảnh. Đây là cân bằng xấp xỉ, không thay đổi tổng ảnh; số phân bổ thực tế theo nhóm nằm trong audit JSON.

Xoay ngẫu nhiên có dấu trong `[1°, 5°]`, không gần 0, scale=1, canvas mở rộng theo toàn bộ bốn góc ảnh. Bốn góc mỗi box được xoay cùng affine matrix rồi bao bằng box song song trục. Không bỏ box. Nếu hình học không hợp lệ, thử lại và báo lỗi khi hết lượt. Flip dùng `xc'=1-xc` hoặc `yc'=1-yc`. Không thêm các phép thay đổi màu, crop hoặc ghép ảnh.

Ảnh biến đổi lưu PNG lossless để có thể replay pixel chính xác, ảnh original được copy byte nguyên vẹn. Cần dung lượng lớn hơn JPG gốc; không resize hoặc nén giảm chất lượng ảnh để đạt số lượng.

Padding xoay dùng màu trung tính `(114,114,114)` như YOLO LetterBox, không phản chiếu hoặc tổng hợp vân vải/lỗi giả. Viền padding có thể ảnh hưởng mô hình; không thể khẳng định tính đúng ngữ nghĩa bằng kiểm tra tọa độ. Xem các ảnh `previews/` có original/rotation/flip và box, đặc biệt lỗi sát biên và ảnh nhiều box, trước khi train nghiên cứu. Box song song trục sau xoay sẽ chứa thêm nền; đây là hệ quả của biểu diễn detection bbox.

Audit kiểm tra chính xác số ảnh/nhãn, IDs, tọa độ finite/normalized, box nằm trong ảnh, không mất box, replay hình học từ nhãn nguồn, replay toàn bộ pixel từ ảnh nguồn theo phép biến đổi, hash manifest, mỗi nguồn một rotation/một flip, tổng ảnh/box từng lớp tăng đúng ba lần. Preview bao phủ các lớp và ảnh nhiều box; kiểm tra pixel/hình học toàn bộ không thay thế đánh giá nhãn bằng mắt.

Một số nhãn nguồn có sai số biên rất nhỏ do làm tròn (giới hạn chấp nhận `1e-5` normalized). Mọi trường hợp ghi trong `dataset_audit.json`. Chỉ nhãn Train ở bản sao được đưa về giao với miền `[0,1]`; cùng bản nhãn này dùng cho Baseline và nhóm original của Augmented. Nhãn gốc và Validation/Test không sửa. Sai số lớn hơn giới hạn, ID sai, NaN, box rỗng hoặc thiếu file đều dừng pipeline. Không tự giữ box sai để đạt số lượng.

Hash tất cả ảnh/nhãn gốc được giữ trong `source_hashes.json` và kiểm tra lại. Validation được copy nguyên byte một lần rồi dùng chung; không có Test trong YAML train/evaluate. Test chỉ được đọc để audit tính toàn vẹn. Audit không phát hiện ảnh byte-identical lọt giữa split; chưa loại trừ near-duplicate hoặc tương quan cùng cuộn vải.

## Ngân sách huấn luyện và giới hạn so sánh

Chế độ chính mặc định:

|Thông số|Baseline|Augmented|
|---|---:|---:|
|Train images|1052|3156|
|Epoch|90|30|
|Batch|16|16|
|Batch/epoch (không drop cuối)|66|198|
|Optimizer steps dự kiến|5940|5940|
|Lượt xem ảnh dự kiến|94680|94680|
|Kích thước batch cuối|12|4|

`nbs=batch` nên gradient accumulation bằng 1, cùng weight decay thực tế. `optimizer=SGD` được đặt rõ, tránh `auto` đổi chính sách theo kích thước dữ liệu. `patience=0` tắt early stopping, `amp=false` tránh bỏ optimizer step khi GradScaler gặp overflow, `warmup_epochs=0` giữ hai ngân sách tương đương. Adapter dừng khi loss không finite thay vì tự replay epoch.

Learning rate cosine (hoặc linear khi `cos_lr=false`) được tính **theo optimizer step**, không theo epoch. Bước đầu dùng `lr0`, bước cuối dùng `lr0*lrf`, nên hai chuỗi LR khớp khi tổng step khớp. Bộ đếm dựa trên hook thật của `optimizer.step`, lưu từng bước và số ảnh đã xem. Khi thay batch/epoch, script so sánh yêu cầu lượt xem ảnh bằng nhau **hoặc** số step chênh tối đa 1%; luôn báo cả hai. Đối chiếu `optimizer_steps.csv` và `budget_check.json` thay vì chỉ dùng số dự kiến.

Batch cuối khác nhau có thể làm trọng số mẫu khác nhau. Thứ tự shuffle, BatchNorm và phân bố batch cũng khác nhau. Validation chọn best mỗi epoch nên Baseline có 90 cơ hội chọn checkpoint và Augmented có 30; báo cáo nêu giới hạn này. Không dùng Test để giải quyết các khác biệt đó. Muốn kết luận ổn định cần nhiều seed và giữ nguyên thiết kế trước khi xem kết quả.

Chế độ phụ lưu riêng, cả hai chạy 30 epoch theo config:

```powershell
.\.venv\Scripts\python.exe train_baseline.py --budget same_epochs
.\.venv\Scripts\python.exe train_augmented.py --budget same_epochs
.\.venv\Scripts\python.exe evaluate_models.py --budget same_epochs
.\.venv\Scripts\python.exe compare_experiments.py --budget same_epochs
```

Không dùng cùng số epoch làm bằng chứng duy nhất về hiệu quả augmentation vì số update của Augmented nhiều hơn.

## Metric và artifact

Evaluator dùng cùng `conf=0.001`, NMS IoU `0.7`, `max_det=300`, input size và batch; FP32, không test-time augmentation. AP matching IoU là `0.50:0.05:0.95`, khác với NMS IoU. Precision/Recall lấy theo điểm mean-F1 tối ưu của evaluator. AP từng lớp xuất cả AP50 và AP50:95; lớp không xuất hiện trong subset smoke ghi N/A, không lấy mAP trung bình điền thay.

Gallery dùng ngưỡng hiển thị `0.25` để dễ xem, cùng ảnh cho cả hai mô hình, có ground truth cạnh hai dự đoán. Ngưỡng gallery không thay ngưỡng tính AP. Báo cáo delta và trạng thái tăng/giảm theo từng lớp; kết luận về mAP chỉ được tạo khi có kết quả train/evaluate thật. Một seed được ghi rõ là một thực nghiệm đơn.

```text
configs/experiment.yaml          # cấu hình tập trung
fabric_pipeline/                # data, augmentation, trainer, evaluation, comparison
audit_data.py                   # audit và bản sao dữ liệu
augment_dataset.py              # tạo hoặc audit lại augmentation
check_environment.py            # CUDA/NMS/phiên bản/checkpoint
train_baseline.py
train_augmented.py
evaluate_models.py
compare_experiments.py
smoke_test.py
outputs/
  audit/                        # dataset_audit.md/json, class_counts.csv, source_hashes.json
  environment.json
  logs/                         # log theo từng lần chạy (tên tệp dùng UTC)
  weights/                      # checkpoint pretrained và SHA256
  prepared/baseline/            # Train/Val copy, data.yaml, log rounding
  prepared/augmented/           # 3156 Train, data.yaml (Val dùng chung)
    augmentation_manifest.csv
    augmentation_audit.md/json
    previews/
  matched/runs/yolov8n_baseline/
  matched/runs/yolov8n_augmented/
    weights/best.pt, last.pt
    args.yaml, experiment_config.yaml, requirements.lock.txt
    results.csv, results.png    # loss và metric qua epoch
    transform_audit.json, run_summary.json
    optimizer_steps.csv, epoch_budget.csv
    validation_metrics.json, per_class_ap.csv
    validation/                # confusion matrices, PR/F1/P/R curves, batch predictions
  matched/comparison/           # comparison.md/csv, per_class, biểu đồ, budget_check, gallery
  same_epochs/                 # chỉ có nếu chạy chế độ phụ
  smoke/                       # artifact kiểm thử, scientific_result=false
  diagnostics/                 # artifact của các lần gặp lỗi trong phiên triển khai
```

Audit có thể chạy lại và so hash. Augmentation đã hoàn tất sẽ được audit lại thay vì ghi đè. Nếu augmentation dở dang, pipeline báo rõ thư mục đó và yêu cầu chọn `output` mới để giữ artifact điều tra. Một run train đã tồn tại sẽ không bị ghi đè; để chạy lại, copy config, đổi `output` sang thư mục mới rồi chạy toàn bộ lệnh với `--config`. Không có resume ngầm làm thay đổi ngân sách. Có thể chạy lại evaluate/compare trên các checkpoint hiện có.

Để tiếp tục một smoke test mà cả các run train đã hoàn tất, dùng `smoke_test.py --device cpu --reuse-completed`; kiểm tra so sánh vẫn yêu cầu metadata và Validation tương thích. Khi thay seed hoặc tham số, dùng output mới. Đường dẫn tuyệt đối trong artifact là provenance của máy đã chạy; khi chuyển máy hãy tạo output mới từ config tương đối và dataset gốc.
