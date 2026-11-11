# Hướng dẫn tải và sử dụng repo CK_DL

```powershell
git clone https://github.com/PhanGiaHuy243/CK_DL.git
cd CK_DL
```

## Xem kết quả ngay, không cần train lại

- [Báo cáo tiếng Việt](outputs/matched/comparison/tong_ket_vi.md).
- [Bảng kết quả CSV](outputs/matched/comparison/comparison.csv).
- [Biểu đồ metric](outputs/matched/comparison/metrics_comparison.png).
- [Biểu đồ AP từng lớp](outputs/matched/comparison/per_class_ap.png).
- [Gallery 14 ảnh Validation](outputs/matched/comparison/gallery/).
- [Audit dataset](outputs/audit/dataset_audit.md).
- [Audit augmentation](reports/augmentation/augmentation_audit.md).
- [Manifest augmentation](reports/augmentation/augmentation_manifest.csv).

|Thông số|Baseline|Augmented|
|---|---:|---:|
|Train images|1052|3156|
|Epoch|90|30|
|Optimizer steps|5940|5940|
|mAP@50:95|3.50%|4.45%|
|mAP@50|8.73%|11.75%|

Kết quả dùng cùng 225 ảnh Validation, ảnh 640, batch 16, seed 42. Mức cải thiện mAP@50:95 là 0,95 điểm phần trăm; mAP và Recall tuyệt đối còn thấp. Một seed chưa chứng minh độ ổn định thống kê. Xem báo cáo để biết đầy đủ giới hạn và các lớp tăng/giảm AP.

## Checkpoint có sẵn

- Baseline: `outputs/matched/runs/yolov8n_baseline/weights/best.pt`.
- Augmented: `outputs/matched/runs/yolov8n_augmented/weights/best.pt`.

Repo có code, cấu hình, kết quả, checkpoint tốt nhất và metadata audit. Dữ liệu ảnh/nhãn gốc, toàn bộ ảnh augmented, môi trường `.venv-gpu`, checkpoint `last.pt`, log tạm và các lượt chạy chẩn đoán được giữ ngoài Git. Ảnh minh họa trong gallery/previews đi kèm để xem kết quả.

## Cài thư viện

Máy chạy thí nghiệm dùng Python 3.10, RTX 3070 Laptop 8 GiB, torch 2.7.1+cu126, torchvision 0.22.1+cu126 và Ultralytics 8.3.221. Tạo môi trường riêng trên máy của bạn; không sao chép thư mục venv giữa hai máy.

Với GPU NVIDIA và driver phù hợp CUDA 12.6:

```powershell
python -m venv .venv-gpu
.\.venv-gpu\Scripts\python.exe -m pip install --no-cache-dir -r requirements-gpu.txt
.\.venv-gpu\Scripts\python.exe -m pip install --no-cache-dir -r requirements.txt --index-url https://pypi.org/simple
```

Nếu chỉ chạy CPU:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Sau khi cài, có thể dự đoán ảnh của bạn bằng checkpoint có sẵn:

```powershell
# CPU; thay source bằng ảnh hoặc thư mục ảnh cần dự đoán
.\.venv\Scripts\yolo.exe predict model=outputs/matched/runs/yolov8n_augmented/weights/best.pt source=duong_dan_anh.jpg device=cpu imgsz=640 conf=0.25
```

Nếu dùng môi trường GPU, đổi executable sang `.\.venv-gpu\Scripts\yolo.exe` và `device=0`. Ngưỡng 0.25 ở đây phục vụ xem ảnh dự đoán; đánh giá AP trong nghiên cứu dùng conf 0.001.

## Lấy dữ liệu và tái lập thí nghiệm

Nhận bộ `CHENAB` gốc từ chủ repo/nhóm, giữ nguyên tên file và các split, rồi đặt tại:

```text
CK_DL/
  CHENAB/
    data.yaml
    train/images/  train/labels/
    valid/images/  valid/labels/
    test/images/   test/labels/
```

Không có link dataset công khai được khai báo trong repo. Có đúng 1052/225/225 cặp Train/Validation/Test, 7 lớp theo YAML. Script tự xử lý đường dẫn cũ trong YAML khi đường dẫn đó không tồn tại. Không chia lại dữ liệu. Không cần tải sẵn 3156 ảnh augmented: script tạo lại từ Train gốc với seed đã ghi.

Các kết quả `outputs/matched/` là artifact đã hoàn tất. Để train lại, tạo cấu hình với **output mới** để giữ nguyên kết quả gốc:

```powershell
Copy-Item configs/full_gpu.yaml configs/reproduce_gpu.yaml
# Mở configs/reproduce_gpu.yaml và đổi duy nhất:
# output: outputs/reproduce

.\.venv-gpu\Scripts\python.exe check_environment.py --config configs/reproduce_gpu.yaml --download-weights
.\.venv-gpu\Scripts\python.exe audit_data.py --config configs/reproduce_gpu.yaml
.\.venv-gpu\Scripts\python.exe augment_dataset.py --config configs/reproduce_gpu.yaml
.\.venv-gpu\Scripts\python.exe run_full_experiment.py --config configs/reproduce_gpu.yaml --device 0
```

Từ đó coordinator tự train Baseline, train Augmented, đánh giá chung và xuất so sánh vào `outputs/reproduce/matched/`. `full_gpu.yaml` yêu cầu CUDA; nếu chạy CPU, đặt device CPU và batch phù hợp trên cấu hình mới cho cả hai mô hình.

Các đường dẫn tuyệt đối/PID trong JSON/YAML kết quả là provenance từ máy đã chạy, không phải đường dẫn cần dùng trên máy mới. Muốn dùng lại pipeline đánh giá cho checkpoint hiện có trên máy khác cần resolve lại đường dẫn dữ liệu trong bản sao metadata; cách chạy lại toàn bộ phía trên tự tạo đường dẫn mới.

Chi tiết phương pháp và cách tổ chức code: [README.md](README.md).
