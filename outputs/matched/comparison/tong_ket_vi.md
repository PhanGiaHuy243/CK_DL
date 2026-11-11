# Thống kê kết quả thực nghiệm GPU

**Trạng thái: hoàn tất toàn bộ pipeline, cả 5 bước thành công.**

- Bắt đầu: 02:13:33 ngày 11/11/2026, giờ Việt Nam.
- Kết thúc: 05:06:23 ngày 11/11/2026, giờ Việt Nam.
- Tổng thời gian pipeline: **2 giờ 52 phút 50 giây**.
- Thiết bị: RTX 3070 Laptop 8 GiB; ảnh 640, batch 16, seed 42.
- Hai best.pt được đánh giá lại trên cùng 225 ảnh Validation. Test không được sử dụng.

|Thông số|Baseline|Augmented|
|---|---:|---:|
|Ảnh Train|1052|3156|
|Epoch|90|30|
|Optimizer step thực tế|5940|5940|
|Lượt xem ảnh thực tế|94680|94680|
|Thời gian train|1 giờ 27 phút 52 giây|1 giờ 18 phút 42 giây|

Thời gian train gồm khởi tạo trainer, huấn luyện, Validation trong training, lưu checkpoint và đánh giá best của framework; không gồm audit augmentation và bước đánh giá chung sau cùng.

## Metric Validation

|Metric|Baseline|Augmented|Chênh lệch, điểm phần trăm|
|---|---:|---:|---:|
|mAP@50:95|3.50%|4.45%|+0.95|
|mAP@50|8.73%|11.75%|+3.02|
|Precision|18.20%|61.23%|+43.02|
|Recall|10.00%|10.48%|+0.48|

Precision/Recall dùng điểm mean-F1 tối ưu theo evaluator Ultralytics; không phải accuracy. AP matching IoU là 0.50:0.05:0.95; confidence đầu vào 0.001, NMS IoU 0.7, max_det 300, FP32, không test-time augmentation.

## AP từng lớp tại IoU 0.50:0.95

|ID|Lớp|Baseline|Augmented|Chênh lệch, điểm phần trăm|
|---|---|---:|---:|---:|
|0|baekra|0.00%|0.43%|+0.43|
|1|color issues|0.08%|1.08%|+1.00|
|2|contamination|3.38%|0.70%|-2.68|
|3|cut|4.80%|1.83%|-2.97|
|4|gray stitch|1.36%|5.16%|+3.81|
|5|selvet|0.23%|2.54%|+2.31|
|6|stain|14.63%|19.39%|+4.76|

## Nhận xét

- mAP@50:95 tăng **0.95 điểm phần trăm**, tương đương tăng tương đối **27.3%** so với Baseline trong lượt chạy này.
- 5/7 lớp tăng AP: baekra, color issues, gray stitch, selvet, stain.
- 2/7 lớp giảm AP: contamination và cut.
- Kết quả tuyệt đối còn thấp: Augmented đạt mAP@50:95 4.45% và Recall khoảng 10.48%. Precision cao hơn không có nghĩa đã phát hiện được phần lớn lỗi.
- Ngân sách được xác nhận bằng số đo thực tế: cùng 5940 optimizer step, 94680 lượt xem ảnh và chuỗi learning rate giống nhau.
- Batch cuối khác nhau (12/4 ảnh); số cơ hội chọn checkpoint theo Validation khác nhau (90/30). Đây là các giới hạn còn lại của thiết kế.
- Chỉ có một seed 42; chưa đủ để khẳng định độ ổn định thống kê. Không suy rộng kết quả Validation thành kết quả Test.

## Artifact

- comparison.md, comparison.csv, per_class_comparison.csv: bảng số liệu đầy đủ theo thang 0–1.
- metrics_comparison.png, per_class_ap.png: biểu đồ.
- gallery/: 14 ảnh Validation, gồm ground truth và hai dự đoán cùng ảnh.
- Hai best.pt tại ../runs/yolov8n_baseline/weights/ và ../runs/yolov8n_augmented/weights/.
- Mỗi run có last.pt, results.csv/png, confusion matrix, PR/P/R/F1 curves và AP từng lớp.
- Thông báo Windows đã được phát sau khi pipeline hoàn tất; watcher ghi nhận đã được xác nhận.
