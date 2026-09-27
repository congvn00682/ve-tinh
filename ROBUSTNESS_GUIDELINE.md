# Phân loại cảnh khi ảnh bị suy giảm chất lượng

Mục tiêu vẫn là dự đoán 9 lớp cảnh hiện tại. Chế độ `weather_robust` bổ sung
augmentation và cách chọn checkpoint; cần train lại trên Windows để có trọng
số mới. Chưa có kết quả chứng minh chế độ này tốt hơn baseline.

## Bốn nhóm điều kiện

| Nhóm | Biến đổi được triển khai | Giới hạn |
|---|---|---|
| Thời tiết: mây/sương | Lớp phủ sáng có cấu trúc không gian, làm giảm tương phản | Không mô phỏng một cơn bão, không gán nhãn thời tiết thật |
| Ánh sáng | Tăng/giảm độ sáng và tương phản | Không phải bằng chứng về ngày/đêm hay mùa |
| Độ phân giải/độ nét | Downsample, resize lại và blur | Không đại diện cho một mức mét/pixel cụ thể |
| Nhiễu ảnh | Nhiễu Gaussian trên RGB | Không đại diện cho sensor cụ thể hoặc nhiễu SAR |

Bão có thể gây mây che, ngập và thay đổi cảnh thật. Mây che kín làm mất thông tin
mặt đất; augmentation không khôi phục được thông tin đó. Model hiện tại không
nhận diện bão và không phát hiện thiệt hại/ngập. Những bài toán này cần dữ liệu,
nhãn và thiết kế khác. Không thêm lớp `storm` vào 9 nhãn cảnh.

Nguồn tham khảo:

- [ESA: Flood Extent](https://knowledge-hub-gda.esa.int/eo_capability/flood-extent/):
  mây hạn chế ảnh quang học; radar SAR hỗ trợ quan sát ngập khi trời nhiều mây.
- [A Comprehensive Study on the Robustness of Image Classification and Object Detection in Remote Sensing](https://arxiv.org/abs/2306.12111):
  nghiên cứu đánh giá robustness trong remote sensing. Các phép biến đổi của
  project là protocol riêng, không phải bản tái lập benchmark của bài báo.

## Training và chọn checkpoint

- `baseline`: giữ nguyên augmentation cũ và chọn checkpoint bằng source validation macro-F1.
- `weather_robust`: mỗi lần lấy ảnh train, xác suất 50% giữ ảnh không có suy giảm
  bổ sung; 50% còn lại chọn đều một trong bốn nhóm, mức 1 hoặc 2.
- Crop/flip/rotation cơ bản vẫn áp dụng trong cả hai chế độ. Ảnh gốc trên đĩa
  không bị sửa; không sinh thêm bản sao dataset.
- Chọn checkpoint robust bằng trung bình macro-F1 của **5 điều kiện source
  validation**: sạch + bốn nhóm biến đổi mức 2. Mỗi điều kiện có trọng số bằng
  nhau; seed biến đổi cố định 7919. Lưu riêng F1 sạch, F1 từng điều kiện và điểm
  chọn checkpoint trong `history.json`/CSV và training curves.
- Mỗi epoch robust cần thêm bốn lượt validation nhỏ; không dùng AID để chọn epoch.
- Sau development, train lại toàn bộ source với augmentation đã chọn, rồi test.
- Mức 3 không dùng trong train/validation: đây là phép thử mức suy giảm mạnh hơn,
  không phải phép thử một loại thời tiết chưa từng thấy.

## Chạy trên Windows

Mở PowerShell tại thư mục project đã cài môi trường theo `WINDOWS_GUIDELINE.md`.
Chạy smoke test trước (2 epoch, SmallCNN):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_subset" `
  -OutputRoot "D:\experiments\robust_smoke_v1" `
  -Device cuda -Workers 4 -Amp -SmokeOnly -Augmentation weather_robust
```

Train cả bốn model với chế độ mới:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_subset" `
  -OutputRoot "D:\experiments\robust_v1" `
  -Device cuda -Workers 4 -Amp -Augmentation weather_robust
```

CPU: dùng `-Device cpu`, bỏ `-Amp`. Dùng thư mục mới cho mỗi thí nghiệm;
không trộn kết quả CPU/CUDA hoặc ghi đè kết quả cũ. Lệnh full study tự test AID
sạch sau khi hoàn tất toàn bộ training; test suy giảm chạy bằng lệnh dưới đây.

## Test robustness một checkpoint

Lệnh này dùng được cho checkpoint baseline cũ lẫn checkpoint mới; không train:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_robustness.py `
  --checkpoint "D:\experiments\robust_v1\final\resnet18_pretrained\seed_13\best.pt" `
  --manifest data\manifests\aid_test.csv `
  --data-root "D:\satellite-data\aid_subset" `
  --output-dir "D:\experiments\robustness_results\robust_resnet18_seed13" `
  --device cuda --workers 4 --corruption-seed 2026
```

Chạy lại lệnh với checkpoint baseline và thư mục output khác; giữ nguyên
manifest, seed biến đổi, thiết bị và phiên bản code. Lặp lại cho các model/seed.
Một checkpoint được đánh giá 13 lượt: ảnh sạch và 4 nhóm × 3 mức.
Với 180 ảnh, đây vẫn là **180 mẫu độc lập**, không phải 2.340 ảnh độc lập.

Kết quả:

- `robustness.csv`: accuracy, balanced accuracy, macro-F1 và mức giảm so với ảnh
  sạch, tính bằng **điểm phần trăm**. Mức giảm âm nghĩa là điểm cao hơn ảnh sạch.
- `robustness_curves.png`: đường biến thiên theo mức suy giảm.
- `summary.json`: F1 sạch, trung bình trên 12 điều kiện suy giảm, điều kiện tệ nhất.
- Mỗi điều kiện có `predictions.csv`, confusion matrix, per-class F1,
  `metrics.json`, manifest và thông tin checkpoint/hash/seed riêng.

Biến đổi đánh giá được thực hiện trên RGB gốc trước resize/crop. RNG phụ thuộc
SHA-256 ảnh, tên điều kiện và seed biến đổi; không phụ thuộc thứ tự dữ liệu,
seed training hoặc số worker. Version biến đổi được lưu để tái lập.

## Cách kết luận cải tiến

So sánh baseline và robust trên cùng manifest, cùng seed training, validation
fold và điều kiện chạy. Báo cáo cả clean F1, degraded mean F1, worst-condition
F1 và từng lớp; không chỉ chọn ô cao nhất. Mean ± SD tính qua các run training,
không coi các phiên bản biến đổi của cùng ảnh là mẫu độc lập.

Chế độ robust thay cả augmentation lẫn tiêu chí chọn checkpoint: đây là so sánh
hai pipeline. Không quy toàn bộ cải thiện riêng cho augmentation nếu chưa có
ablation giữ cố định tiêu chí chọn checkpoint.

180 ảnh AID đã được xem trong các lần trước. Nếu dùng kết quả đó để định hướng
cải tiến, báo cáo chúng là tập phát triển/thăm dò; dùng một tập AID mới chưa xem
để đánh giá cuối. Khóa protocol trước khi mở tập đó. Những phép biến đổi trên
tập cũ không tạo thành tập test độc lập mới.

Muốn khẳng định khả năng dùng trong bão thực tế, cần thêm ảnh thật có metadata
sự kiện, mức che mây và nhãn cảnh; chia theo khu vực/sự kiện để tránh rò rỉ.
Ảnh không còn đủ thông tin cần được đánh dấu để kiểm tra thủ công. Chưa triển
khai bộ nhận diện chất lượng hoặc cơ chế từ chối dự đoán tự động.

## Kiểm tra code

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Các kiểm thử xác minh tính tái lập, giữ nguyên ảnh gốc và tính đúng bảng so sánh.
Chúng không thay thế smoke test PyTorch hoặc bằng chứng cải thiện accuracy.
