# Lưu bằng chứng thực nghiệm và xuất biểu đồ

Pipeline vẫn chỉ báo cáo cross-domain trên AID. Source validation dùng chọn epoch.
Tính năng này không thay đổi architecture hay hyperparameter nhằm tăng accuracy.

## 1. Với các kết quả bạn đã chạy trên Windows

Nếu còn toàn bộ thư mục `outputs`, không cần train lại chỉ để lấy biểu đồ.
Cập nhật code và dùng Python trong môi trường đã cài trên Windows:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m satdomain.reports --output-root "D:\experiments\outputs_cuda"
```

Lệnh đọc mọi `history.json` và `metrics.json` bên trong đường dẫn được chọn,
xuất lại CSV/PNG cạnh file gốc. Chạy tương tự cho `outputs_cpu` nếu cần.
Lệnh không tải checkpoint, không train/test lại và không yêu cầu GPU.
Nó thay thế các file biểu đồ/bảng xuất cùng tên, giữ nguyên JSON nguồn và weights.

| File còn giữ | Có thể xuất lại |
|---|---|
| `history.json` | Training curves và bảng lịch sử theo epoch |
| `aid_evaluation/metrics.json` | Confusion matrix, precision/recall/F1 và số ảnh từng lớp |
| `best.pt` + đúng ảnh/manifest AID | Có thể chạy evaluation lại để tạo metrics và biểu đồ |
| Chỉ `cross_domain_models.csv` | Không khôi phục được confusion matrix, F1 từng lớp hay training curves |

Seed, architecture, epoch và training arguments đã nằm trong checkpoint của code cũ.
Code mới không tự nhận một manifest hiện tại là manifest đã dùng trong quá khứ.
Nếu thiếu manifest cũ, phải ghi rõ thiếu thông tin này. Không thể khôi phục
training curves chỉ từ weights hoặc từ accuracy tổng hợp.

Nếu còn checkpoint cũ nhưng thiếu metrics, đánh giá lại vào thư mục MỚI:

```powershell
.\.venv\Scripts\python.exe -m satdomain.evaluate `
  --checkpoint "D:\experiments\outputs_cuda\final\deit_tiny_pretrained\seed_13\best.pt" `
  --manifest "D:\experiments\manifests_original\aid_test.csv" `
  --data-root "D:\satellite-data\aid_target" `
  --output-dir "D:\experiments\reevaluation_deit_seed13" `
  --device cpu --workers 0
```

Đây là một evaluation mới. `run.json` ghi thiết bị test mới, hash checkpoint,
seed lấy từ checkpoint và manifest test thực tế. Với checkpoint cũ không có
training run ID, giá trị này để trống; không suy đoán hoặc tạo provenance cũ.

## 2. Những lần train mới: tự động lưu đầy đủ

Dùng output directory mới để không trộn kết quả trước và sau thay đổi code:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -OutputRoot "D:\experiments\study_cuda_v2" `
  -Device cuda -Amp -Workers 4
```

Nếu dùng CPU, thay `-Device cuda -Amp` bằng `-Device cpu`, và chọn output khác.
Không bắt buộc phải train lại các kết quả cũ nếu bạn chỉ cần xuất lại biểu đồ.

Mỗi study lưu:

```text
study_cuda_v2/
  study.json                         # seeds, settings, hashes của code và manifest
  manifests/
    source.csv
    aid_test.csv
    generation_summary.json          # nếu có; gồm split seed của manifest mới
  development/<model>/seed_13/
    best.pt                          # chọn bằng source-validation macro-F1
    run.json                         # run ID, seed, args, môi trường, AMP, preprocessing
    source_manifest.csv              # bản sao manifest đầu vào
    train_manifest.csv               # chính xác những ảnh dùng để train
    validation_manifest.csv          # chính xác những ảnh dùng validation
    summary.json                     # epoch, hash checkpoint và source manifest
    history.json
    history.csv
    training_curves.png
    complete.json                    # chỉ ghi khi run hoàn tất, kèm hash artifacts
  final/<model>/seed_13/
    best.pt                          # checkpoint epoch cuối sau fit toàn bộ source
    run.json
    source_manifest.csv
    train_manifest.csv
    development_summary.json         # nguồn quyết định số epoch cho final fit
    summary.json
    history.json
    history.csv
    training_curves.png
    complete.json
    aid_evaluation/
      run.json                       # device test, checkpoint SHA-256, training seed/run ID
      target_manifest.csv
      metrics.json
      predictions.csv
      per_class_metrics.csv
      per_class_f1.png
      confusion_matrix_counts.csv
      confusion_matrix_counts.png
      confusion_matrix_normalized.csv
      confusion_matrix_normalized.png
      confusion_matrix.png           # tên tương thích cũ, dạng normalized
      complete.json
```

Seed 37 và 73 có cấu trúc tương tự. Tên `best.pt` của final model được giữ để
tương thích lệnh inference; nó là checkpoint epoch cuối, không chọn theo AID.
Các file JSON mô tả môi trường giúp phân biệt CPU/CUDA, AMP, phiên bản thư viện.
Lưu seed không đảm bảo kết quả CPU và CUDA giống từng bit.

Manifest lưu đường dẫn tương đối và SHA-256 của từng ảnh. Trước train/test,
code kiểm tra nội dung ảnh thực tế khớp manifest, class mapping và domain.
Hash phát hiện thay đổi byte/trùng tuyệt đối, không phát hiện mọi ảnh gần trùng.
Ảnh gốc không được copy vào outputs; cần giữ dataset gốc riêng để tái lập.

## 3. Cách đọc và sử dụng báo cáo

- `per_class_metrics.csv`: class index, tên lớp, precision, recall, F1, support.
  Giá trị metric từ 0 đến 1; support là số ảnh test thật của lớp đó.
- `per_class_f1.png`: biểu đồ F1 từng lớp có ghi số ảnh.
- Confusion matrix: hàng là nhãn thật, cột là nhãn dự đoán. Bản counts là số ảnh;
  bản normalized là tỷ lệ trong mỗi hàng.
- Development curves: loss, accuracy và macro-F1 của train/validation.
- Final curves: chỉ có train vì final fit dùng toàn bộ source. Không tạo đường
  validation giả và không dùng accuracy test AID làm đường validation.

Để tổng hợp bảng từng lớp giữa model/seed:

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_results.py `
  --output-root "D:\experiments\study_cuda_v2" `
  --models small_cnn resnet18_scratch resnet18_pretrained deit_tiny_pretrained
```

Ngoài `cross_domain_runs.csv` và `cross_domain_models.csv`, có thêm:

```text
per_class_runs.csv       # từng model, seed, lớp: precision/recall/F1/support
per_class_models.csv     # mean và sample std qua seeds, theo model và lớp
```

Bảng cross-domain từng run có hash checkpoint và target manifest khi dữ liệu
evaluation mới cung cấp. Với outputs cũ chưa có hash, trường này để trống.
Không cộng support qua seeds rồi coi là số ảnh độc lập: vẫn cùng một tập AID.

## 4. Bảo toàn kết quả và tiếp tục chạy

Train/evaluate từ chối ghi đè thư mục output không rỗng. Dùng thư mục mới khi
đổi cấu hình, code hoặc dataset. `run_study.py --skip-existing` chỉ bỏ qua run
có `complete.json` và các file còn khớp hash; có `best.pt` chưa có nghĩa đã hoàn tất.

Nếu run bị ngắt giữa chừng, đổi tên thư mục run chưa hoàn tất để giữ lại lịch sử
(ví dụ chuyển ra một thư mục backup ngoài study), rồi chạy lại study với
`-SkipExisting`. Các run đã hoàn tất được giữ; run còn thiếu bắt đầu lại từ đầu.
Đây không phải resume optimizer từ epoch cuối.

Study cũ không có `study.json` không được tiếp tục bằng cơ chế mới; hãy xuất báo
cáo cũ hoặc dùng output root mới. Không thêm `complete.json` thủ công để bỏ qua kiểm tra.

Để nộp giáo viên, lưu cả folder study (bao gồm development và final), code/
Git commit, cùng thông tin nguồn dữ liệu. Checkpoint/manifests/outputs thường
được Git bỏ qua; phải sao lưu riêng. Không chỉ lưu bảng accuracy tổng hợp.

## 5. Kiểm tra trên Windows trước khi chạy full

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -OutputRoot "D:\experiments\artifact_smoke_v2" `
  -Device cpu -Workers 0 -SmokeOnly
```

Tests kiểm tra integrity của manifest/checkpoint, việc không ghi đè run, tên lớp,
số liệu CSV và phân biệt development/final curves. Phần plotting trong unit
tests được mock; smoke test mới kiểm tra PyTorch/Matplotlib thực tế trên máy bạn.
