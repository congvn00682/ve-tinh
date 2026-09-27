# Satellite Image Classification across Different Domains

Project nghiên cứu khả năng tổng quát hóa của các model scene-classification khi
train trên source domain hiện có và test trên AID. Code được chuẩn bị trên máy
phát triển nhưng việc cài PyTorch và train được thực hiện ở máy khác.

Hướng dẫn chi tiết để chuyển và chạy trên máy khác nằm tại
[RUN_ON_ANOTHER_MACHINE_GUIDELINE.md](RUN_ON_ANOTHER_MACHINE_GUIDELINE.md).

Riêng Windows có bộ cài PowerShell và hướng dẫn tại
[WINDOWS_GUIDELINE.md](WINDOWS_GUIDELINE.md).

Methodology đã khóa nằm tại [research/methodology.md](research/methodology.md).

## Thiết kế dữ liệu

Thí nghiệm chính có 9 lớp chung giữa source và AID:

```text
airport, baseball_diamond, beach, bridge, church,
commercial_area, dense_residential, desert, forest
```

Repository chỉ giữ 9 lớp này; mọi model dùng cùng một label space.

- Source hiện tại có 100 ảnh/lớp; code không bắt buộc số lượng cố định.
- Target AID có thể có số ảnh khác nhau, chỉ dùng cho lần đánh giá cuối.
- Không dùng AID để early stopping hoặc chọn hyperparameter.
- Báo cáo chỉ chứa kết quả cross-domain trên AID, không có same-domain test.

## 1. Chuẩn bị trên máy train

Khuyến nghị Python 3.10–3.13. Tạo virtual environment và cài bản PyTorch phù
hợp với CUDA/CPU của máy train trước, sau đó cài project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Cài torch/torchvision theo GPU của máy train, rồi:
python -m pip install -e .
```

Không cần Docker.

## 2. Cấu trúc folder AID

Script chấp nhận cả tên folder AID gốc và các biến thể viết thường. Ví dụ:

```text
/path/to/aid_subset/
  Airport/
  BaseballField/
  Beach/
  Bridge/
  Church/
  Commercial/
  DenseResidential/
  Desert/
  Forest/
```

Số ảnh trong mỗi folder không bắt buộc phải bằng nhau. Manifest sẽ ghi lại số
lượng thực tế và dừng nếu một lớp bị rỗng, có ảnh trùng trong một domain, hoặc
source và target có ảnh trùng nội dung. Có thể dùng
`--expected-target-per-class N` khi chủ động muốn kiểm tra đúng `N` ảnh/lớp.

## 3. Tạo manifest và audit

Chạy từ thư mục repository:

```bash
python scripts/prepare_manifests.py \
  --source-root . \
  --aid-root /path/to/aid_subset \
  --output-dir data/manifests
```

Kết quả:

```text
data/manifests/source.csv
data/manifests/aid_test.csv
data/manifests/summary.json
```

Mỗi dòng chứa relative path, class, class index, fold, domain và SHA-256.

## 4. Smoke test một model

```bash
python -m satdomain.train \
  --manifest data/manifests/source.csv \
  --data-root . \
  --arch small_cnn \
  --mode development \
  --validation-fold 0 \
  --epochs 2 \
  --workers 4 \
  --output-dir outputs/smoke
```

Trên NVIDIA GPU có thể thêm `--amp`. Không dùng `--amp` cho CPU/MPS.

## 5. Chạy toàn bộ nghiên cứu

Nên chạy phiên dài trong tmux. Script dùng source train/validation để chọn số
epoch, sau đó train lại bằng toàn bộ source. Chỉ khi mọi model đã train xong,
script mới mở AID để thực hiện cross-domain test:

```bash
python scripts/run_study.py \
  --source-manifest data/manifests/source.csv \
  --source-root . \
  --target-manifest data/manifests/aid_test.csv \
  --target-root /path/to/aid_subset \
  --output-root outputs \
  --workers 4 \
  --amp
```

Các model mặc định:

```text
small_cnn
resnet18_scratch
resnet18_pretrained
deit_tiny_pretrained
```

`--skip-existing` cho phép tiếp tục một study bị ngắt mà không train lại các run
đã có kết quả.

## 6. Tổng hợp kết quả

```bash
python scripts/aggregate_results.py \
  --output-root outputs \
  --models small_cnn resnet18_scratch resnet18_pretrained deit_tiny_pretrained
```

Hai bảng cross-domain được sinh ra:

```text
outputs/cross_domain_runs.csv
outputs/cross_domain_models.csv
```

Mỗi final run còn có AID `metrics.json`, `predictions.csv`, confusion matrix PNG
và training curves. Source validation chỉ dùng điều khiển training, không được
đưa vào bảng kết quả nghiên cứu.

## 7. Inference một ảnh mới

```bash
python -m satdomain.infer \
  --checkpoint outputs/final/resnet18_pretrained/seed_13/best.pt \
  --image /path/to/satellite_image.jpg \
  --top-k 3
```

Kết quả là JSON gồm nhãn và confidence. Đây là closed-set classifier: ảnh ngoài
9 lớp vẫn bị ép vào một lớp đã biết. Confidence cũng chưa được đảm bảo calibrated
trên domain mới.

## Lưu ý tái lập

- Giữ nguyên manifest sau khi bắt đầu thí nghiệm.
- Không điều chỉnh model dựa trên kết quả AID.
- Không báo cáo kết quả của smoke test như kết quả nghiên cứu.
- Commit code/config trước khi chạy final study và ghi lại commit hash.
