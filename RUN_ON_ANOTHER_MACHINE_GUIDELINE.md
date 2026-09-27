# Guideline chạy project trên máy khác

Tài liệu này hướng dẫn chuyển source code, chuẩn bị môi trường, train các model,
thực hiện cross-domain test trên AID, tổng hợp kết quả và inference ảnh mới.

## 1. Pipeline sẽ chạy

```text
Source dataset
  -> source train/validation để chọn số epoch
  -> train lại model bằng toàn bộ source
  -> lưu checkpoint
  -> test duy nhất trên AID target
  -> accuracy, balanced accuracy, precision, recall, macro-F1, ECE
  -> confusion matrix và bảng so sánh model
```

Các model mặc định:

```text
small_cnn
resnet18_scratch
resnet18_pretrained
deit_tiny_pretrained
```

Project không dùng Docker và không yêu cầu cài đặt database hay dịch vụ nền.

## 2. Chuyển project sang máy train

### Cách A: Git

Các thay đổi hiện tại phải được commit và push trước khi clone trên máy khác.
Trên máy đang chứa project:

```bash
git status
git add .
git commit -m "Add cross-domain satellite classification pipeline"
git push
```

Lưu ý: commit này cũng ghi nhận việc xóa hai lớp không còn sử dụng. Kiểm tra kỹ
`git status` trước khi commit.

Trên máy train:

```bash
git clone <URL_REPOSITORY>
cd ve-tinh
```

### Cách B: copy trực tiếp qua SSH

Chạy từ thư mục cha của project trên máy nguồn:

```bash
rsync -a \
  --exclude .venv \
  --exclude outputs \
  --exclude __pycache__ \
  ve-tinh/ user@training-machine:/duong-dan/ve-tinh/
```

Nếu dùng USB hoặc công cụ đồng bộ file, cần copy toàn bộ folder `ve-tinh`, bao
gồm source images, `src`, `scripts`, `configs`, `research`, `pyproject.toml` và
tài liệu này.

Sau khi chuyển, kiểm tra:

```bash
cd /duong-dan/ve-tinh
ls
```

Phải nhìn thấy tối thiểu:

```text
airport/
baseball_diamond/
beach/
bridge/
church/
commercial_area/
dense_residential/
desert/
forest/
configs/
research/
scripts/
src/
pyproject.toml
README.md
```

## 3. Yêu cầu máy train

Khuyến nghị:

- Python 3.11 hoặc 3.12 được khuyến nghị.
- NVIDIA GPU nếu muốn train nhanh.
- Driver NVIDIA hoạt động nếu dùng CUDA.
- Có Internet trong lần đầu chạy pretrained ResNet18 và DeiT để tải weights.
- Dung lượng trống đủ cho virtual environment, pretrained weights và outputs.

CPU vẫn chạy được nhưng chậm. Apple Silicon có thể dùng backend MPS. PyTorch
cung cấp bộ chọn lệnh cài đặt chính thức tại:

<https://docs.pytorch.org/get-started/locally/>

Metadata của project chấp nhận Python 3.10–3.13, nhưng khả năng có sẵn PyTorch
wheel còn phụ thuộc hệ điều hành và phiên bản PyTorch. Python 3.11/3.12 là lựa
chọn an toàn hơn. Không dùng Python 3.14 vì `pyproject.toml` hiện giới hạn
Python `<3.14`.

## 4. Tạo Python virtual environment

### Linux hoặc macOS

```bash
cd /duong-dan/ve-tinh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

Nếu máy chỉ có lệnh `python3`, thay `python3.12` bằng `python3` và kiểm tra:

```bash
python --version
```

### Windows PowerShell

```powershell
cd C:\duong-dan\ve-tinh
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
```

Nếu PowerShell chặn activation script, có thể dùng Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Mỗi lần mở terminal mới, phải activate lại `.venv` trước khi chạy project.

## 5. Cài PyTorch đúng loại máy

### Máy NVIDIA CUDA

Kiểm tra GPU và driver:

```bash
nvidia-smi
```

Mở trang cài đặt PyTorch chính thức, chọn:

```text
PyTorch Build: Stable
OS: hệ điều hành của máy
Package: Pip
Language: Python
Compute Platform: CUDA phù hợp
```

Sau đó chạy đúng lệnh mà trang chính thức sinh ra. Không nên chép một lệnh CUDA
cố định từ tài liệu này vì phiên bản wheel được PyTorch cập nhật theo thời gian.

### Máy chỉ dùng CPU

Chọn `Compute Platform: CPU` trên trang cài đặt chính thức rồi chạy lệnh được
cung cấp.

### Apple Silicon

Thông thường có thể cài:

```bash
python -m pip install torch torchvision
```

Project tự nhận diện MPS bằng `torch.backends.mps.is_available()`. Không dùng
`--amp` khi train bằng MPS.

## 6. Cài các dependency còn lại

Sau khi đã cài đúng PyTorch/torchvision:

```bash
python -m pip install -e .
```

Lệnh này cài package `satdomain` cùng các dependency như `timm`, `pandas`,
`scikit-learn`, `Pillow` và `matplotlib`.

Kiểm tra installation:

```bash
python -c "import torch, torchvision, timm, satdomain; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('mps=', torch.backends.mps.is_available())"
```

Kết quả mong đợi:

- NVIDIA: `cuda=True`.
- Apple Silicon: `mps=True`.
- CPU: cả CUDA và MPS có thể là `False`; project vẫn chạy bằng CPU.

## 7. Chuẩn bị AID target dataset

AID có thể nằm ngoài repository. Cấu trúc cần có đủ chín lớp:

```text
/data/aid_target/
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

Số ảnh target không bắt buộc là 100 ảnh/lớp và các lớp không bắt buộc có số ảnh
bằng nhau. Tuy nhiên:

- Mỗi lớp phải có ít nhất một ảnh.
- Càng ít ảnh thì accuracy và per-class metrics càng thiếu ổn định.
- Nếu số ảnh giữa các lớp khác nhau, ưu tiên macro-F1 và balanced accuracy.
- 5 ảnh/lớp chỉ phù hợp cho kiểm tra nhanh, không phù hợp làm kết quả nghiên cứu
  cuối cùng.

Các extension được hỗ trợ:

```text
.jpg .jpeg .png .tif .tiff
```

## 8. Tạo manifest và audit dữ liệu

Từ thư mục repository:

```bash
python scripts/prepare_manifests.py \
  --source-root . \
  --aid-root /data/aid_target \
  --output-dir data/manifests
```

Windows PowerShell có thể chạy một dòng:

```powershell
python scripts/prepare_manifests.py --source-root . --aid-root "D:\data\aid_target" --output-dir data/manifests
```

Kết quả:

```text
data/manifests/source.csv
data/manifests/aid_test.csv
data/manifests/summary.json
```

Script kiểm tra:

- Đủ chín class folder.
- Mỗi class không bị rỗng.
- Source có đủ ảnh để tạo năm validation folds.
- Không có ảnh trùng tuyệt đối trong cùng domain.
- Không có ảnh trùng tuyệt đối giữa source và target.
- Ghi SHA-256 cho từng ảnh.

Mở `data/manifests/summary.json` và kiểm tra số ảnh trước khi train.

Nếu chủ động yêu cầu target phải có đúng 5 ảnh/lớp, có thể thêm:

```bash
--expected-target-per-class 5
```

Không thêm tham số này nếu target có số lượng thay đổi.

## 9. Chạy smoke test trước

Không chạy toàn bộ study ngay. Đầu tiên kiểm tra SmallCNN với hai epoch:

```bash
python -m satdomain.train \
  --manifest data/manifests/source.csv \
  --data-root . \
  --arch small_cnn \
  --mode development \
  --validation-fold 0 \
  --epochs 2 \
  --batch-size 16 \
  --workers 4 \
  --output-dir outputs/smoke
```

Trên NVIDIA có thể thêm:

```bash
--device cuda --amp
```

Trên Apple Silicon có thể thêm:

```bash
--device mps
```

Trên CPU:

```bash
--device cpu
```

Smoke test thành công khi xuất hiện:

```text
outputs/smoke/best.pt
outputs/smoke/history.json
outputs/smoke/summary.json
outputs/smoke/training_curves.png
```

Smoke test không phải kết quả nghiên cứu.

## 10. Chạy toàn bộ cross-domain study

### NVIDIA GPU

```bash
python scripts/run_study.py \
  --source-manifest data/manifests/source.csv \
  --source-root . \
  --target-manifest data/manifests/aid_test.csv \
  --target-root /data/aid_target \
  --output-root outputs \
  --batch-size 32 \
  --workers 4 \
  --device cuda \
  --amp
```

### CPU hoặc Apple Silicon

Bỏ `--amp` và chọn `--device cpu`, `--device mps`, hoặc để `--device auto`:

```bash
python scripts/run_study.py \
  --source-manifest data/manifests/source.csv \
  --source-root . \
  --target-manifest data/manifests/aid_test.csv \
  --target-root /data/aid_target \
  --output-root outputs \
  --batch-size 32 \
  --workers 4 \
  --device auto
```

Windows PowerShell có thể đặt toàn bộ tham số trên một dòng.

Quy trình chạy theo đúng thứ tự:

1. Development trên source train/validation.
2. Chọn số epoch chỉ từ source validation.
3. Train lại bằng toàn bộ source.
4. Sau khi tất cả model hoàn tất, mới test trên AID.

Mặc định có bốn model và ba seed. Lần đầu chạy pretrained model cần tải weights.

Nếu phiên trước bị ngắt, chạy lại cùng lệnh và thêm:

```bash
--skip-existing
```

Trên Linux/macOS, nên dùng một phiên terminal bền vững như tmux:

```bash
tmux new -s satellite-cross-domain
```

Sau đó chạy study trong tmux. Dùng `Ctrl+B`, rồi `D` để detach và:

```bash
tmux attach -t satellite-cross-domain
```

để quay lại.

## 11. Vị trí checkpoint và kết quả test

Checkpoint cuối:

```text
outputs/final/<model>/seed_<seed>/best.pt
```

Ví dụ:

```text
outputs/final/resnet18_pretrained/seed_13/best.pt
```

Kết quả AID của từng model/seed:

```text
outputs/final/<model>/seed_<seed>/aid_evaluation/
  metrics.json
  predictions.csv
  confusion_matrix.png
```

Terminal cũng in:

```text
accuracy=... balanced_accuracy=... macro_f1=...
```

Giá trị `0.80` tương ứng `80%`.

## 12. Tổng hợp và so sánh model

Sau khi tất cả model test xong:

```bash
python scripts/aggregate_results.py \
  --output-root outputs \
  --models small_cnn resnet18_scratch resnet18_pretrained deit_tiny_pretrained
```

Kết quả:

```text
outputs/cross_domain_runs.csv
outputs/cross_domain_models.csv
```

Ý nghĩa:

- `cross_domain_runs.csv`: kết quả từng model và từng seed.
- `cross_domain_models.csv`: mean và standard deviation qua các seed.
- Macro-F1 là chỉ số chính khi số ảnh mỗi lớp target khác nhau.
- Balanced accuracy cho mỗi lớp trọng số ngang nhau.
- Accuracy cho biết tỷ lệ đúng trên toàn bộ ảnh.
- ECE càng thấp thì confidence càng gần với độ chính xác thực tế hơn.

So sánh có kiểm soát ảnh hưởng của pretraining bằng:

```text
resnet18_scratch vs resnet18_pretrained
```

## 13. Test thủ công một checkpoint

```bash
python -m satdomain.evaluate \
  --checkpoint outputs/final/resnet18_pretrained/seed_13/best.pt \
  --manifest data/manifests/aid_test.csv \
  --data-root /data/aid_target \
  --domain target \
  --output-dir outputs/manual_test \
  --workers 4 \
  --device auto
```

Kết quả nằm trong `outputs/manual_test`.

## 14. Inference một ảnh mới

```bash
python -m satdomain.infer \
  --checkpoint outputs/final/resnet18_pretrained/seed_13/best.pt \
  --image /data/new_image.jpg \
  --top-k 3 \
  --device auto
```

Output là JSON gồm ba class có confidence cao nhất. Model là closed-set
classifier: ảnh ngoài chín lớp vẫn bị ép vào một trong chín class đã biết.

## 15. Lỗi thường gặp

### `ModuleNotFoundError: No module named 'satdomain'`

Activate đúng virtual environment và cài lại project:

```bash
source .venv/bin/activate
python -m pip install -e .
```

Windows:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

### `ModuleNotFoundError: No module named 'torch'`

Cài PyTorch bằng lệnh từ bộ chọn chính thức, trong đúng virtual environment.

### `torch.cuda.is_available()` trả về `False`

Kiểm tra:

```bash
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Nếu driver hoạt động nhưng CUDA vẫn `False`, thường là đã cài CPU-only PyTorch.
Cài lại wheel CUDA theo bộ chọn chính thức.

### CUDA out of memory

Giảm batch size:

```bash
--batch-size 16
```

Nếu vẫn lỗi, thử `8`. Giữ cùng batch size giữa các model khi có thể và ghi lại
mọi thay đổi trong báo cáo.

### DataLoader lỗi trên Windows

Thử:

```bash
--workers 0
```

Sau khi xác nhận chạy được, có thể tăng lại nhưng không vượt quá tài nguyên máy.

### Không tìm thấy class folder

Kiểm tra tên folder AID theo phần 7. Không đặt ảnh trực tiếp ở root AID.

### Pretrained model không tải được weights

ResNet18 pretrained và DeiT-Tiny pretrained cần Internet trong lần tải đầu.
Kiểm tra kết nối hoặc chuẩn bị cache weights trước khi chạy trên máy offline.

### Study bị dừng giữa chừng

Chạy lại đúng command cũ với:

```bash
--skip-existing
```

Không xóa các checkpoint đã hoàn tất.

## 16. Quy tắc để kết quả nghiên cứu hợp lệ

1. Không dùng AID để train hoặc early stopping.
2. Dùng cùng một `aid_test.csv` cho tất cả model.
3. Không đổi target images giữa các model.
4. Không chọn augmentation/hyperparameter dựa trên AID test rồi báo cáo lại AID
   như một test set hoàn toàn độc lập.
5. Lưu `summary.json`, manifest, config, checkpoint và Git commit tương ứng.
6. Không so sánh hai model nếu chúng được đánh giá trên hai target set khác nhau.
7. Nếu thay class hoặc target dataset, tạo experiment/output folder mới và chạy
   lại tất cả model.

## 17. Checklist trước khi chạy full study

- [ ] Project đã được copy đầy đủ hoặc clone từ commit mới nhất.
- [ ] Đang dùng Python 3.11 hoặc 3.12 nếu có thể.
- [ ] Virtual environment đang active.
- [ ] PyTorch nhận đúng CUDA/MPS/CPU.
- [ ] `python -m pip install -e .` đã thành công.
- [ ] AID có đủ chín class folder.
- [ ] `summary.json` có số ảnh đúng dự kiến.
- [ ] Smoke test hai epoch đã tạo `best.pt`.
- [ ] Đã chọn `batch-size` không gây out-of-memory.
- [ ] Mọi model dùng cùng target manifest.
- [ ] Có thư mục riêng để giữ outputs và kết quả báo cáo.
