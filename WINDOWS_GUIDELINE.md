# Hướng dẫn cài và chạy trên Windows

Đây là quy trình ngắn dành riêng cho Windows 10/11. Hai PowerShell scripts đi
kèm tự tạo virtual environment và gọi đúng Python bên trong `.venv`; người dùng
không cần activate environment thủ công.

## 1. Chuẩn bị

Cài:

1. Python 3.11 hoặc 3.12 từ <https://www.python.org/downloads/windows/>.
2. Git for Windows nếu chuyển project qua Git.
3. NVIDIA driver nếu máy có NVIDIA GPU.

Copy hoặc clone toàn bộ repository rồi mở PowerShell tại thư mục `ve-tinh`.

## 2. Cho phép chạy script chỉ trong tiến trình hiện tại

Không cần thay đổi execution policy toàn hệ thống. Mọi lệnh dưới đây dùng:

```powershell
powershell -ExecutionPolicy Bypass -File <script>
```

## 3. Cài environment

### Cách đơn giản

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

Script sẽ:

1. Tạo `.venv` bằng Python 3.12.
2. Cập nhật pip.
3. Cài `torch` và `torchvision`.
4. Cài project cùng `timm`, pandas, scikit-learn, Pillow và matplotlib.
5. Kiểm tra CUDA/MPS và import package.

### NVIDIA CUDA

Mở bộ chọn chính thức:

<https://docs.pytorch.org/get-started/locally/>

Chọn Windows, Pip, Python và CUDA phù hợp. Lấy URL đứng sau `--index-url` trong
lệnh do PyTorch cung cấp, sau đó chạy ví dụ:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 `
  -TorchIndexUrl "URL_LAY_TU_TRANG_PYTORCH" `
  -RequireCuda
```

Ví dụ trên cố ý không hard-code phiên bản CUDA vì PyTorch thay đổi các wheel
được hỗ trợ theo thời gian.

### CPU-only

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 `
  -TorchIndexUrl "https://download.pytorch.org/whl/cpu"
```

### Tạo lại environment

Nếu environment cũ bị lỗi:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -RecreateVenv
```

Thao tác này chỉ xóa `.venv`, không xóa dữ liệu, code, checkpoint hoặc outputs.

## 4. Chuẩn bị AID

Ví dụ:

```text
D:\satellite-data\aid_target\
  Airport\
  BaseballField\
  Beach\
  Bridge\
  Church\
  Commercial\
  DenseResidential\
  Desert\
  Forest\
```

Số ảnh mỗi lớp có thể khác nhau nhưng mỗi lớp phải có ít nhất một ảnh.

## 5. Smoke test

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -Device cuda `
  -Amp `
  -SmokeOnly
```

CPU-only:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -Device cpu `
  -SmokeOnly
```

Smoke test chạy SmallCNN trong hai epoch. Thành công khi có:

```text
outputs\smoke\best.pt
outputs\smoke\summary.json
outputs\smoke\training_curves.png
```

## 6. Chạy full study

NVIDIA:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -Device cuda `
  -BatchSize 32 `
  -Workers 4 `
  -Amp
```

CPU:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -Device cpu `
  -BatchSize 16 `
  -Workers 4
```

Nếu bị lỗi DataLoader trên Windows, chạy lại với:

```powershell
-Workers 0
```

Nếu hết GPU memory, giảm `-BatchSize` xuống 16 hoặc 8.

Để tiếp tục một study bị ngắt:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -Device cuda `
  -Amp `
  -SkipExisting
```

## 7. Kiểm tra đúng số ảnh target nếu cần

Con số không bắt buộc. Tuy nhiên, nếu muốn script dừng khi target không đúng 5
ảnh/lớp:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -ExpectedTargetPerClass 5 `
  -SmokeOnly
```

## 8. Kết quả

Sau full study:

```text
outputs\cross_domain_runs.csv
outputs\cross_domain_models.csv
outputs\final\<model>\seed_<seed>\best.pt
outputs\final\<model>\seed_<seed>\aid_evaluation\metrics.json
outputs\final\<model>\seed_<seed>\aid_evaluation\predictions.csv
outputs\final\<model>\seed_<seed>\aid_evaluation\confusion_matrix.png
```

`cross_domain_models.csv` là bảng chính để so sánh model.

## 9. Inference trên Windows

Không cần activate `.venv`:

```powershell
.\.venv\Scripts\python.exe -m satdomain.infer `
  --checkpoint "outputs\final\resnet18_pretrained\seed_13\best.pt" `
  --image "D:\satellite-data\new_image.jpg" `
  --top-k 3 `
  --device cuda
```

## 10. Kiểm tra CUDA

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Nếu `nvidia-smi` hoạt động nhưng PyTorch trả `False`, cài lại PyTorch bằng index
URL lấy từ bộ chọn chính thức.

