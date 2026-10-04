# Installing and running on Windows

Save experimental evidence and generate reports from previous results:
[RESEARCH_ARTIFACTS_GUIDELINE.md](RESEARCH_ARTIFACTS_GUIDELINE.md).
Add `-OutputRoot "D:\experiments\study_v2"` to create a separate study and avoid overwriting.

This is a short procedure specifically for Windows 10/11. The two included
PowerShell scripts automatically create a virtual environment and call the
correct Python inside `.venv`; manual environment activation is unnecessary.

## 1. Preparation

Install:

1. Python 3.11 or 3.12 from <https://www.python.org/downloads/windows/>.
2. Git for Windows if transferring the project through Git.
3. The NVIDIA driver if the machine has an NVIDIA GPU.

Copy or clone the entire repository, then open PowerShell in the `ve-tinh` directory.
Source training images are in `data\source_subset\`, in nine class directories;
`run_windows.ps1` uses this path automatically.

## 2. Allow scripts only in the current process

There is no need to change the system-wide execution policy. All commands below use:

```powershell
powershell -ExecutionPolicy Bypass -File <script>
```

## 3. Install the environment

### Simple method

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

The script will:

1. Create `.venv` using Python 3.12.
2. Upgrade pip.
3. Install `torch` and `torchvision`.
4. Install the project with `timm`, pandas, scikit-learn, Pillow, and matplotlib.
5. Check CUDA/MPS and import the package.

### NVIDIA CUDA

Open the official installation selector:

<https://docs.pytorch.org/get-started/locally/>

Select Windows, Pip, Python, and the appropriate CUDA option. Take the URL
following `--index-url` in the command provided by PyTorch, then run this example:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 `
  -TorchIndexUrl "URL_LAY_TU_TRANG_PYTORCH" `
  -RequireCuda
```

This example deliberately does not hard-code a CUDA version because PyTorch
updates its supported wheels over time.

### CPU-only

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 `
  -TorchIndexUrl "https://download.pytorch.org/whl/cpu"
```

### Recreate the environment

If the old environment is broken:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -RecreateVenv
```

This only deletes `.venv`; it does not delete data, code, checkpoints, or outputs.

## 4. Prepare AID

Example:

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

Image counts may differ between classes, but every class must have at least one image.

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

The smoke test runs SmallCNN for two epochs. It succeeds when these files exist:

```text
outputs\smoke\best.pt
outputs\smoke\summary.json
outputs\smoke\training_curves.png
```

## 6. Run the full study

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

If a DataLoader error occurs on Windows, rerun with:

```powershell
-Workers 0
```

If GPU memory runs out, reduce `-BatchSize` to 16 or 8.

To continue an interrupted study:

Only runs with a valid `complete.json` are skipped. Move incomplete runs to a
backup directory before rerunning; those runs train again from the beginning,
rather than resuming an epoch.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -Device cuda `
  -Amp `
  -SkipExisting
```

## 7. Check the exact target image count if needed

There is no mandatory count. However, to make the script stop if the target
does not contain exactly 5 images per class:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -ExpectedTargetPerClass 5 `
  -SmokeOnly
```

## 8. Results

After the full study:

```text
outputs\cross_domain_runs.csv
outputs\cross_domain_models.csv
outputs\final\<model>\seed_<seed>\best.pt
outputs\final\<model>\seed_<seed>\aid_evaluation\metrics.json
outputs\final\<model>\seed_<seed>\aid_evaluation\predictions.csv
outputs\final\<model>\seed_<seed>\aid_evaluation\confusion_matrix.png
```

`cross_domain_models.csv` is the main table for comparing models.

## 9. Inference on Windows

There is no need to activate `.venv`:

```powershell
.\.venv\Scripts\python.exe -m satdomain.infer `
  --checkpoint "outputs\final\resnet18_pretrained\seed_13\best.pt" `
  --image "D:\satellite-data\new_image.jpg" `
  --top-k 3 `
  --device cuda
```

## 10. Check CUDA

```powershell
nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If `nvidia-smi` works but PyTorch returns `False`, reinstall PyTorch using the
index URL from the official selector.
