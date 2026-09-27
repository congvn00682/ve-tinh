[CmdletBinding()]
param(
    [string]$PythonVersion = "3.12",
    [string]$TorchIndexUrl = "",
    [switch]$RequireCuda,
    [switch]$RecreateVenv
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VenvRoot = Join-Path $ProjectRoot ".venv"
$PythonExe = Join-Path $VenvRoot "Scripts\python.exe"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [scriptblock]$Command,
        [Parameter(Mandatory = $true)]
        [string]$Description
    )

    Write-Host "`n==> $Description" -ForegroundColor Cyan
    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Project: $ProjectRoot" -ForegroundColor Green

if ($RecreateVenv -and (Test-Path $VenvRoot)) {
    Write-Host "Removing existing virtual environment: $VenvRoot" -ForegroundColor Yellow
    Remove-Item -LiteralPath $VenvRoot -Recurse -Force
}

if (-not (Test-Path $PythonExe)) {
    $PyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -ne $PyLauncher) {
        $VersionArgument = "-$PythonVersion"
        Invoke-Checked -Description "Create Python $PythonVersion virtual environment" -Command {
            & py $VersionArgument -m venv $VenvRoot
        }
    }
    else {
        $SystemPython = Get-Command python -ErrorAction SilentlyContinue
        if ($null -eq $SystemPython) {
            throw "Python was not found. Install Python 3.11 or 3.12, then run this script again."
        }
        Invoke-Checked -Description "Create Python virtual environment" -Command {
            & python -m venv $VenvRoot
        }
    }
}

if (-not (Test-Path $PythonExe)) {
    throw "Virtual environment creation failed: $PythonExe does not exist."
}

Invoke-Checked -Description "Upgrade pip, setuptools and wheel" -Command {
    & $PythonExe -m pip install --upgrade pip setuptools wheel
}

if ([string]::IsNullOrWhiteSpace($TorchIndexUrl)) {
    Write-Host "`nNo PyTorch index URL was supplied." -ForegroundColor Yellow
    Write-Host "Installing torch and torchvision from the default Python package index."
    Write-Host "For a specific CUDA build, copy the index URL from:" 
    Write-Host "https://docs.pytorch.org/get-started/locally/" -ForegroundColor Blue
    Invoke-Checked -Description "Install PyTorch and torchvision" -Command {
        & $PythonExe -m pip install torch torchvision
    }
}
else {
    Invoke-Checked -Description "Install PyTorch from selected package index" -Command {
        & $PythonExe -m pip install torch torchvision --index-url $TorchIndexUrl
    }
}

Push-Location $ProjectRoot
try {
    Invoke-Checked -Description "Install satdomain and research dependencies" -Command {
        & $PythonExe -m pip install -e .
    }
}
finally {
    Pop-Location
}

Invoke-Checked -Description "Verify Python packages" -Command {
    & $PythonExe -c "import torch, torchvision, timm, satdomain; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available()); print('satdomain: OK')"
}

if ($RequireCuda) {
    $CudaAvailable = (& $PythonExe -c "import torch; print(torch.cuda.is_available())").Trim()
    if ($CudaAvailable -ne "True") {
        throw "CUDA was required but PyTorch reports CUDA=False. Check the NVIDIA driver and install the correct CUDA wheel."
    }
}

Write-Host "`nInstallation completed successfully." -ForegroundColor Green
Write-Host "Virtual environment Python: $PythonExe"
Write-Host "Next step example:"
Write-Host "powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 -AidRoot D:\data\aid_target -SmokeOnly"

