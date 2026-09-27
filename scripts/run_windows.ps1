[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AidRoot,
    [ValidateSet("auto", "cuda", "cpu", "mps")]
    [string]$Device = "auto",
    [int]$BatchSize = 32,
    [ValidateRange(0, 4)]
    [int]$Workers = 4,
    [switch]$Amp,
    [switch]$SmokeOnly,
    [switch]$SkipExisting,
    [int]$ExpectedTargetPerClass = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PythonExe = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ManifestRoot = Join-Path $ProjectRoot "data\manifests"
$OutputRoot = Join-Path $ProjectRoot "outputs"

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

if (-not (Test-Path $PythonExe)) {
    throw "Missing $PythonExe. Run scripts\setup_windows.ps1 first."
}

if (-not (Test-Path -LiteralPath $AidRoot -PathType Container)) {
    throw "AID target folder does not exist: $AidRoot"
}

$AidRoot = (Resolve-Path -LiteralPath $AidRoot).Path
Write-Host "Project: $ProjectRoot" -ForegroundColor Green
Write-Host "AID target: $AidRoot" -ForegroundColor Green
Write-Host "Device: $Device; batch size: $BatchSize; workers: $Workers"

Push-Location $ProjectRoot
try {
    $ManifestArguments = @(
        "scripts/prepare_manifests.py",
        "--source-root", ".",
        "--aid-root", $AidRoot,
        "--output-dir", $ManifestRoot
    )
    if ($ExpectedTargetPerClass -gt 0) {
        $ManifestArguments += @(
            "--expected-target-per-class",
            $ExpectedTargetPerClass.ToString()
        )
    }
    Invoke-Checked -Description "Audit datasets and create manifests" -Command {
        & $PythonExe @ManifestArguments
    }

    if ($SmokeOnly) {
        $SmokeRoot = Join-Path $OutputRoot "smoke"
        $SmokeArguments = @(
            "-m", "satdomain.train",
            "--manifest", (Join-Path $ManifestRoot "source.csv"),
            "--data-root", ".",
            "--arch", "small_cnn",
            "--mode", "development",
            "--validation-fold", "0",
            "--epochs", "2",
            "--batch-size", ([Math]::Min($BatchSize, 16)).ToString(),
            "--workers", $Workers.ToString(),
            "--device", $Device,
            "--output-dir", $SmokeRoot
        )
        if ($Amp) {
            $SmokeArguments += "--amp"
        }
        Invoke-Checked -Description "Run two-epoch SmallCNN smoke test" -Command {
            & $PythonExe @SmokeArguments
        }
        Write-Host "`nSmoke test completed: $SmokeRoot" -ForegroundColor Green
        exit 0
    }

    $StudyArguments = @(
        "scripts/run_study.py",
        "--source-manifest", (Join-Path $ManifestRoot "source.csv"),
        "--source-root", ".",
        "--target-manifest", (Join-Path $ManifestRoot "aid_test.csv"),
        "--target-root", $AidRoot,
        "--output-root", $OutputRoot,
        "--batch-size", $BatchSize.ToString(),
        "--workers", $Workers.ToString(),
        "--device", $Device
    )
    if ($Amp) {
        $StudyArguments += "--amp"
    }
    if ($SkipExisting) {
        $StudyArguments += "--skip-existing"
    }
    Invoke-Checked -Description "Run complete cross-domain study" -Command {
        & $PythonExe @StudyArguments
    }

    $AggregateArguments = @(
        "scripts/aggregate_results.py",
        "--output-root", $OutputRoot,
        "--models",
        "small_cnn",
        "resnet18_scratch",
        "resnet18_pretrained",
        "deit_tiny_pretrained"
    )
    Invoke-Checked -Description "Aggregate model comparison tables" -Command {
        & $PythonExe @AggregateArguments
    }
}
finally {
    Pop-Location
}

Write-Host "`nCross-domain study completed successfully." -ForegroundColor Green
Write-Host "Per-run results: $OutputRoot\cross_domain_runs.csv"
Write-Host "Model comparison: $OutputRoot\cross_domain_models.csv"

