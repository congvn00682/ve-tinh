# Satellite Image Classification across Different Domains

This project studies the generalization of scene-classification models trained
on the existing source domain and tested on AID. Code is prepared on the
development machine, while PyTorch installation and training take place on
another machine.

## Repository contents

| Requirement | File/directory |
|---|---|
| Complete source code | `src/satdomain/` and `scripts/`: data, four models, training/refit, evaluation, inference, robustness, and reports |
| README | This document describes installation, data, and execution commands |
| Dependencies | [pyproject.toml](pyproject.toml) is the canonical declaration; [requirements.txt](requirements.txt) installs the project with its dependencies |
| Dataset documentation | [DATA.md](DATA.md): official URLs, revisions, splits, preprocessing, frozen manifest, and source subset ZIP |
| Training/evaluation instructions | The steps below and the [Windows guide](WINDOWS_GUIDELINE.md) |
| Results | [results/README.md](results/README.md): provided original CSVs, summary tables, confusion matrices, and evidence limitations |

```text
ve-tinh/
├── src/satdomain/             ML package: data, models, train, evaluate, infer...
├── scripts/                  Research CLI, Windows wrappers, CSV aggregation
├── data/source_subset/       900 source images in nine class directories
├── data/aid_subsets/         Clean/bright/lowres, 180 images per set
├── data/provenance/          Frozen manifests and data metadata
├── data/releases/            Source/target ZIPs and SHA-256
├── configs/experiment.json  Study description; not yet runtime configuration
├── research/methodology.md  Protocol and scope of conclusions
├── results/                 Provided results and tables recalculated from CSVs
├── tests/                   Artifact, robustness, and result table tests
├── pyproject.toml           Python/package/dependencies
└── requirements.txt         Install the editable package through pyproject.toml
```

Runtime parameters come from CLI arguments/script defaults; editing
`configs/experiment.json` does not automatically change training. `outputs/`,
generated manifests, and weights are ignored by Git; submitted result CSVs are
in `results/` so they can be stored in Git.

For self-study, read the [complete project analysis](PROJECT_ANALYSIS_VI.md):
structure, models, training/evaluation, limitations, and recommended code reading
order. The [reusable handover prompt](REUSABLE_PROJECT_PROMPT.md) preserves context
for continuing in a new thread; fill in the specific task before using it.

Detailed instructions for transferring and running on another machine are in
[RUN_ON_ANOTHER_MACHINE_GUIDELINE.md](RUN_ON_ANOTHER_MACHINE_GUIDELINE.md).

Windows has a PowerShell installer and a dedicated guide at
[WINDOWS_GUIDELINE.md](WINDOWS_GUIDELINE.md).

Saving manifests/seeds/checkpoints and exporting confusion matrices, per-class
F1, and training curves, including for existing results:
[RESEARCH_ARTIFACTS_GUIDELINE.md](RESEARCH_ARTIFACTS_GUIDELINE.md).

The fixed methodology is in [research/methodology.md](research/methodology.md).

The improvement profile for experiments with clouds/haze, illumination,
resolution, and image noise is described in
[ROBUSTNESS_GUIDELINE.md](ROBUSTNESS_GUIDELINE.md). Use `-Augmentation weather_robust`
on Windows; baseline remains the default. Training and evaluation are needed
to establish its effectiveness.

## Data design

The main experiment has 9 classes shared by the source and AID:

```text
airport, baseball_diamond, beach, bridge, church,
commercial_area, dense_residential, desert, forest
```

The repository retains only these 9 classes; every model uses the same label space.

- The current source has 100 images per class; the code does not require a fixed count.
- Source images are in `data/source_subset/`, in nine separate class directories.
- AID target counts may vary; it is only used for the final evaluation.
- Do not use AID for early stopping or hyperparameter selection.
- Reports contain only cross-domain results on AID, with no same-domain test.

Data sources confirmed by the user on 04/10/2026:

- Training source: a NWPU-RESISC45 subset from [blanchon/RESISC45](https://huggingface.co/datasets/blanchon/RESISC45).
  The dataset card describes the full set of 45 classes and 31,500 RGB images at
  256×256; the repo uses only nine classes, totaling 900 images.
- Test target: a subset from [blanchon/AID](https://huggingface.co/datasets/blanchon/AID).
  The dataset card describes the full set of 30 classes and 10,000 RGB images at
  600×600; the user-provided test uses 20 images per class in nine classes,
  totaling 180 images.

The dataset pages were checked, and SHA-256 hashes of 900 local source images
were matched against the mirror's LFS metadata at the pinned revision. Target
image bytes have not been matched against the mirror, and historical download
revisions/subset selection procedures have not been identified. The manifest
reconstructed from the provided clean images matches the hash of all 12 runs;
bright/lowres filenames and labels match the CSVs. See [DATA.md](DATA.md) to
download the source ZIP and reconstruct the current images/split.
Retain the name `source_subset` in the configuration and directory `data/source_subset/`.
The Hugging Face split name `train` does not determine its research role:
AID remains an evaluation-only target under the project protocol.

## 1. Preparation on the training machine

Python 3.10–3.13 is recommended. Create a virtual environment and first install
the appropriate PyTorch build for the training machine's CUDA/CPU, then install
the project:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install torch/torchvision for the training machine GPU, then:
python -m pip install -e .
```

The final command can be replaced with `python -m pip install -r requirements.txt`.
Version constraints are declared in `pyproject.toml`:

| Group | Dependencies |
|---|---|
| Python | `>=3.10,<3.14` |
| Model/training | `torch>=2.3`, `torchvision>=0.18`, `timm>=1.0` |
| Data/metrics/plots | `numpy>=1.26`, `pandas>=2.2`, `Pillow>=10.2`, `scikit-learn>=1.4`, `matplotlib>=3.8` |
| Build | `setuptools>=69` |

These are installation constraints, not a lockfile for the environment that
produced the CSVs. GPU/CUDA/library versions for the provided results have not
been fully confirmed.

On Windows, use PowerShell in the repository directory:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_subset" `
  -OutputRoot "D:\experiments\smoke_new" -Device cpu -Workers 0 -SmokeOnly
```

For CUDA, select the wheel/index URL following the [Windows guide](WINDOWS_GUIDELINE.md)
and install with `setup_windows.ps1 -TorchIndexUrl ... -RequireCuda`.

Docker is not required.

## 2. AID directory structure

The script accepts both original AID directory names and lowercase variants. Example:

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

Directories need not have equal image counts. The manifest records actual counts
and stops if a class is empty, duplicate images exist within a domain, or source
and target contain images with identical contents. Use
`--expected-target-per-class N` when explicitly checking for exactly `N` images per class.

## 3. Generate manifests and audit

Run from the repository directory:

```bash
python scripts/prepare_manifests.py \
  --source-root data/source_subset \
  --aid-root /path/to/aid_subset \
  --output-dir data/manifests
```

Outputs:

```text
data/manifests/source.csv
data/manifests/aid_test.csv
data/manifests/summary.json
```

Each row contains a relative path, class, class index, fold, domain, and SHA-256.

## 4. Smoke test one model

```bash
python -m satdomain.train \
  --manifest data/manifests/source.csv \
  --data-root data/source_subset \
  --arch small_cnn \
  --mode development \
  --validation-fold 0 \
  --epochs 2 \
  --workers 4 \
  --output-dir outputs/smoke
```

On an NVIDIA GPU, you can add `--amp`. Do not use `--amp` for CPU/MPS.

## 5. Run the entire study

Long sessions should run in tmux. The script uses source training/validation to
select the epoch count, then retrains on the entire source. Only after every
model has finished training does the script open AID for cross-domain testing:

```bash
python scripts/run_study.py \
  --source-manifest data/manifests/source.csv \
  --source-root data/source_subset \
  --target-manifest data/manifests/aid_test.csv \
  --target-root /path/to/aid_subset \
  --output-root outputs \
  --workers 4 \
  --device cuda \
  --amp
```

Default models:

```text
small_cnn
resnet18_scratch
resnet18_pretrained
deit_tiny_pretrained
```

CPU: replace `--device cuda --amp` with `--device cpu`. By default, seeds 13/37/73
use validation folds 0/1/2 respectively in the five-fold manifest; this is not
full 5-fold CV. Baseline selects epochs by source validation macro-F1, recreates
the model, and trains on the entire source for the selected epoch count.
Final `best.pt` contains last-epoch weights; checkpoints are not selected using AID.

Equivalent full-study command on Windows:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_subset" `
  -OutputRoot "D:\experiments\study_new" -Device cuda -Workers 4 -Amp
```

Add `--augmentation weather_robust` to `run_study.py`, or
`-Augmentation weather_robust` to the Windows wrapper, to run the robust profile.
Use a separate output root for each study/profile.

`--skip-existing` only skips completed runs with matching hashes. Incomplete
runs must be moved to a backup directory before restarting from the beginning;
the optimizer is not resumed.

## 6. Aggregate results

```bash
python scripts/aggregate_results.py \
  --output-root outputs \
  --models small_cnn resnet18_scratch resnet18_pretrained deit_tiny_pretrained
```

Two cross-domain tables are generated:

```text
outputs/cross_domain_runs.csv
outputs/cross_domain_models.csv
```

Each final run also contains AID `metrics.json`, `predictions.csv`, confusion
matrix PNGs, and training curves. Source validation only controls training and
is not included in the research result tables.

## 7. Evaluate one checkpoint

A checkpoint and the correct target images/manifest are required. Provided test
images are in `data/aid_subsets/`; the clean manifest is
`data/provenance/aid_clean_v1.csv`. Weights for the provided runs are still absent
from the repo. After training with the steps above, evaluate a final checkpoint
separately:

```bash
python -m satdomain.evaluate \
  --checkpoint outputs/final/deit_tiny_pretrained/seed_13/best.pt \
  --manifest data/provenance/aid_clean_v1.csv \
  --data-root data/aid_subsets/clean \
  --output-dir outputs/reevaluation_deit_seed13 \
  --device cuda --workers 4
```

Windows uses the same CLI; replace `python` with `.\.venv\Scripts\python.exe` and
write a single-line command or use backticks for line continuation. Choose a new
output directory.

Test the current code's 13 simulated conditions:

```bash
python scripts/evaluate_robustness.py \
  --checkpoint outputs/final/deit_tiny_pretrained/seed_13/best.pt \
  --manifest data/provenance/aid_clean_v1.csv \
  --data-root data/aid_subsets/clean \
  --output-dir outputs/robustness_deit_seed13 \
  --device cuda --workers 4 --corruption-seed 2026
```

This is clean + four corruptions × three severities, distinct from the
user-provided bright/lowres CSVs: the tool/parameters used to create bright/lowres
images have not been confirmed. Keep the same checkpoint, manifest, and
corruption seed when comparing conditions. See
[ROBUSTNESS_GUIDELINE.md](ROBUSTNESS_GUIDELINE.md) for simulation limitations.

## 8. Inference on a new image

```bash
python -m satdomain.infer \
  --checkpoint outputs/final/resnet18_pretrained/seed_13/best.pt \
  --image /path/to/satellite_image.jpg \
  --top-k 3
```

The output is JSON containing labels and confidence. This is a closed-set
classifier: images outside the 9 classes are still forced into a known class.
Confidence is not guaranteed to be calibrated on a new domain either.

## 9. Existing results and recalculation

Accuracy below is mean ± sample SD (%), for three seeds 13/37/73 and 180 images
per condition/run. CUDA was confirmed by the user; whether it was used for
training, testing, or both is unclear.

| Model | Set described as clean | Bright | Lowres |
|---|---:|---:|---:|
| SmallCNN | 72.78 ± 2.22 | 45.19 ± 5.89 | 59.26 ± 0.64 |
| ResNet18 scratch | 74.44 ± 1.92 | 54.63 ± 5.04 | 67.59 ± 1.16 |
| ResNet18 pretrained | 88.89 ± 3.09 | 80.56 ± 1.92 | 86.85 ± 0.32 |
| DeiT-Tiny pretrained | 95.19 ± 1.40 | 87.22 ± 1.11 | 85.93 ± 2.25 |

These are provided results, not new training during repository preparation.
The baseline/weather_robust profile and use of the same checkpoints across
conditions have not been confirmed. Bright/lowres filenames do not overlap;
whether these are renamed versions of the same original images or different
sets is unknown. Do not interpret differences from clean as corruption drops.
Mean ± SD is not a confidence interval.

The following script only reads saved CSVs and recalculates tables; ML
dependencies are not required:

```bash
python scripts/summarize_submitted_results.py
```

See [results/README.md](results/README.md) for file sources, macro-F1, per-class
metrics, confusion matrices, SHA-256, and missing information. The script
preserves original CSVs and replaces derived tables with the same names.
It does not recover checkpoints/training curves.

## 10. Check the code

In an environment with dependencies installed on the training machine:

```bash
python -m unittest discover -s tests -v
```

To check only the result bundle and artifacts using the Python standard library:

```bash
python -m unittest discover -s tests -p test_submitted_results.py -v
python -m unittest discover -s tests -p test_research_artifacts.py -v
```

These tests verify integrity and calculations; they do not demonstrate an
accuracy improvement. The Mac is used for code/documentation development;
full PyTorch training/evaluation runs on Windows. Keep workers at a maximum
of 4 on the shared machine and close processes when finished.

## Reproducibility notes

- Keep manifests unchanged after starting an experiment.
- Do not tune models based on AID results.
- Do not report smoke-test results as research results.
- Commit code/configuration before running the final study and record the commit hash.
