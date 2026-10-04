# Scene classification under image degradation

The goal remains predicting the existing 9 scene classes. The `weather_robust`
profile adds augmentation and a checkpoint selection criterion; retraining on
Windows is needed to obtain new weights. No results yet demonstrate that this
profile outperforms baseline.

## Four condition groups

| Group | Implemented transformation | Limitation |
|---|---|---|
| Weather: clouds/haze | Spatially structured bright overlay that reduces contrast | Does not simulate a storm or assign real weather labels |
| Illumination | Increase/decrease brightness and contrast | Not evidence of day/night or seasons |
| Resolution/sharpness | Downsample, resize back, and blur | Does not represent a specific meters-per-pixel value |
| Image noise | Gaussian noise on RGB | Does not represent a particular sensor or SAR noise |

Storms can cause cloud cover, flooding, and actual scene changes. Complete cloud
cover removes ground information; augmentation cannot recover that information.
The current model does not identify storms or detect damage/flooding. Those
tasks require different data, labels, and designs. Do not add a `storm` class
to the 9 scene labels.

References:

- [ESA: Flood Extent](https://knowledge-hub-gda.esa.int/eo_capability/flood-extent/):
  clouds limit optical imagery; SAR radar supports flood observation under cloudy conditions.
- [A Comprehensive Study on the Robustness of Image Classification and Object Detection in Remote Sensing](https://arxiv.org/abs/2306.12111):
  a study evaluating robustness in remote sensing. The project's transformations
  constitute its own protocol, not a reproduction of the paper's benchmark.

## Training and checkpoint selection

- `baseline`: retains the previous augmentation and selects checkpoints by source validation macro-F1.
- `weather_robust`: each time a training image is sampled, there is a 50%
  probability of no additional degradation; in the remaining 50%, one of the four groups
  and severity 1 or 2 is selected uniformly.
- Basic crop/flip/rotation still applies in both profiles. Original files on
  disk are not modified, and no additional dataset copies are generated.
- Robust checkpoint selection uses mean macro-F1 across **5 source validation
  conditions**: clean + four severity-2 transformation groups. Each condition has
  equal weight; the fixed transformation seed is 7919. Clean F1, each condition's
  F1, and the selection score are saved separately in `history.json`/CSV and training curves.
- Each robust epoch requires four additional small validation passes; AID is not used to select epochs.
- After development, retrain on the entire source with the selected augmentation, then test.
- Severity 3 is not used in training/validation: it tests stronger degradation,
  not a previously unseen type of weather.

## Run on Windows

Open PowerShell in the project directory after installing the environment as
described in `WINDOWS_GUIDELINE.md`. Run a smoke test first (2 epochs, SmallCNN):

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_subset" `
  -OutputRoot "D:\experiments\robust_smoke_v1" `
  -Device cuda -Workers 4 -Amp -SmokeOnly -Augmentation weather_robust
```

Train all four models with the new profile:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_subset" `
  -OutputRoot "D:\experiments\robust_v1" `
  -Device cuda -Workers 4 -Amp -Augmentation weather_robust
```

CPU: use `-Device cpu` and remove `-Amp`. Use a new directory for each experiment;
do not mix CPU/CUDA results or overwrite earlier results. The full-study command
automatically tests clean AID after all training finishes; degraded testing uses
the command below.

## Test one checkpoint's robustness

This command works with both old baseline and new checkpoints; it does not train:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_robustness.py `
  --checkpoint "D:\experiments\robust_v1\final\resnet18_pretrained\seed_13\best.pt" `
  --manifest data\manifests\aid_test.csv `
  --data-root "D:\satellite-data\aid_subset" `
  --output-dir "D:\experiments\robustness_results\robust_resnet18_seed13" `
  --device cuda --workers 4 --corruption-seed 2026
```

Rerun with the baseline checkpoint and a different output directory; preserve
the manifest, transformation seed, device, and code version. Repeat for each
model/seed. A checkpoint is evaluated in 13 passes: clean and 4 groups × 3 severities.
With 180 images, this is still **180 independent samples**, not 2,340 independent images.

Outputs:

- `robustness.csv`: accuracy, balanced accuracy, macro-F1, and the drop relative
  to clean images, expressed in **percentage points**. A negative drop means a
  higher score than on clean images.
- `robustness_curves.png`: curves across degradation severities.
- `summary.json`: clean F1, mean across 12 degraded conditions, and the worst condition.
- Each condition has its own `predictions.csv`, confusion matrix, per-class F1,
  `metrics.json`, manifest, and checkpoint/hash/seed information.

Evaluation transformations operate on original RGB images before resize/crop.
RNG depends on image SHA-256, condition name, and transformation seed; it is
independent of data order, training seed, and worker count. The transformation
version is recorded for reproducibility.

## Drawing conclusions about improvements

Compare baseline and robust using the same manifest, training seed, validation
fold, and execution conditions. Report clean F1, degraded mean F1,
worst-condition F1, and per-class results; do not select only the highest cell.
Mean ± SD is computed across training runs; transformed versions of the same
image are not independent samples.

The robust profile changes both augmentation and checkpoint selection: this
compares two pipelines. Do not attribute the entire improvement specifically
to augmentation without an ablation holding checkpoint selection fixed.

The 180 AID images were viewed in earlier sessions. If those results guide
improvements, report them as a development/exploratory set; use a new, unseen
AID set for final evaluation. Fix the protocol before opening that set.
Transformations of the old set do not create a new independent test set.

Claiming real-world storm applicability requires additional real images with
event metadata, cloud-cover levels, and scene labels; split by region/event to
avoid leakage. Images lacking sufficient information should be flagged for
manual review. A quality detector or automatic prediction rejection mechanism
has not been implemented.

## Check the code

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests verify reproducibility, preservation of original images, and correct
comparison tables. They do not replace a PyTorch smoke test or evidence of
improved accuracy.
