# Saving experimental evidence and exporting plots

The pipeline still reports only cross-domain results on AID. Source validation
is used to select epochs. This feature does not change architectures or
hyperparameters to improve accuracy.

## 1. Results already run on Windows

If the entire `outputs` directory is still available, retraining is unnecessary
just to obtain plots. Update the code and use Python in the installed Windows
environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m satdomain.reports --output-root "D:\experiments\outputs_cuda"
```

The command reads every `history.json` and `metrics.json` under the selected
path and regenerates CSV/PNG files next to the originals. Run it similarly for
`outputs_cpu` if needed. It does not load checkpoints, retrain/retest, or require
a GPU. It replaces exported plots/tables with the same names while preserving
source JSON files and weights.

| Retained file | What can be regenerated |
|---|---|
| `history.json` | Training curves and per-epoch history tables |
| `aid_evaluation/metrics.json` | Confusion matrix, precision/recall/F1, and image counts per class |
| `best.pt` + exact AID images/manifest | Evaluation can be rerun to generate metrics and plots |
| Only `cross_domain_models.csv` | Confusion matrices, per-class F1, and training curves cannot be recovered |

Seed, architecture, epoch, and training arguments were already stored in old
code checkpoints. New code does not automatically treat a current manifest as
one used in the past. If the old manifest is missing, explicitly state that
this information is missing. Training curves cannot be recovered from weights
or aggregate accuracy alone.

If an old checkpoint is available but metrics are missing, reevaluate into a
NEW directory:

```powershell
.\.venv\Scripts\python.exe -m satdomain.evaluate `
  --checkpoint "D:\experiments\outputs_cuda\final\deit_tiny_pretrained\seed_13\best.pt" `
  --manifest "D:\experiments\manifests_original\aid_test.csv" `
  --data-root "D:\satellite-data\aid_target" `
  --output-dir "D:\experiments\reevaluation_deit_seed13" `
  --device cpu --workers 0
```

This is a new evaluation. `run.json` records the new test device, checkpoint
hash, seed from the checkpoint, and actual test manifest. For old checkpoints
without a training run ID, that value is left empty; historical provenance is
neither inferred nor fabricated.

## 2. New training runs: automatically save complete artifacts

Use a new output directory to avoid mixing results from before and after code changes:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -OutputRoot "D:\experiments\study_cuda_v2" `
  -Device cuda -Amp -Workers 4
```

For CPU, replace `-Device cuda -Amp` with `-Device cpu` and choose a different
output location. Retraining earlier results is unnecessary if you only need
to regenerate plots.

Each study saves:

```text
study_cuda_v2/
  study.json                         # seeds, settings, code and manifest hashes
  manifests/
    source.csv
    aid_test.csv
    generation_summary.json          # if available; includes the new manifest split seed
  development/<model>/seed_13/
    best.pt                          # selected by source-validation macro-F1
    run.json                         # run ID, seed, args, environment, AMP, preprocessing
    source_manifest.csv              # copy of the input manifest
    train_manifest.csv               # exact images used for training
    validation_manifest.csv          # exact images used for validation
    summary.json                     # epoch, checkpoint and source manifest hashes
    history.json
    history.csv
    training_curves.png
    complete.json                    # written only after completion, with artifact hashes
  final/<model>/seed_13/
    best.pt                          # last-epoch checkpoint after fitting the entire source
    run.json
    source_manifest.csv
    train_manifest.csv
    development_summary.json         # basis for the final fit epoch count
    summary.json
    history.json
    history.csv
    training_curves.png
    complete.json
    aid_evaluation/
      run.json                       # test device, checkpoint SHA-256, training seed/run ID
      target_manifest.csv
      metrics.json
      predictions.csv
      per_class_metrics.csv
      per_class_f1.png
      confusion_matrix_counts.csv
      confusion_matrix_counts.png
      confusion_matrix_normalized.csv
      confusion_matrix_normalized.png
      confusion_matrix.png           # backward-compatible name, normalized version
      complete.json
```

Seeds 37 and 73 have the same structure. The final model's `best.pt` name is
retained for compatibility with inference commands; it is the last-epoch
checkpoint, not one selected using AID. Environment JSON files distinguish
CPU/CUDA, AMP, and library versions. Recording a seed does not guarantee
bit-identical CPU and CUDA results.

Manifests store relative paths and SHA-256 for each image. Before training/testing,
the code checks that actual image contents match the manifest, class mapping,
and domain. Hashes detect byte changes/exact duplicates, but not all
near-duplicate images. Original images are not copied into outputs; retain
the original dataset separately for reproducibility.

## 3. Reading and using reports

- `per_class_metrics.csv`: class index, class name, precision, recall, F1, support.
  Metrics range from 0 to 1; support is the actual number of test images in that class.
- `per_class_f1.png`: per-class F1 chart with image counts.
- Confusion matrix: rows are true labels, columns are predicted labels. The
  counts version contains image counts; the normalized version contains proportions within each row.
- Development curves: training/validation loss, accuracy, and macro-F1.
- Final curves: training only, because the final fit uses the entire source.
  No artificial validation curve is created, and AID test accuracy is not used as a validation curve.

To aggregate per-class tables across models/seeds:

```powershell
.\.venv\Scripts\python.exe scripts\aggregate_results.py `
  --output-root "D:\experiments\study_cuda_v2" `
  --models small_cnn resnet18_scratch resnet18_pretrained deit_tiny_pretrained
```

In addition to `cross_domain_runs.csv` and `cross_domain_models.csv`, these are generated:

```text
per_class_runs.csv       # each model, seed, class: precision/recall/F1/support
per_class_models.csv     # mean and sample std across seeds, by model and class
```

Per-run cross-domain tables include checkpoint and target manifest hashes when
new evaluation data provides them. These fields are left empty for old outputs
without hashes. Do not add support across seeds and treat it as independent
image counts: the AID set remains the same.

## 4. Preserve results and continue execution

Training/evaluation refuse to overwrite nonempty output directories. Use a new
directory when changing configuration, code, or datasets.
`run_study.py --skip-existing` only skips runs with `complete.json` whose files
still match their hashes; having `best.pt` alone does not mean a run is complete.

If a run is interrupted, rename its incomplete directory to retain the history
(for example, move it to a backup directory outside the study), then rerun the
study with `-SkipExisting`. Completed runs are retained; missing runs start
again from the beginning. This does not resume the optimizer from the last epoch.

Old studies without `study.json` cannot be continued using the new mechanism;
export old reports or use a new output root. Do not manually add `complete.json`
to bypass checks.

For submission to the instructor, save the entire study directory, including
development and final, code/Git commit, and data source information.
Checkpoints/manifests/outputs are usually ignored by Git and must be backed up
separately. Do not save only aggregate accuracy tables.

## 5. Check on Windows before the full run

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
powershell -ExecutionPolicy Bypass -File scripts\run_windows.ps1 `
  -AidRoot "D:\satellite-data\aid_target" `
  -OutputRoot "D:\experiments\artifact_smoke_v2" `
  -Device cpu -Workers 0 -SmokeOnly
```

Tests check manifest/checkpoint integrity, prevention of run overwrites, class
names, CSV values, and the distinction between development/final curves.
Plotting is mocked in unit tests; a smoke test checks actual PyTorch/Matplotlib
execution on your machine.
