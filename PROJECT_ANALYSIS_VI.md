**Analysis of the Satellite Image Classification across Different Domains project**

Updated against the code and conversation history on **27/09/2026**. This is a
self-study and handover guide; recheck it when the code changes. See the reusable
prompt at [REUSABLE_PROJECT_PROMPT.md](REUSABLE_PROJECT_PROMPT.md).

**1. What problem does the project solve? — VERY IMPORTANT**

The input is an overhead RGB image. The output is one of nine scene labels:
airport, baseball diamond, beach, bridge, church, commercial area, dense
residential area, desert, or forest. The label describes **the whole image
scene**. The model does not locate individual objects or classify individual pixels.

The research focuses on **generalization across data sources**: training on the
existing source and testing on AID. Under the agreed scope, the main report
contains only cross-domain results, with no same-domain test and no measurement
of a same-domain/cross-domain gap.

The `weather_robust` extension attempts to retain scene classification under
clouds/haze, illumination changes, loss of sharpness, or noise. It is not yet a
model for storm recognition, weather classification, or post-disaster damage assessment.

| Concept | Meaning in the project |
|---|---|
| Class | Content to predict, such as `forest` |
| Domain | Data source or distribution, such as source and AID |
| Architecture | Network structure, such as ResNet18 |
| Pretraining | Weights learned from other data before learning these nine classes |
| Checkpoint | Weights and information to reconstruct a run's model |
| Robustness | Stability when input images are transformed |
| Generalization | Ability to work on data not used for learning/model selection |

An image retains the `forest` label after a brightness change, provided its
scene remains meaningful. Do not add `storm` to the same nine-class label list
simply because weather is being studied.

**2. Actual data and remaining unknowns — VERY IMPORTANT**

The inventory was checked again: nine source directories in `data/source_subset/`,
100 images per directory, totaling 900 images.
The `airplane` and `basketball_court` classes were removed as previously requested.

| Label index | Source directory | Common corresponding AID name |
|---:|---|---|
| 0 | `airport` | Airport |
| 1 | `baseball_diamond` | BaseballField |
| 2 | `beach` | Beach |
| 3 | `bridge` | Bridge |
| 4 | `church` | Church |
| 5 | `commercial_area` | Commercial |
| 6 | `dense_residential` | DenseResidential |
| 7 | `desert` | Desert |
| 8 | `forest` | Forest |

This order is defined in [constants.py](src/satdomain/constants.py). Changing it
without updating checkpoints/manifests corrupts output meanings even if the
code still runs.

On 04/10/2026, the user confirmed the source as a NWPU-RESISC45 subset from
[blanchon/RESISC45](https://huggingface.co/datasets/blanchon/RESISC45), and the
target from [blanchon/AID](https://huggingface.co/datasets/blanchon/AID).
Both dataset pages were checked; individual local/Windows images had not yet
been compared with Hugging Face images, nor had the download revision and
subset selection procedure been identified.
Code/directories still call the source `source_subset`; reports may state
“NWPU-RESISC45 subset, origin confirmed by the user.” Geographic, capture-date,
seasonal, or sensor metadata is unavailable for individual experiment images.

Audit update on 04/10/2026: SHA-256 of 900 local source images matches the
RESISC45 mirror's LFS metadata at the pinned revision. [DATA.md](DATA.md) records
official URLs, versions, splits, preprocessing, a frozen manifest, and a ZIP for
reconstructing the current source. This does not confirm source manifests of
Windows runs. Three test directories have been provided:
`data/aid_subsets/{clean,bright,lowres}`, 180 images each. The reconstructed clean
manifest matches the hash of all 12 runs; bright/lowres filenames/labels match
the CSVs. Manifests, ZIPs, and a restore script are available; checkpoints,
processing code, and original-image mappings across the three conditions are
still missing.

According to your information, AID tested on Windows had **20 images per class =
180 images**. Code supports other counts or unequal class sizes but requires
all nine classes. Do not assume this Mac has the full AID dataset, checkpoints,
or outputs from Windows.

**3. Directory map of the entire project**

```text
ve-tinh/
├── data/source_subset/           9 source image directories, 100 images per directory
│   └── airport/ ... forest/
├── src/satdomain/                Python package implementing the study
│   ├── __init__.py               Package initialization
│   ├── constants.py              Labels and normalization parameters
│   ├── data.py                   Image loading, transforms, source split
│   ├── models.py                 SmallCNN and factory for 4 models
│   ├── train.py                  Development/final training, weight selection and saving
│   ├── evaluate.py               Evaluate one checkpoint on a manifest
│   ├── metrics.py                Accuracy, F1, confusion matrix, ECE
│   ├── infer.py                  Top-k prediction for one image
│   ├── robustness.py             Four simulated degradations
│   ├── artifacts.py              Hashes, provenance, snapshots, completion marker
│   ├── reports.py                Export tables and figures from saved results
│   └── runtime.py                Seeds, device selection, JSON writing
├── scripts/
│   ├── setup_windows.ps1         Create a venv, install dependencies on Windows
│   ├── run_windows.ps1           Wrapper for data preparation, training/testing/aggregation
│   ├── prepare_manifests.py      Inventory, label mapping, fold and manifest generation
│   ├── run_study.py              Coordinate development → final → AID
│   ├── aggregate_results.py      Aggregate models/seeds and per-class results
│   └── evaluate_robustness.py     Test 13 conditions for one checkpoint
├── configs/experiment.json       Study configuration description, not yet runtime configuration
├── research/methodology.md       Protocol and scope of conclusions
├── tests/
│   ├── test_research_artifacts.py Integrity and table export checks
│   └── test_robustness.py         Transformation and comparison checks
├── data/manifests/               CSV/JSON generated during data preparation
├── outputs/ or another path      Outputs on the execution machine; not assumed to exist
├── pyproject.toml                Dependencies, package, and CLI entry points
├── .gitignore                    Exclude weights/outputs/cache from Git
├── .orca/                        Conversation attachments; not the training source
└── *.md                          Usage and analysis documentation
```

`data/manifests/`, `outputs/`, `runs/`, and `.pt/.pth/.ckpt` checkpoints are ignored
by Git according to `.gitignore`. Therefore, cloning/pulling code does not mean
weights or results have been downloaded. `.git/` stores source history; `.venv/`,
if present, is a Python environment, not research data.

| Document | When to read it |
|---|---|
| [README.md](README.md) | For an overview of the main operations |
| [WINDOWS_GUIDELINE.md](WINDOWS_GUIDELINE.md) | To install and run on Windows CPU/CUDA |
| [RUN_ON_ANOTHER_MACHINE_GUIDELINE.md](RUN_ON_ANOTHER_MACHINE_GUIDELINE.md) | To transfer the project to another machine |
| [RESEARCH_ARTIFACTS_GUIDELINE.md](RESEARCH_ARTIFACTS_GUIDELINE.md) | To find manifests, checkpoints, F1, and plots |
| [ROBUSTNESS_GUIDELINE.md](ROBUSTNESS_GUIDELINE.md) | To run baseline/robust and understand simulation limitations |
| [research/methodology.md](research/methodology.md) | To write methods and determine valid conclusions |

**4. Execution flow from images to results — VERY IMPORTANT**

```text
Source folders + AID folders
        ↓ prepare_manifests.py
source.csv + aid_test.csv + summary.json
        ↓ run_study.py — stage 1
Source training/validation → select epoch → development/.../best.pt
        ↓ stage 2: recreate the model
Entire source → train for the selected epoch count → final/.../best.pt
        ↓ stage 3, after every model finishes training
AID → evaluate.py → predictions + metrics + confusion matrix
        ↓ aggregate_results.py
Model/seed comparison table and per-class F1
```

`run_windows.ps1` calls the corresponding scripts and automatically aggregates
results. When running `run_study.py` directly, call `aggregate_results.py` separately.

Additional branch: `evaluate_robustness.py` takes a final checkpoint and tests
clean AID images and transformed versions. Practical-use branch: `infer.py`
takes a checkpoint and a new image and returns top-k labels/confidence.

**5. Manifests and data splits — VERY IMPORTANT**

Read [prepare_manifests.py](scripts/prepare_manifests.py), especially `collect_domain()`.
A manifest is a frozen list of the images used, containing:

| Column | Meaning |
|---|---|
| `path` | Relative path from the data root |
| `label` | Canonical class name |
| `class_index` | Integer 0–8, consistent with the model |
| `fold` | Source fold or `-1` for the target |
| `domain` | `source` / `target` |
| `sha256` | File content fingerprint |

The script maps AID directory names to canonical names, checks counts, and
detects duplicate file contents within domains and between source/target.
However, **hashes do not detect near-duplicate images**, images of the same
location taken at different times, or recompressed versions of the same image.
The manifest preparation script does not fully decode images to verify labels/content.

The source is stratified by class into five folds using default split seed
`20260926`. With 100 images per class, each fold has 20 images per class.
A development run holds out one fold for validation: 720 training images and
180 validation images.

A key distinction: default training seeds `[13, 37, 73]` use folds `0, 1, 2`
respectively. This is **not full 5-fold cross-validation**. Between-run variation
includes initialization effects and changes in validation splits. When comparing
baseline and robust, preserve seed/fold pairs. To study seed effects alone,
design an experiment with a fixed fold.

**6. Preprocessing and DataLoader — VERY IMPORTANT**

Read [data.py](src/satdomain/data.py): `ManifestDataset`, `build_transforms()`, and
`source_train_validation_split()`.

| Stage | Image processing |
|---|---|
| Baseline training | RGB → Resize shorter edge to 256 → RandomResizedCrop 224, scale 0.8–1.0 → horizontal/vertical flip → rotation by multiples of 90° → tensor → normalization |
| Robust training | Same as baseline, with `RandomDegradation` after crop/flip/rotation and before tensor conversion |
| Clean validation/test | RGB → Resize shorter edge to 256 → CenterCrop 224 → tensor → normalization |
| Degraded validation/test | Transform the original RGB image before Resize/CenterCrop, then follow the test pipeline |

Network input tensors have shape `[batch, 3, 224, 224]`. All four models use the
ImageNet mean/std defined in `constants.py`; normalization is not estimated from AID.

Cropping can remove content near image edges; study this effect if changing
preprocessing. Keep the same rules when comparing models. Augmentation creates
transformed samples during loading, without modifying original image files or
creating additional independent samples.

`DataLoader` groups images into batches and uses workers to load them. Defaults
are batch size 32 and workers 4. Batch size, worker count, and GPU memory are
different concepts.

**7. Four models and their roles — VERY IMPORTANT**

Read [models.py](src/satdomain/models.py). Models only produce logits: nine
unnormalized scores. Softmax turns logits into confidence during evaluation/inference.

| CLI name | Initialization | Research role |
|---|---|---|
| `small_cnn` | Random | Custom CNN baseline, learning from the source |
| `resnet18_scratch` | Random | Standard CNN control |
| `resnet18_pretrained` | ImageNet weights through torchvision | Measure pretraining benefits with the same ResNet18 architecture |
| `deit_tiny_pretrained` | Pretrained through timm | Small transformer representative |

**SmallCNN: understand each block before studying complex networks.**

```text
3-channel RGB
 → Conv 3→32  + BatchNorm + ReLU + MaxPool
 → Conv 32→64 + BatchNorm + ReLU + MaxPool
 → Conv 64→128 + BatchNorm + ReLU + MaxPool
 → Conv 128→256 + BatchNorm + ReLU + MaxPool
 → AdaptiveAvgPool 1×1 → Flatten → Dropout 0.30 → Linear 256→9
```

Convolutions use 3×3 kernels, padding 1, and no bias. For a 224×224 image, four
pooling operations reduce spatial size to 14×14 before global pooling.
Convolutions learn features; pooling reduces spatial dimensions; the classifier
converts features into nine-class scores. Simplicity guarantees neither good
nor poor generalization: conclusions must rely on results.

**Scratch and pretrained ResNet18: the most important control pair.**

Both call `torchvision.models.resnet18()` and replace `fc` with a Linear layer
having nine outputs. The main difference is that scratch uses `weights=None`,
while pretrained uses `ResNet18_Weights.DEFAULT`. Residual connections are the
foundation of ResNet; read the [ResNet paper](https://arxiv.org/abs/1512.03385)
to understand why the network learns a correction through shortcut connections.

In the current code, the entire pretrained model is fine-tuned from the start;
**the backbone is not frozen, there is no two-stage fine-tuning, and learning
rates are not separated for backbone/classifier**. Earlier recommendations for
these techniques are not implemented features. Read about the difference between
full-network fine-tuning and using a backbone as a feature extractor in the
[PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

**DeiT-Tiny: a transformer representative.**

The factory calls `timm.create_model("deit_tiny_patch16_224.fb_in1k", pretrained=..., num_classes=9)`.
The configuration name specifies patch size 16 and input size 224. Study patch
embedding, attention, and the classifier. The [DeiT paper](https://arxiv.org/abs/2012.12877)
presents data-efficient transformer training and distillation; **this project
does not implement a teacher/student or distillation loss**, but fine-tunes an
existing model.

Comparing SmallCNN with DeiT changes architecture, representation capacity,
and pretraining simultaneously. Do not use that pair alone to attribute the
entire difference to pretraining. Do not assume pretrained models/transformers
always win on every domain either.

**8. Training and checkpoint selection — VERY IMPORTANT**

Read [train.py](src/satdomain/train.py): `run_epoch()`, `checkpoint_payload()`, and `main()`.

One training batch: load images → forward → CrossEntropyLoss → backward → optimizer
update. CrossEntropyLoss takes logits directly; do not add softmax before the
loss. During validation/testing, `model.eval()` and no-grad disable weight
updates and switch Dropout/BatchNorm behavior to evaluation mode.

| Default parameter | Current value/code |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | AdamW for all parameters |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Scheduler | CosineAnnealingLR, `T_max = epoch count of the run` |
| Maximum development length | 50 epochs |
| Patience | 8 epochs without selection-score improvement |
| Batch / image size | 32 / 224 |
| Default training seeds | 13, 37, 73 |
| AMP | Enabled only with the flag and a CUDA device |

Development selects epochs through validation. Final training **recreates the
model** and trains on the entire source for the selected epoch count; it does
not continue from development weights. For pretrained models, recreation means
returning to pretrained initialization and a new head; scratch models use
random initialization according to the seed.

`best.pt` has two distinct meanings:

- Development: weights at the epoch with the best validation score.
- Final: weights at the last epoch of the fixed epoch count; the file is rewritten
  each epoch. There is no AID validation to select a “best target checkpoint.”

The final scheduler uses a new `T_max` equal to the final epoch count, so its
learning-rate trajectory need not match the initial development segment.
This is the current refit choice; account for this difference when studying schedulers.

One default profile: 4 models × 3 seeds = 12 development runs + 12 final runs,
i.e. **24 training runs and 12 final checkpoints**. Run baseline and robust as
two separate studies; do not combine each model's best results from two different
environments.

**9. How does the new robustness feature work? — VERY IMPORTANT**

Read [robustness.py](src/satdomain/robustness.py), then the robust branch in
`train.py` and [evaluate_robustness.py](scripts/evaluate_robustness.py).

| Condition code | Transformation | What cannot be inferred directly |
|---|---|---|
| `cloud_haze` | Bright overlay with spatial variation | Storm presence/type, rainfall, or real cloud fraction |
| `illumination` | Change brightness/contrast | Season or capture time |
| `resolution` | Downsample, upscale, and blur | Actual sensor or GSD |
| `sensor_noise` | Add Gaussian noise to RGB | Sensor identity or SAR radar noise |

`RandomDegradation`: 50% probability of no additional degradation; in the remaining 50%,
uniformly select one of four types and severity 1 or 2. “Clean images” here
still receive basic crop/flip/rotation.

Baseline selects epochs by clean source-validation macro-F1. Robust selects by:

```text
selection_score = (clean F1 + cloud/haze F1 + illumination F1
                   + resolution F1 + noise F1) / 5
```

The four validation transformations use severity 2 and seed 7919, fixed across
epochs. Clean F1 and each transformed-condition F1 are still saved separately.
Each condition has 20% weight; a higher overall score does not guarantee higher
clean F1, so inspect both.

Robustness testing uses the same checkpoint on clean + 4 types × 3 severities =
13 passes. The default test seed is 2026; RNG depends on image hash, condition,
seed, and version `optical-proxies-v1`. Thus, the same image/condition receives
the same transformation across models. Severity 3 is not used in training/selection;
it is stronger, not an entirely new corruption type. Parameters for each severity
are defined directly in `degrade()`.

With 180 images, 13 passes produce 2,340 predictions but still only **180
independent images**. Baseline→robust changes both augmentation and checkpoint
selection; claiming a benefit specifically from augmentation requires an
ablation retaining the same selection criterion.

Operating instructions and limitations: [ROBUSTNESS_GUIDELINE.md](ROBUSTNESS_GUIDELINE.md).
Compare the scope with the [remote sensing robustness study](https://arxiv.org/abs/2306.12111);
the current transformations constitute a separate protocol and have not reproduced
the paper's benchmark.

**10. How should metrics be read? — VERY IMPORTANT**

Read [metrics.py](src/satdomain/metrics.py), function `classification_metrics()`.

| Metric | Question it answers |
|---|---|
| Accuracy | What percentage of images was predicted correctly? |
| Class precision | Of images predicted as that class, how many are correct? |
| Class recall | Of actual images of that class, how many does the model recognize correctly? |
| Class F1 | How are precision and recall balanced? |
| Macro-F1 | Equally weighted mean F1 across nine classes |
| Balanced accuracy | Equally weighted mean recall across classes |
| Confusion matrix | Which class pairs are confused? |
| ECE with 15 bins | How far does confidence differ from observed correctness rates? |

An example for understanding formulas, not an experimental result: if a class
has 20 actual images, 15 are recognized, and 18 images in total are predicted
as that class, recall = 15/20 and precision = 15/18. F1 is their harmonic mean.
Macro-F1 is the mean of per-class F1, not F1 computed from macro precision and
macro recall. See definitions in the [scikit-learn metrics documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).

Project matrices use rows for true labels and columns for predicted labels.
The row-normalized version makes each class's confusion rates easier to read.
With 20 images per class, one image changes that class's recall by 5 percentage
points; over 180 images, one image changes accuracy by approximately 0.56 points.

Mean ± standard deviation in summary tables measures between-run variation,
not a confidence interval. Bootstrap confidence intervals and statistical
significance tests have not been implemented. ECE exists, but fitted temperature
calibration, NLL, and reliability diagrams do not. `temperature=1.0` in a checkpoint
does not prove the model has been calibrated.

Robustness uses `drop_pp = 100 × (metric_clean − metric_degraded)`. This is the
drop due to image transformation within the same target set, **not the
source→target domain gap**. Negative drops are possible and must be reported honestly.

**11. Checkpoints, results, and reproducibility — VERY IMPORTANT**

Read [artifacts.py](src/satdomain/artifacts.py) and [reports.py](src/satdomain/reports.py).

```text
<output-root>/
├── study.json
├── manifests/{source.csv, aid_test.csv, generation_summary.json if available}
├── development/<model>/seed_<n>/
│   ├── best.pt, run.json, summary.json, complete.json
│   ├── source_manifest.csv, train_manifest.csv, validation_manifest.csv
│   └── history.json, history.csv, training_curves.png
├── final/<model>/seed_<n>/
│   ├── best.pt, run.json, summary.json, complete.json
│   ├── source_manifest.csv, train_manifest.csv, development_summary.json
│   ├── history.json, history.csv, training_curves.png
│   └── aid_evaluation/
│       ├── run.json, metrics.json, predictions.csv, target_manifest.csv
│       ├── confusion_matrix_counts.csv/.png
│       ├── confusion_matrix_normalized.csv/.png, confusion_matrix.png
│       └── per_class_metrics.csv, per_class_f1.png, complete.json
├── cross_domain_runs.csv, cross_domain_models.csv
└── per_class_runs.csv, per_class_models.csv
```

This structure is generated when all steps are run; outputs may be on the
Windows D: drive rather than in the Mac repository. Older runs may lack
provenance files added later.

`best.pt` contains state_dict, architecture, labels/class order, image size,
mean/std, seed, epoch, training arguments, temperature, and provenance. Metadata
links weights with manifests/hashes/code/environment. `complete.json` records
hashes of primary artifacts to verify completion; it does not hash every extra
exported plot.

`--skip-existing` only skips complete and valid runs; it **does not resume the
optimizer partway through an epoch/run**. Checkpoints do not yet save
optimizer/scheduler states for resumption. Keep incomplete outputs separately
for investigation and use a new directory for new training.

`reports.py` can regenerate figures from `history.json` and `metrics.json`.
Aggregate CSVs alone cannot recover individual predictions, training curves,
manifests, or weights.

`evaluate_robustness.py` creates separate output containing 13 condition
directories, `robustness.csv`, `summary.json`, and `robustness_curves.png`.
There is no script to automatically aggregate robustness across every
model/seed yet; run each checkpoint and aggregate the corresponding tables
while preserving the protocol.

**12. Inference and usage limitations**

Read [infer.py](src/satdomain/infer.py). It recreates the architecture with
`load_pretrained=False`, loads state_dict, switches to evaluation mode,
normalizes the image, computes softmax, and returns top-k results. Pretrained
weights need not be downloaded again when a complete checkpoint is available.
First-time pretrained training may need Internet access to obtain weights.

Images outside the nine classes are still forced into one of them: this is a
closed-set classifier. There is no out-of-distribution detector, cloud mask,
or automatic prediction rejection mechanism. Confidence of 94% does not mean
the probability of correctness is 94% in every domain.

Preprocessing currently uses `build_transforms()` and constants from the running
code. Although checkpoints store normalization, the loader does not automatically
restore an arbitrary normalization pipeline from metadata. If preprocessing
changes later, synchronize training/evaluation/inference and check compatibility
with old checkpoints.

**13. Common misunderstandings about configuration and reproducibility**

| Item | Actual status and implications |
|---|---|
| `configs/experiment.json` | Scripts do not yet read it as runtime configuration. Editing it does not automatically change training commands. Actual values come from CLI/defaults and are recorded in run metadata. |
| Class labels | Defined in both `constants.py` and `prepare_manifests.py`; adding a class requires synchronizing mapping, data, and classifier, then retraining. |
| Package parameters | `pyproject.toml` uses minimum version constraints without pinning every dependency exactly. Save actual environment versions when running. |
| Seed | Split seed, training seed, and corruption seed each control different things. |
| CPU/CUDA | The same seed does not guarantee identical weights/results across backends; the code has not enabled all deterministic algorithms. |
| `PYTHONHASHSEED` | Assigned when the process calls the seeding function; do not treat this as a guarantee that hash randomization was fixed at process startup. |
| Result aggregation | The aggregator saves hashes but does not fully verify that every run uses the same data/protocol; check provenance before drawing conclusions. |
| Data leakage | Hashes protect file content; they do not replace checks for near-duplicates, the same location, or targets already used to select improvements. |

**14. Existing results and what has not been demonstrated**

The following table was recorded from CSVs you sent and CPU/CUDA explanations
in a previous thread; these are not new measurements during this documentation
session, nor robustness results.

| Model | CUDA accuracy | CPU accuracy |
|---|---:|---:|
| SmallCNN | 68.33% | 68.33% |
| ResNet18 scratch | 71.11% | 77.22% |
| ResNet18 pretrained | 90.37% | 93.52% |
| DeiT-Tiny pretrained | 93.33% | 94.07% |

Whether CPU runs retrained or only tested CUDA checkpoints is unconfirmed;
shared manifests/seeds/configuration have not been fully verified. Differences
cannot yet be attributed to the device, nor can DeiT be concluded to always
outperform ResNet. Choose one consistent environment for the main results.

| Item | Status at the time of documentation |
|---|---|
| Four-model code, training/testing/inference | Implemented; user-run baseline results on Windows are available |
| Manifest/checkpoint provenance and reports | Added; only new runs have all new metadata |
| `weather_robust` and 13-condition testing | Code implemented; new weights/results have not been confirmed |
| Tests | The previous session passed 10 unit tests and checked transformations on a real source image |
| New end-to-end PyTorch/Matplotlib pipeline | Not run on the Mac; a Windows smoke test is needed |
| Geography, seasons, real weather | Not evaluated separately due to missing metadata |
| Storm, flood, or damage recognition | Not implemented |

The 10 tests focus on integrity, report tables, and transformation
reproducibility; they do not prove model quality or correct execution of all
CUDA training.

The 180 AID images have been viewed for analysis and to guide improvements.
If used during development, explicitly report results as exploratory and obtain
a new unseen target set for final evaluation. Blurring/adding haze to old images
does not restore independence.

**15. What should be prioritized for self-study?**

| Priority | Topic and files | Questions you should answer after reading |
|---|---|---|
| **Very important — 1** | Methodology + manifest + split | Why must AID not select epochs? How do 5 folds differ from full cross-validation? |
| **Very important — 2** | `data.py`, `constants.py` | How does an image become a tensor? What happens with incorrect class order/normalization? |
| **Very important — 3** | `models.py` | What does SmallCNN do? Which factor does comparing the ResNet pair control? |
| **Very important — 4** | `train.py`, `run_study.py` | How do loss/backprop/optimizer differ? How does development `best.pt` differ from final `best.pt`? |
| **Very important — 5** | `metrics.py`, `evaluate.py` | Why is high accuracy insufficient? Which classes are weak? Is confidence trustworthy? |
| **Very important — 6** | `robustness.py`, `evaluate_robustness.py` | What can simulations establish? Why do 13 passes not mean 13 times as many independent samples? |
| Important | `artifacts.py`, `reports.py` | How can you establish which checkpoint and dataset produced a result? |
| Important | `infer.py`, `runtime.py` | How is a model reused? Why can CPU/CUDA differ? |
| Read when operating | PowerShell scripts, `pyproject.toml`, Windows guide | How do you install the environment, run a smoke test, and find outputs? |

Suggested six-session path: (1) data/domains/leakage; (2) preprocessing and
SmallCNN; (3) ResNet/pretraining/DeiT; (4) training/validation/refit; (5) metrics
and misclassified-image analysis; (6) robustness/provenance and report result tables.

Check your understanding by explaining how an image goes from a directory to
a predicted label, how a checkpoint is selected, and which table supports a
research conclusion. If you cannot explain these three things, prioritize them
before adding a new architecture.

**16. Most valuable next steps**

1. Run a smoke test of the new version on Windows following the robustness guide.
2. Add the revision/subset selection procedure for declared sources, save actual
   image sets/manifests, and clarify the two CPU/CUDA experiment rounds.
3. Fix baseline/robust protocols, seeds/folds, environment, and an unseen target set.
4. Retrain and evaluate clean and degraded images; save complete outputs before aggregation.
5. Read per-class F1 and confusion matrices, rather than comparing only one accuracy column.
6. If the instructor requires real storms or geography/weather, design an
   additional dataset with appropriate metadata and labels instead of calling
   simulated corruptions real-world evidence.

The Mac is currently used for code/documentation development; training runs on
Windows. Continue following the shared machine's AGENTS.md: limit workers,
no stress/load tests, no Docker/Colima on the Mac, and clean up background
processes after work.
