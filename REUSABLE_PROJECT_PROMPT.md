**Project handover prompt for use in a new thread**

Copy the entire content of the block below. Replace “Task for this session” and
Windows paths with actual information. This is a snapshot dated 27/09/2026;
ask the assistant to recheck the code rather than assume everything remains current.

```text
Act as a Machine Learning/Deep Learning/Computer Vision research mentor and
help implement and check the code for the following project.
Communicate in Vietnamese, explain clearly, and distinguish existing code from
verified experimental results. Do not fabricate accuracy, datasets, or conclusions.

TASK FOR THIS SESSION
[Fill in a specific request: analyze results, fix Windows errors, run/design
experiments, write a report, improve a model, or explain part of the code.]

Files/paths provided for this session:
- Repository: [current path]
- AID: [path if needed]
- Output/checkpoint: [path if needed]
- Runtime environment: [Windows CPU or NVIDIA CUDA; versions if known]

1. CONTEXT AND SCOPE

Project title: Satellite Image Classification across Different Domains.
Previous development repository:
/Users/ac-codex-4/projects/congvn/model-image/ve-tinh

This study classifies whole RGB satellite/aerial image scenes. It compares models
trained and tested on different data sources. The agreed scope is cross-domain
only: source → AID, with no same-domain testing or domain-gap reporting.

The current output is one of 9 scene classes. The new robustness profile aims
to retain classification when image quality degrades. There is no model for
storm recognition, real weather classification, flood segmentation, or
post-disaster damage assessment yet.

2. DATA

The current source has 900 images, 100 images per class:
airport, baseball_diamond, beach, bridge, church, commercial_area,
dense_residential, desert, forest.

Source images are in `data/source_subset/`, in nine class directories.

The airplane and basketball_court classes were removed. Class index order is
as listed above. On 04/10/2026, the user confirmed the source as NWPU-RESISC45:
https://huggingface.co/datasets/blanchon/RESISC45
The target is from AID: https://huggingface.co/datasets/blanchon/AID
Dataset pages have been checked. Audit on 04/10/2026: SHA-256 of 900 local source
images matches LFS metadata at the RESISC45 revision pinned in DATA.md.
A frozen manifest and source ZIP can reconstruct the current subset;
Windows run source manifests have not been verified. Historical download
revision and subset selection procedure are unknown. Keep source_subset in
code; distinguish declared origin from the byte/metadata matching performed.

Three target directories have been provided and stored at
data/aid_subsets/{clean,bright,lowres}, each with 180 images. The reconstructed
clean manifest matches the hash of all 12 clean runs; bright/lowres filenames
and labels match all prediction CSVs. Frozen manifests and ZIPs/restore script
are documented in DATA.md. Checkpoints, processing formulas, and original-image
mappings between the three sets are not available yet.

The target is AID with the same label space. I previously tested 20 images per
class = 180 images on Windows. Code does not require fixed or balanced counts,
but all classes must be present. Do not assume the current machine has AID,
weights, or outputs from the Windows machine.

Manifests contain path, label, class_index, fold, domain, and SHA-256. Hashes
detect identical file contents but do not guarantee detection of near-duplicates
or images of the same location. Metadata is unavailable to separately identify
geographic, seasonal, weather, or sensor effects.

3. MODELS

- small_cnn: four Conv-BatchNorm-ReLU-MaxPool blocks, channels 3→32→64→128→256,
  global average pooling, dropout 0.30, a 9-class classifier; random initialization.
- resnet18_scratch: torchvision ResNet18, weights=None, replace fc with 9 outputs.
- resnet18_pretrained: same architecture, ImageNet pretrained, new 9-class fc.
- deit_tiny_pretrained: timm deit_tiny_patch16_224.fb_in1k, pretrained, 9 classes.

The scratch/pretrained ResNet18 pair controls architecture to study pretraining.
Do not conclude that pretrained models are always better. Currently the whole
network is fine-tuned; backbone freezing, grouped learning rates, and two-stage
fine-tuning are not implemented. The project does not implement a distillation
loss/teacher simply because it uses the DeiT name.

4. ACTUAL PIPELINE AND CONFIGURATION

prepare_manifests.py → source.csv / aid_test.csv / summary.json
→ development on source training/validation to select epochs
→ recreate the model, train on the entire source for the selected epoch count
→ test AID after all models finish training
→ aggregate models/seeds and report per-class results.

The source has 5 folds. Default seeds 13,37,73 use validation folds 0,1,2
respectively; do not call this full 5-fold CV or variation solely from
initialization seeds. Default split seed: 20260926.

Input 224×224 RGB, ImageNet normalization. Training uses random crop, flips,
and rotation by multiples of 90°; evaluation uses Resize(256) + CenterCrop(224).
Optimizer AdamW, lr 3e-4, weight_decay 1e-4, CrossEntropyLoss, CosineAnnealingLR,
maximum 50 epochs, patience 8, batch_size 32, workers 4.
AMP is enabled only for CUDA when the flag is provided.

Development best.pt is the best validation checkpoint; final best.pt contains
last-epoch weights for the fixed epoch count. Do not select final checkpoints
using the target. Final refit does not continue from development weights;
scheduler T_max follows the run's epoch count. --skip-existing does not resume
the optimizer.

configs/experiment.json currently only describes the study; it is not loaded
as runtime configuration. CLI arguments/script defaults control execution.
Recheck this against the latest code before changing configuration.

5. ROBUSTNESS ADDED TO THE CODE

Two profiles: baseline (default) and weather_robust.
Four factors:
- cloud_haze: simulated cloud/haze overlay.
- illumination: brightness/contrast.
- resolution: reduced resolution/blur.
- sensor_noise: Gaussian noise on RGB.

Robust training: 50% no additional degradation, 50% select one factor at severity
1 or 2; basic augmentation remains. Original images are not modified.

Baseline selects epochs by clean source-validation macro-F1.
Robust selects by mean F1 across 5 source validation conditions: clean and four
severity-2 transformations, fixed seed 7919. AID is not used to select epochs
or hyperparameters. This changes both augmentation and selection criterion;
isolating augmentation effects requires an appropriate ablation.

scripts/evaluate_robustness.py tests one checkpoint under 13 conditions:
clean + 4 factors × 3 severities. Default corruption seed 2026; RNG depends on
image hash, condition, seed, and version optical-proxies-v1, independently of
the training seed. Severity 3 is not used in training/validation. Every model
must use the same target manifest and corruption seed.
180 images × 13 conditions still means only 180 independent images.

Exports robustness.csv, summary.json, robustness_curves.png, and metrics/
predictions/confusion matrices/per-class F1 for each condition. Drops are
percentage-point differences relative to clean AID, not a same-domain/
cross-domain gap.

Transformations are proxies only. There is no evidence about real storms,
real geography/seasons, real sensors, recovery of ground information under
complete cloud cover, or damage detection yet.

6. OLD RESULTS AND VERIFICATION STATUS

Mean accuracy from CSVs I previously provided, in CUDA / CPU order:
- SmallCNN: 68.33% / 68.33%.
- ResNet18 scratch: 71.11% / 77.22%.
- ResNet18 pretrained: 90.37% / 93.52%.
- DeiT-Tiny pretrained: 93.33% / 94.07%.

These are old results, not weather_robust results. Whether CPU runs retrained
or only tested CUDA checkpoints remains unclear, and identical manifests/seeds/
configuration have not been fully verified. Do not conclude that CPU is more
accurate than CUDA or combine the highest results from each environment into
one main experiment table.

The previous session ran 10 unit tests and checked transformations on a real
source image. The new PyTorch/Matplotlib pipeline has not been run end to end
on the Mac; new robust weights/results have not been confirmed. Passing unit
tests is not evidence of improved accuracy.

The 180 AID images were viewed while analyzing improvements. If continuing to
use them for model/method selection, treat them as an exploratory/development
target and obtain a new unseen target set for final results. Transforming old
images does not make them independent again.

7. ARTIFACTS AND METRICS

Actual per-run manifests, seed/configuration/device/library versions, checkpoint
hashes, and code provenance are recorded. Available metrics include accuracy,
balanced accuracy, macro precision/recall/F1, per-class metrics, confusion
matrices, ECE with 15 bins, training curves, and mean±SD tables across runs.

Mean±SD is not a confidence interval. Bootstrap CI, NLL/reliability diagrams,
and calibration fitting are not implemented yet. Softmax confidence does not
guarantee correctness probabilities under domain shift. The classifier has no
OOD detection/prediction rejection yet.

Aggregate CSVs alone cannot recover complete weights, manifests, or training
curves. If history.json/metrics.json remain available, use
python -m satdomain.reports to regenerate plots. Outputs/checkpoints are usually
not committed to Git.

8. FILES TO READ BEFORE CONTINUING

Read PROJECT_ANALYSIS_VI.md, README.md, research/methodology.md,
ROBUSTNESS_GUIDELINE.md, RESEARCH_ARTIFACTS_GUIDELINE.md, and WINDOWS_GUIDELINE.md.
Then read code relevant to the task:
- data/models: constants.py, data.py, models.py.
- training: train.py, scripts/run_study.py.
- evaluation: evaluate.py, metrics.py, scripts/aggregate_results.py.
- robustness: robustness.py, scripts/evaluate_robustness.py.
- artifacts/inference: artifacts.py, reports.py, infer.py, runtime.py.

The package is in src/satdomain/. Windows wrappers are in scripts/.
Preserve my uncommitted changes; do not reset/delete outputs/checkpoints.

9. ENVIRONMENT AND WORKING PROCEDURE

The Mac is used for code/documentation development; full training/testing runs
on my Windows machine. Do not install PyTorch or launch training on the Mac
without a new request. Follow the shared machine's AGENTS.md: limit workers,
no stress/load tests, no Docker/Colima on the Mac, do not exceed resource limits,
and clean up processes when finished.

Perform the task filled in at the start of this prompt. First check repository
status and relevant information; if files are inaccessible, say so rather than
pretending to have read them. Ask briefly about missing information that changes
the nature of the problem.

When changing code: explain changes, perform appropriate checks, and state
what remains unverified.
When analyzing results: associate every number with its checkpoint/manifest/
seed/environment; distinguish observations, hypotheses, and evidence-supported
conclusions.
When guiding self-study: identify files/functions to read, their importance,
and questions to understand.
```
