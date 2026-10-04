# Methodology: source subset -> AID cross-domain evaluation

## Research task

The study reports cross-domain results only. Models learn from the existing
source subset and are evaluated on the AID subset. There is no same-domain test
result and no same-domain/cross-domain gap in the final report.

On 2026-10-04, the user identified the source as a NWPU-RESISC45 subset from
[blanchon/RESISC45](https://huggingface.co/datasets/blanchon/RESISC45) and the
target as an AID subset from
[blanchon/AID](https://huggingface.co/datasets/blanchon/AID).
The dataset pages have been inspected. The 900 local source image SHA-256 values
match Hugging Face LFS metadata at the RESISC45 revision pinned in [DATA.md](../DATA.md).
The reconstructed clean target manifest matches the SHA-256 reported by all
12 clean Windows runs; the supplied bright/lowres filename-label sets match
their prediction CSVs, which contain no image hashes. Historical source
manifests and checkpoints have not been verified; the original download
revision and processing procedure are not recorded. Supplied target image
snapshots, manifests and ZIP archives are documented in DATA.md.
Keep `source_subset` as the internal identifier. Reports may describe it as a
NWPU-RESISC45 subset with user-reported origin, a content-matched current source
snapshot, and these historical verification limits.

Both domains use the same nine-class label space:

1. airport
2. baseball_diamond
3. beach
4. bridge
5. church
6. commercial_area
7. dense_residential
8. desert
9. forest

The source images are stored in `data/source_subset/`, with one subdirectory per class.
The current source contains 100 images per class. Target class counts are not
fixed and the tooling does not require a specific count. Actual counts must be
recorded in the dataset summary.

## Source-only model development

The source manifest distributes each class as evenly as possible over five
deterministic folds. For each seed, one fold (approximately 20%) is used only
for validation and the other four folds (approximately 80%) are used for
training.

Validation macro-F1 controls early stopping and selects the epoch count. Source
validation results are diagnostics, not research test results, and are not
included in the model comparison table.

After selecting the epoch count, the model is reinitialized and fitted using
all available source images. All models and seeds are fully trained before
any AID result is opened.

## Cross-domain test

Every final model is evaluated on every available AID test image. AID images must
never be used for training, early stopping, hyperparameter selection,
normalization-statistic estimation, threshold selection, or augmentation
design.

The final comparison reports only AID metrics. Looking at AID results and then
changing training choices invalidates the locked-test protocol.

## Models

- `small_cnn`: compact CNN initialized from scratch.
- `resnet18_scratch`: ResNet-18 initialized from scratch.
- `resnet18_pretrained`: the same ResNet-18 architecture with ImageNet weights.
- `deit_tiny_pretrained`: a small pretrained transformer.

The scratch/pretrained ResNet-18 pair controls for architecture when measuring
the effect of pretraining.

## Reported metrics

For AID only, report accuracy, balanced accuracy, macro precision, macro recall,
macro F1,
per-class precision/recall/F1, row-normalized confusion matrices, and expected
calibration error. Aggregate the final metrics as mean and standard deviation
over random seeds.

Raw softmax scores are confidence values and must not be described as calibrated
probabilities.

## Reproducibility

Manifests store relative path, canonical class, class index, validation fold,
domain, and SHA-256 digest. Checkpoints store weights, class order,
preprocessing metadata, seed, and training arguments. Generated manifests,
checkpoints, plots, and predictions are not committed to Git by default.

## Optional robustness experiment

The `weather_robust` profile uses synthetic cloud/haze, illumination, resolution
loss and RGB noise on source training data. It selects epochs by mean macro-F1
across clean and four deterministically degraded source-validation views, then
refits on all source data. Baseline selection remains clean validation macro-F1.
Compare these as two pipelines; isolating augmentation requires an additional
ablation with the same selection criterion. AID remains excluded from selection.

Evaluate fixed checkpoints on clean AID and each degradation at three severity
levels. Report paired clean-to-degraded drops, not a same-domain/cross-domain
gap. Transformed views are not additional independent test samples. This protocol
does not establish real cyclone detection, damage assessment, geographic or
seasonal generalization. See [ROBUSTNESS_GUIDELINE.md](../ROBUSTNESS_GUIDELINE.md)
for commands, limitations and the need for unseen target data after exploratory
experiments on the previously inspected AID subset.
