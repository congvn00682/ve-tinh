# Methodology: source subset -> AID cross-domain evaluation

## Research task

The study reports cross-domain results only. Models learn from the existing
source subset and are evaluated on the AID subset. There is no same-domain test
result and no same-domain/cross-domain gap in the final report.

The source folders match NWPU-RESISC45 naming and image dimensions, but the
repository does not contain provenance metadata. Reports must call it
`source_subset` until its origin is documented.

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
