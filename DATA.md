# Data and reproducibility

Updated on 04/10/2026. The task is RGB scene classification, **source → AID**,
with nine shared classes. There is no source test set or same-domain/domain-gap
reporting. This document describes the current code, source data, and three
provided target directories, and explicitly identifies missing historical
artifacts. Results are in [results/README.md](results/README.md).

## 1. Official URLs and data sources

| Role | Dataset | Official author URL | User-reported mirror |
|---|---|---|---|
| Training/development source | NWPU-RESISC45 | [Gong Cheng — Datasets](https://gcheng-nwpu.github.io/#Datasets) | [blanchon/RESISC45](https://huggingface.co/datasets/blanchon/RESISC45) |
| Cross-domain target | Aerial Image Dataset (AID) | [AID — Wuhan University](https://captain-whu.github.io/AID/) | [blanchon/AID](https://huggingface.co/datasets/blanchon/AID) |

Hugging Face is a mirror, not the authors' official website.
The NWPU page describes the full dataset of 45 classes and 31,500 images and
provides the original download link. The AID page describes the full dataset
of 30 classes and 10,000 images and provides the original download link.
The project uses a nine-class subset; full dataset counts are not experiment counts.

## 2. Dataset versions and image identity

Do not assign a release tag such as `v1.0` to an original dataset without evidence.
The year 2017 in the citation is the publication year, not a revision identifier
for downloaded files.

| Data | Recordable version/revision | Verification status |
|---|---|---|
| Source mirror | `blanchon/RESISC45@f8fb2c69b80203f0b9a622f9fdb7f6a6001a1676` | SHA-256 of all 900 local images matches the LFS SHA-256 of each file at this revision |
| Source subset in the repo | **`source-subset-v1`**, frozen on 04/10/2026 | Image list, hashes, class indices, and folds are stored in Git together with the ZIP |
| AID reference mirror | `blanchon/AID@d9b5531544523f29d365a77a007b2ace3cea6882` | Reference revision from the API; target image bytes have not been matched against the mirror |
| Three provided AID subsets | **`aid-subsets-v1`**, snapshots `aid-clean-v1`, `aid-bright-v1`, `aid-lowres-v1` | 540 files preserved byte for byte; three manifests and ZIPs have SHA-256 hashes |
| AID in the Windows results | Historical download revision/processing formula **unknown** | Reconstructed clean manifest matches the hash of all 12 runs; bright/lowres filenames and labels match every CSV, but the CSVs contain no image hashes |

The revisions were retrieved on 04/10/2026 through the APIs of the
[source mirror](https://huggingface.co/api/datasets/blanchon/RESISC45) and
[target mirror](https://huggingface.co/api/datasets/blanchon/AID).
Machine-readable metadata: [upstream_references.json](data/provenance/upstream_references.json)
and [source_subset_v1.json](data/provenance/source_subset_v1.json).
Target metadata: [aid_subsets_v1.json](data/provenance/aid_subsets_v1.json);
result comparison: [data_verification.json](results/cuda_aid_9class/data_verification.json).

Source matching uses LFS metadata; all 900 remote images were not downloaded
again individually during this audit. The download script below checks received
image bytes using SHA-256. The original download revision remains unknown;
matching content allows the exact current images to be reconstructed from the
pinned revision, but does not prove that historical Windows source manifests
are identical to the current manifest.

## 3. Subsets, label mapping, and data splits

Current source: `data/source_subset/`, 900 images, 100 images per class. The exact
list is in [source_subset_v1.csv](data/provenance/source_subset_v1.csv).
Keep the exact files in the manifest; do not randomly select 100 different
images from the full dataset. The original subset selection procedure was not
recorded. Snapshot v1 preserves image bytes without creating new labels or
exporting resized/normalized images.

The three user-provided directories were copied byte for byte, retaining the
originals in `.orca/drops` and excluding `.DS_Store` from the snapshot:

| Condition | Original directory | Snapshot in the repo | Manifest | Verified header properties |
|---|---|---|---|---|
| Clean | `aid_target` | `data/aid_subsets/clean/` | [aid_clean_v1.csv](data/provenance/aid_clean_v1.csv) | 180 images at 600×600: 162 JPEG, 18 PNG |
| Bright | `aid_bright copy` | `data/aid_subsets/bright/` | [aid_bright_v1.csv](data/provenance/aid_bright_v1.csv) | 180 JPEG, 600×600 |
| Lowres | `aid_lowres` | `data/aid_subsets/lowres/` | [aid_lowres_v1.csv](data/provenance/aid_lowres_v1.csv) | 180 JPEG, 300×300 |

Each set has 20 images per class. All files have the `.jpg` extension, including
the 18 PNGs in clean; do not rename or re-encode them because this would change
historical hashes/manifests. Pillow detects formats from the contents, and the
loader then converts to RGB. A header audit does not replace full pixel decoding
checks in the ML environment.

Image names within each class: clean `_21`–`_40`, bright `_41`–`_60`, lowres `_61`–`_80`.
Filename and SHA-256 lists do not overlap between conditions; no original-image
mapping is available to establish whether these are paired views or independent
scenes. There are no duplicate bytes between the 900 current source images and
the 540 target files.

| Class index | Canonical label / source directory | Mapped AID directory name |
|---:|---|---|
| 0 | airport | Airport |
| 1 | baseball_diamond | BaseballField |
| 2 | beach | Beach |
| 3 | bridge | Bridge |
| 4 | church | Church |
| 5 | commercial_area | Commercial |
| 6 | dense_residential | DenseResidential |
| 7 | desert | Desert |
| 8 | forest | Forest |

The order is defined by [constants.py](src/satdomain/constants.py); directory
aliases are in [prepare_manifests.py](scripts/prepare_manifests.py).
The airplane and basketball_court classes are not used.

Data split protocol in the current code:

| Stage | Data | Image count with the current source |
|---|---|---:|
| Development training | Four source folds | 720 = 80 images per class |
| Development validation | One source fold | 180 = 20 images per class |
| Final refit | Entire source; recreate the model for the epoch count selected on the source | 900 |
| Cross-domain test | Nine AID classes; actual count recorded in the target manifest | Provided runs: 180 = 20 images per class |

Split seed **20260926**, five folds, 180 images per fold. Within each class, the
script sorts paths, shuffles them using `random.Random(f"{seed}:source:{label}")`,
and assigns `fold = shuffled_index % 5`. This algorithm was checked to reproduce
all folds in the frozen manifest exactly. Use the manifest to preserve the split
even when the environment changes.

`run_study.py` defaults: training seeds **13/37/73** use validation folds **0/1/2**
according to seed position in the list. This is not full 5-fold CV; between-run
variation includes both the seed and validation split. Historical run metadata
is not available to verify that the supplied Windows results used these exact
defaults. AID must not be used to select epochs, hyperparameters, or normalization.

## 4. Preprocessing procedure

Implemented in [data.py](src/satdomain/data.py), `ManifestDataset.__getitem__()`
and `build_transforms()`. This is the current code pipeline; profiles/library
versions for historical CSV runs have not been fully confirmed.

### Baseline training

1. Read the file with Pillow and convert it to three-channel RGB.
2. `Resize(256)`: set the shorter edge to 256 while preserving the aspect ratio.
3. `RandomResizedCrop(224, scale=(0.80, 1.0))`.
4. Random horizontal flip, random vertical flip.
5. Randomly rotate by one of 0°, 90°, 180°, 270°.
6. `ToTensor()`: convert RGB uint8 to a floating-point tensor using torchvision's rules.
7. Normalize using ImageNet mean **(0.485, 0.456, 0.406)**,
   std **(0.229, 0.224, 0.225)**, applying `(x - mean) / std` to each channel.

The default model input is `[batch, 3, 224, 224]`.
Properties not explicitly specified, such as resize interpolation/antialiasing
and crop aspect ratio, use the installed torchvision defaults. Detailed
reproduction requires retaining library versions recorded in `run.json`; do not
assume versions for old runs. Normalization is not estimated from the target.

### Clean validation/test and inference

RGB → `Resize(256)` → `CenterCrop(224)` → `ToTensor()` → the same ImageNet normalization.
No random crop, flip, or rotation is applied. Evaluation/inference must match
the checkpoint's preprocessing; arbitrary normalization metadata is not yet
fully restored automatically by the loader.

### Weather robustness in the current code

[robustness.py](src/satdomain/robustness.py) defines version **`optical-proxies-v1`**.
In robust training, after crop/flip/rotation and before tensor conversion, there
is a 50% probability of no additional degradation; in the remaining 50%, one of the four
corruptions and severity 1 or 2 is selected uniformly.
Original image files are not modified, and no extra training images are exported to disk.

| Corruption | Severity 1 / 2 / 3 in the code |
|---|---|
| cloud_haze | Blend RGB with 245; opacity = base (0.10/0.20/0.30) + amplitude (0.25/0.40/0.55) × smooth texture |
| illumination | Brightness factor for the bright branch 1.15/1.35/1.60 or dark branch 0.80/0.60/0.40; contrast factor 0.95/0.85/0.75 |
| resolution | Downsample by divisor 2/4/8 using BOX, resize back using BILINEAR; Gaussian blur radius = `min(image.size)/224 × (0.3/0.6/1.0)` |
| sensor_noise | Gaussian RGB noise, sigma 4/10/20 on the 0–255 pixel scale |

Robust validation includes clean and four severity-2 corruptions, seed **7919**.
Validation/test transformations occur on the original RGB image **before**
Resize/CenterCrop. Baseline selects epochs by clean source macro-F1; robust
selects them by the mean F1 across five conditions.

Robustness evaluation: clean + four corruptions × three severities = 13 conditions,
default corruption seed **2026**. RNG depends on `version|seed|image_sha256|condition`,
independently of the training seed and data order. Severity 3 is not used in
training/validation. Preserve image hashes, corruption seed, and code version
when comparing models; transformed versions of one image are not new independent samples.

These proxies have not been confirmed as the procedure used to create the
bright/lowres images in the submitted CSVs. “Brightness +30, contrast +100”
cannot be converted into a fixed formula without knowing the tool, parameter
scale, and processing order. The blur kernel/sigma is also missing.
There is no script to reproduce the historical **transformation procedure**
from original images. However, the ZIPs and restore script below recover the
exact **processed image bytes** provided by the user, so reconstructing the
inputs does not require guessing brightness/blur parameters.

## 5. Downloadable dataset snapshots

**[Download source_subset_v1.zip](data/releases/source_subset_v1.zip)** — 900 images
with manifest, metadata, and NOTICE; size **12,483,887 bytes**.
[Checksum file](data/releases/source_subset_v1.zip.sha256):

```text
e84aeda4a598357e1cba008c2267710153e875e7ae68cbd5b4a2894addd44d23
```

This is a real file in the current repository; the relative link follows the
repository when committed/published. The archive has not been published to a
separate public URL; do not treat a GitHub URL for an unpushed file as an active
download link.

The official NWPU page specifies CC BY-NC 4.0; the snapshot retains attribution
and the terms URL in NOTICE/metadata. This is a subset of the original dataset,
not imagery captured by the group.

The processed test sets have download links in the repo. Each archive contains
180 byte-preserved images, a `target.csv` manifest, metadata, and an AID attribution NOTICE:

| Set | Download link | Size in bytes | Checksum |
|---|---|---:|---|
| Clean | [aid_clean_v1.zip](data/releases/aid_clean_v1.zip) | 41,378,232 | [SHA-256](data/releases/aid_clean_v1.zip.sha256) |
| Bright | [aid_bright_v1.zip](data/releases/aid_bright_v1.zip) | 33,314,744 | [SHA-256](data/releases/aid_bright_v1.zip.sha256) |
| Lowres | [aid_lowres_v1.zip](data/releases/aid_lowres_v1.zip) | 5,943,750 | [SHA-256](data/releases/aid_lowres_v1.zip.sha256) |

These ZIPs exist in the current checkout and have not been published to public URLs.
AID images were provided by the user; the snapshot does not assign them a new
license. ZIPs use the received image bytes rather than recreating an unconfirmed
transformation formula.

## 6. Scripts required to reproduce the data

Run commands from the repository root. `reproduce_source_data.py`,
`reproduce_target_data.py`, and `prepare_manifests.py` only require Python >=3.10,
not PyTorch. On Windows, use an installed Python or
`.\.venv\Scripts\python.exe`.
Do not run training on the Mac to check the data.

### A. Verify the existing source snapshot

```bash
python scripts/reproduce_source_data.py verify
python scripts/reproduce_source_data.py verify-archive
```

The script checks manifest hashes, class mapping, each image's SHA-256, domain,
duplicate content, and archive structure. Hashes do not detect near-duplicates
or images of the same location. The script does not decode images to visually
confirm their contents/labels.

### B. Reconstruct the source from ZIP without Internet access

```bash
python scripts/reproduce_source_data.py restore --root outputs/data_reproduction/source_subset
python scripts/reproduce_source_data.py verify --root outputs/data_reproduction/source_subset
```

Checks the checksum and every image before restoration; does not overwrite files
with different contents. The archive created by `package` contains
`source_subset_v1/images/<class>/<file>`; restore places images in
`<root>/<class>/<file>`.

### C. Reconstruct the source directly from the pinned mirror

```bash
python scripts/reproduce_source_data.py download --root outputs/data_reproduction/source_subset --workers 2
```

Only downloads manifest-listed images from `/resolve/<revision>/data/<class>/<file>`;
does not download all 31,500 images or use the moving revision `main`. SHA-256
must match before saving. Existing files with matching hashes are skipped;
files with different contents stop the script and are preserved. Workers are
limited to 1–4. If the network fails, rerun the same command to download remaining
files. Neither `datasets` nor `huggingface_hub` is required.

### D. Regenerate manifests and the source split

```bash
python scripts/prepare_manifests.py --source-root data/source_subset --output-dir outputs/data_reproduction/manifests --seed 20260926 --folds 5 --expected-source-per-class 100
```

For the current snapshot, all 900 rows, label order, hashes, and folds must match
the frozen manifest. `summary.json` contains the current machine's absolute
roots/arguments, so its hash need not be identical on two machines. Do not
modify the manifest of an old study.

When the **exact AID directory used** is available with all nine classes, add the
target and audit overlap:

```bash
python scripts/prepare_manifests.py --source-root data/source_subset --aid-root data/aid_subsets/clean --output-dir outputs/data_reproduction/manifests_with_aid --seed 20260926 --folds 5 --expected-source-per-class 100 --expected-target-per-class 20
```

`--expected-target-per-class 20` checks counts; it does not select 20 images from
the full dataset. Preserve the exact subset before generating a manifest.
The target has `fold=-1` and is only used for evaluation. To reproduce the
provided clean runs, the target manifest must match the hash recorded in
`results/cuda_aid_9class/raw/clean/cross_domain_runs.csv`:

```text
c71a5411bf4bd3ce22f53ec1d6594180279aaaf76e3bff776fa9b3c55fa8c599
```

Selecting 20 new images per class creates a new target subset; do not call it a
reproduction of an old run. Hashing only the CSV file is insufficient: each
image's content hash must also be checked.
This comparison was performed: the reconstructed clean manifest matches the
hash above, and all 180 files match their manifest SHA-256 hashes. For
bright/lowres, replace `--aid-root` with the corresponding snapshot and use a
different output directory; do not overwrite the clean manifest.

### E. Recreate the source archive

```bash
python scripts/reproduce_source_data.py package --archive outputs/data_reproduction/source_subset_v1.zip
```

Refuses to overwrite existing archives/checksums. The ZIP uses a fixed entry
order and timestamps and stores image bytes unchanged; it does not depend on
zlib compression.

### F. Restore the three provided AID sets exactly

```bash
python scripts/reproduce_target_data.py verify
python scripts/reproduce_target_data.py verify-archive
python scripts/reproduce_target_data.py restore --root outputs/data_reproduction/aid_subsets
python scripts/reproduce_target_data.py verify --root outputs/data_reproduction/aid_subsets
```

By default, all three conditions are verified/restored. Add `--condition clean`,
`bright`, or `lowres` to select one set. The script checks checksums, manifest
hashes, every image, labels, domain `target`, and fold `-1`; it does not overwrite
images with different contents. It preserves the bytes of PNGs carrying the
`.jpg` extension too. `restore` uses the provided ZIPs and needs neither
Internet access nor historical brightness/blur code. Recreate the archives in
a new directory using:

```bash
python scripts/reproduce_target_data.py package --archive-dir outputs/data_reproduction/releases
```

### G. Generate views and run new experiments

Once dependencies, checkpoints, and a valid target manifest are available, use
[run_study.py](scripts/run_study.py) and [evaluate_robustness.py](scripts/evaluate_robustness.py)
following [README](README.md) and [ROBUSTNESS_GUIDELINE](ROBUSTNESS_GUIDELINE.md).
`optical-proxies-v1` views are generated in the loader; no separate corruption
dataset download is required. They do not recover the bright/lowres dataset
used in old runs. [summarize_submitted_results.py](scripts/summarize_submitted_results.py)
only recalculates tables from saved predictions; it does not reconstruct images.

## 7. Missing components for full reproduction of historical results

| Component | Available | Still needed |
|---|---|---|
| Current source | 900 images, frozen manifest, matched revision, ZIP, verify/download/restore script | Actual Windows run source manifest to confirm the same subset/split |
| Clean AID | 180 images, reconstructed manifest matching the hash of 12 runs, snapshot/ZIP and restore script | Original download revision/subset selection procedure and checkpoint for reevaluation |
| Bright/lowres AID | 180 images per set, manifests/hashes, ZIPs and restore script; filenames/labels match prediction CSVs | Processing tool/code/formula, original-image mapping, and run metadata/image hashes to verify historical use |
| Data split/run | Current code protocol/defaults, seeds in CSVs | `run.json`, `study.json`, and actual manifests to verify folds/configuration |

Bright and lowres have different filename lists; whether they were renamed or
come from different original images is unknown. Do not pair them with clean to
compute a corruption drop until the same images and checkpoint are verified.
These missing items cannot be replaced by choosing other images from the
original URL or assigning a current revision to a past run.
