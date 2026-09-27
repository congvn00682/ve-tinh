**Phân tích dự án Satellite Image Classification across Different Domains**

Cập nhật theo code và lịch sử trao đổi ngày **27/09/2026**. Đây là bản hướng dẫn
tự học và bàn giao; khi code thay đổi, cần đối chiếu lại. Xem prompt dùng lại tại
[REUSABLE_PROJECT_PROMPT.md](REUSABLE_PROJECT_PROMPT.md).

**1. Dự án đang giải quyết bài toán gì? — RẤT QUAN TRỌNG**

Đầu vào là một ảnh RGB chụp từ trên cao. Đầu ra là một trong chín nhãn cảnh:
sân bay, sân bóng chày, bãi biển, cầu, nhà thờ, khu thương mại, khu dân cư dày đặc,
sa mạc hoặc rừng. Nhãn mô tả **toàn cảnh của ảnh**. Model không đánh dấu vị trí
từng vật thể và không phân loại từng pixel.

Trọng tâm nghiên cứu là **khả năng tổng quát hóa khi đổi nguồn dữ liệu**:
train trên source hiện có, test trên AID. Theo phạm vi đã chốt, báo cáo chính
chỉ có cross-domain, không có same-domain test và không đo same-domain/cross-domain gap.

Phần mở rộng `weather_robust` thử giữ khả năng phân loại cảnh khi ảnh bị mây/sương,
thay đổi ánh sáng, mất độ nét hoặc nhiễu. Đây chưa phải model nhận diện bão,
phân loại thời tiết hoặc đánh giá thiệt hại sau thiên tai.

| Khái niệm | Ý nghĩa trong dự án |
|---|---|
| Class / lớp | Nội dung cần dự đoán, ví dụ `forest` |
| Domain / miền | Nguồn hoặc phân phối dữ liệu, ví dụ source và AID |
| Architecture | Cấu trúc mạng, ví dụ ResNet18 |
| Pretraining | Trọng số học từ dữ liệu khác trước khi học chín lớp này |
| Checkpoint | Trọng số cùng thông tin để dựng lại model của một run |
| Robustness | Mức ổn định khi ảnh đầu vào bị biến đổi |
| Generalization | Khả năng hoạt động trên dữ liệu chưa dùng để học/chọn model |

Một ảnh vẫn mang nhãn `forest` sau biến đổi độ sáng, miễn là cảnh còn có ý nghĩa.
Không thêm nhãn `storm` vào cùng danh sách chín lớp chỉ vì muốn nghiên cứu thời tiết.

**2. Dữ liệu thực tế và những gì còn chưa biết — RẤT QUAN TRỌNG**

Đã kiểm kê lại: chín folder source ở gốc repo, mỗi folder 100 ảnh, tổng 900 ảnh.
Hai lớp `airplane` và `basketball_court` đã bị loại theo yêu cầu trước đó.

| Chỉ số nhãn | Folder source | Tên tương ứng thường gặp trong AID |
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

Thứ tự này nằm trong [constants.py](src/satdomain/constants.py). Đổi thứ tự mà
không cập nhật checkpoint/manifest sẽ làm sai ý nghĩa đầu ra, kể cả khi code chạy được.

Source giống một subset NWPU-RESISC45 về tên lớp và đặc điểm đã kiểm kê, nhưng
chưa có tài liệu xác nhận nguồn. Trong báo cáo vẫn gọi là `source_subset` cho
đến khi bổ sung provenance. Chưa có metadata địa lý, ngày chụp, mùa hoặc sensor.

Theo thông tin bạn cung cấp, AID đã test trên Windows có **20 ảnh/lớp = 180 ảnh**.
Code cho phép số ảnh khác hoặc lệch giữa các lớp, nhưng yêu cầu đủ chín lớp.
Không mặc định rằng máy Mac này có toàn bộ AID, checkpoint hoặc output từ Windows.

**3. Bản đồ thư mục toàn dự án**

```text
ve-tinh/
├── airport/ ... forest/          9 folder ảnh source, 100 ảnh/folder
├── src/satdomain/                Package Python thực hiện nghiên cứu
│   ├── __init__.py               Khởi tạo package
│   ├── constants.py              Nhãn và thông số normalization
│   ├── data.py                   Đọc ảnh, transforms, source split
│   ├── models.py                 SmallCNN và factory tạo 4 model
│   ├── train.py                  Train development/final, chọn và lưu weights
│   ├── evaluate.py               Đánh giá một checkpoint trên manifest
│   ├── metrics.py                Accuracy, F1, confusion matrix, ECE
│   ├── infer.py                  Dự đoán top-k cho một ảnh
│   ├── robustness.py             Bốn phép suy giảm mô phỏng
│   ├── artifacts.py              Hash, provenance, snapshot, complete marker
│   ├── reports.py                Xuất bảng và hình từ kết quả đã lưu
│   └── runtime.py                Seed, chọn device, ghi JSON
├── scripts/
│   ├── setup_windows.ps1         Tạo venv, cài dependencies trên Windows
│   ├── run_windows.ps1           Wrapper chuẩn bị data, train/test/tổng hợp
│   ├── prepare_manifests.py      Kiểm kê, ánh xạ nhãn, tạo fold và manifest
│   ├── run_study.py              Điều phối development → final → AID
│   ├── aggregate_results.py      Tổng hợp model/seed và từng lớp
│   └── evaluate_robustness.py     Test 13 điều kiện cho một checkpoint
├── configs/experiment.json       Mô tả cấu hình nghiên cứu, chưa là config runtime
├── research/methodology.md       Protocol và phạm vi kết luận
├── tests/
│   ├── test_research_artifacts.py Kiểm tra integrity và xuất bảng
│   └── test_robustness.py         Kiểm tra biến đổi và phép so sánh
├── data/manifests/               CSV/JSON sinh ra khi chuẩn bị dữ liệu
├── outputs/ hoặc đường dẫn khác  Output trên máy chạy; không mặc định đã tồn tại
├── pyproject.toml                Dependencies, package và CLI entry points
├── .gitignore                    Loại weights/output/cache khỏi Git
├── .orca/                        File đính kèm trao đổi; không phải nguồn train
└── *.md                          Tài liệu sử dụng và phân tích
```

`data/manifests/`, `outputs/`, `runs/` và checkpoint `.pt/.pth/.ckpt` bị Git bỏ qua
theo `.gitignore`. Vì vậy clone/pull code không đồng nghĩa đã tải weights hoặc kết quả.
`.git/` lưu lịch sử mã nguồn; `.venv/` nếu có là môi trường Python, không phải dữ liệu nghiên cứu.

| Tài liệu | Đọc khi nào |
|---|---|
| [README.md](README.md) | Muốn nhìn toàn bộ thao tác chính |
| [WINDOWS_GUIDELINE.md](WINDOWS_GUIDELINE.md) | Cài và chạy trên Windows CPU/CUDA |
| [RUN_ON_ANOTHER_MACHINE_GUIDELINE.md](RUN_ON_ANOTHER_MACHINE_GUIDELINE.md) | Chuyển dự án sang máy khác |
| [RESEARCH_ARTIFACTS_GUIDELINE.md](RESEARCH_ARTIFACTS_GUIDELINE.md) | Tìm manifest, checkpoint, F1 và biểu đồ |
| [ROBUSTNESS_GUIDELINE.md](ROBUSTNESS_GUIDELINE.md) | Chạy baseline/robust và đọc giới hạn mô phỏng |
| [research/methodology.md](research/methodology.md) | Viết phương pháp và xác định kết luận hợp lệ |

**4. Luồng thực thi từ ảnh đến kết quả — RẤT QUAN TRỌNG**

```text
Source folders + AID folders
        ↓ prepare_manifests.py
source.csv + aid_test.csv + summary.json
        ↓ run_study.py — giai đoạn 1
Source train/validation → chọn epoch → development/.../best.pt
        ↓ giai đoạn 2: tạo lại model
Toàn bộ source → train với số epoch đã chọn → final/.../best.pt
        ↓ giai đoạn 3, sau khi mọi model đã train xong
AID → evaluate.py → predictions + metrics + confusion matrix
        ↓ aggregate_results.py
Bảng so sánh model/seed và F1 từng lớp
```

`run_windows.ps1` gọi các script tương ứng và tự tổng hợp kết quả. Nếu chạy
`run_study.py` trực tiếp thì cần gọi `aggregate_results.py` riêng.

Nhánh bổ sung: `evaluate_robustness.py` lấy một final checkpoint, test ảnh AID sạch
và các bản biến đổi. Nhánh sử dụng thực tế: `infer.py` lấy checkpoint và một ảnh
mới, xuất top-k nhãn/confidence.

**5. Manifest và chia dữ liệu — RẤT QUAN TRỌNG**

Đọc [prepare_manifests.py](scripts/prepare_manifests.py), nhất là `collect_domain()`.
Manifest là danh sách đóng băng ảnh được dùng, gồm:

| Cột | Ý nghĩa |
|---|---|
| `path` | Đường dẫn tương đối tính từ data root |
| `label` | Tên lớp chuẩn |
| `class_index` | Số nguyên 0–8, thống nhất với model |
| `fold` | Fold source hoặc `-1` đối với target |
| `domain` | `source` / `target` |
| `sha256` | Dấu vân tay nội dung file |

Script ánh xạ tên folder AID sang tên chuẩn, kiểm tra số lượng, phát hiện trùng
nội dung file trong mỗi domain và giữa source/target. Tuy nhiên, **hash không phát
hiện ảnh gần trùng**, ảnh cùng địa điểm chụp khác lúc, hoặc cùng ảnh được nén lại.
Script chuẩn bị manifest không giải mã toàn bộ ảnh để xác minh nhãn/nội dung ảnh.

Source được phân tầng theo lớp vào năm fold bằng split seed mặc định `20260926`.
Với 100 ảnh/lớp, mỗi fold có 20 ảnh/lớp. Một run development giữ một fold để
validation: 720 ảnh train, 180 ảnh validation.

Điểm cần hiểu đúng: mặc định seed training `[13, 37, 73]` lần lượt dùng fold
`0, 1, 2`. Đây **không phải full 5-fold cross-validation**. Dao động giữa các run
bao gồm cả ảnh hưởng của khởi tạo và thay đổi validation split. Khi so baseline
và robust, giữ nguyên cặp seed/fold. Muốn nghiên cứu riêng ảnh hưởng của seed,
cần thiết kế thí nghiệm cố định fold.

**6. Preprocessing và DataLoader — RẤT QUAN TRỌNG**

Đọc [data.py](src/satdomain/data.py): `ManifestDataset`, `build_transforms()` và
`source_train_validation_split()`.

| Giai đoạn | Xử lý ảnh |
|---|---|
| Train baseline | RGB → Resize cạnh ngắn 256 → RandomResizedCrop 224, scale 0,8–1,0 → flip ngang/dọc → xoay bội 90° → tensor → normalization |
| Train robust | Như baseline, thêm `RandomDegradation` sau crop/flip/rotate và trước tensor |
| Validation/test sạch | RGB → Resize cạnh ngắn 256 → CenterCrop 224 → tensor → normalization |
| Validation/test suy giảm | Biến đổi ảnh RGB gốc trước Resize/CenterCrop, sau đó theo pipeline test |

Tensor đưa vào mạng có dạng `[batch, 3, 224, 224]`. Cả bốn model dùng mean/std
ImageNet khai báo trong `constants.py`; không ước lượng normalization từ AID.

Crop có thể loại bỏ nội dung sát mép; ảnh hưởng này cần nghiên cứu nếu định thay
preprocessing. Giữ cùng quy tắc khi so sánh model. Augmentation tạo mẫu biến đổi
trong lúc nạp dữ liệu, không sửa file ảnh gốc và không tạo thêm mẫu độc lập.

`DataLoader` gom ảnh thành batch và dùng worker để nạp ảnh. Mặc định batch 32,
workers 4. Batch size, worker count và GPU memory là các khái niệm khác nhau.

**7. Bốn model và vai trò của từng model — RẤT QUAN TRỌNG**

Đọc [models.py](src/satdomain/models.py). Model chỉ tạo logits: chín điểm số chưa
chuẩn hóa. Softmax biến logits thành confidence ở bước đánh giá/inference.

| Tên CLI | Cách khởi tạo | Vai trò nghiên cứu |
|---|---|---|
| `small_cnn` | Ngẫu nhiên | Baseline CNN tự xây, học từ source |
| `resnet18_scratch` | Ngẫu nhiên | CNN tiêu chuẩn để làm đối chứng |
| `resnet18_pretrained` | ImageNet weights qua torchvision | Đo ích lợi pretraining trên cùng kiến trúc ResNet18 |
| `deit_tiny_pretrained` | Pretrained qua timm | Đại diện transformer nhỏ |

**SmallCNN: hiểu từng khối trước khi học mạng phức tạp.**

```text
RGB 3 kênh
 → Conv 3→32  + BatchNorm + ReLU + MaxPool
 → Conv 32→64 + BatchNorm + ReLU + MaxPool
 → Conv 64→128 + BatchNorm + ReLU + MaxPool
 → Conv 128→256 + BatchNorm + ReLU + MaxPool
 → AdaptiveAvgPool 1×1 → Flatten → Dropout 0,30 → Linear 256→9
```

Conv dùng kernel 3×3, padding 1, không bias. Với ảnh 224×224, bốn lần pooling
giảm kích thước không gian còn 14×14 trước global pooling. Conv học đặc trưng;
pooling thu gọn không gian; classifier đổi đặc trưng thành điểm số chín lớp.
Độ đơn giản không đảm bảo generalization tốt hay xấu: phải dựa vào kết quả.

**ResNet18 scratch và pretrained: cặp đối chứng quan trọng nhất.**

Cả hai gọi `torchvision.models.resnet18()` và thay `fc` bằng Linear có chín đầu ra.
Khác biệt chính là scratch dùng `weights=None`, pretrained dùng
`ResNet18_Weights.DEFAULT`. Residual connections là ý tưởng nền tảng của ResNet;
đọc [bài báo ResNet](https://arxiv.org/abs/1512.03385) để hiểu vì sao mạng học phần
hiệu chỉnh qua đường nối tắt.

Trong code hiện tại, toàn bộ model pretrained được fine-tune ngay từ đầu;
**chưa khóa backbone, chưa fine-tune hai giai đoạn, chưa chia learning rate theo
backbone/classifier**. Khuyến nghị trước đây về các kỹ thuật đó chưa phải tính
năng đã triển khai. Đọc thêm sự khác nhau giữa fine-tuning toàn mạng và dùng
backbone làm feature extractor tại [hướng dẫn PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html).

**DeiT-Tiny: đại diện transformer.**

Factory gọi `timm.create_model("deit_tiny_patch16_224.fb_in1k", pretrained=..., num_classes=9)`.
Tên cấu hình thể hiện patch 16 và input 224. Cần học patch embedding, attention
và classifier. [Bài báo DeiT](https://arxiv.org/abs/2012.12877) trình bày hướng
huấn luyện transformer tiết kiệm dữ liệu và distillation; **project này không
cài teacher/student hoặc distillation loss**, mà fine-tune model có sẵn.

So SmallCNN với DeiT thay đổi đồng thời kiến trúc, năng lực biểu diễn và pretraining.
Không dùng riêng cặp đó để quy toàn bộ chênh lệch cho pretraining. Cũng không mặc
định pretrained/transformer luôn thắng trên mọi domain.

**8. Training và chọn checkpoint — RẤT QUAN TRỌNG**

Đọc [train.py](src/satdomain/train.py): `run_epoch()`, `checkpoint_payload()` và `main()`.

Một batch training: nạp ảnh → forward → CrossEntropyLoss → backward → optimizer
update. CrossEntropyLoss nhận logits trực tiếp; không thêm softmax trước loss.
Khi validation/test, `model.eval()` và no-grad tắt cập nhật trọng số, đồng thời
thay hành vi của Dropout/BatchNorm sang chế độ đánh giá.

| Tham số mặc định | Giá trị/code hiện tại |
|---|---|
| Loss | CrossEntropyLoss |
| Optimizer | AdamW cho toàn bộ tham số |
| Learning rate | `3e-4` |
| Weight decay | `1e-4` |
| Scheduler | CosineAnnealingLR, `T_max = số epoch của run` |
| Development tối đa | 50 epoch |
| Patience | 8 epoch không tăng điểm chọn |
| Batch / image size | 32 / 224 |
| Training seeds mặc định | 13, 37, 73 |
| AMP | Chỉ bật khi có cờ và device là CUDA |

Giai đoạn development chọn epoch bằng validation. Giai đoạn final **tạo lại model**
và train toàn bộ source theo số epoch đã chọn; không tiếp tục từ weights development.
Với pretrained, tạo lại model nghĩa là quay về pretrained initialization và head
mới; với scratch là khởi tạo ngẫu nhiên theo seed.

`best.pt` có hai ý nghĩa cần phân biệt:

- Development: weights tại epoch có điểm validation tốt nhất.
- Final: weights ở epoch cuối của số epoch đã khóa; file được ghi lại mỗi epoch.
  Không có AID validation để chọn “best target checkpoint”.

Scheduler final dùng `T_max` mới bằng số epoch final, nên đường learning rate
không nhất thiết giống đoạn đầu của development. Đây là lựa chọn refit hiện tại;
nếu nghiên cứu scheduler, cần lưu ý sự khác biệt này.

Một profile mặc định: 4 model × 3 seed = 12 run development + 12 run final,
tức **24 lượt train và 12 final checkpoint**. Chạy baseline và robust riêng
thành hai study; không lấy kết quả tốt nhất từng model từ hai môi trường khác nhau.

**9. Robustness mới hoạt động thế nào? — RẤT QUAN TRỌNG**

Đọc [robustness.py](src/satdomain/robustness.py), sau đó đọc nhánh robust trong
`train.py` và [evaluate_robustness.py](scripts/evaluate_robustness.py).

| Mã điều kiện | Phép biến đổi | Không thể suy ra trực tiếp |
|---|---|---|
| `cloud_haze` | Lớp phủ sáng có biến thiên không gian | Có bão, loại bão, lượng mưa hoặc tỷ lệ mây thật |
| `illumination` | Đổi độ sáng/tương phản | Mùa hoặc giờ chụp |
| `resolution` | Downsample, phóng lại và blur | Sensor hoặc GSD thực tế |
| `sensor_noise` | Thêm Gaussian noise trên RGB | Danh tính sensor hay nhiễu radar SAR |

`RandomDegradation`: xác suất 50% không thêm suy giảm; 50% chọn đều một trong
bốn loại, mức 1 hoặc 2. “Ảnh sạch” ở đây vẫn có crop/flip/rotate cơ bản.

Baseline chọn epoch bằng clean source-validation macro-F1. Robust chọn bằng:

```text
selection_score = (F1 sạch + F1 mây/sương + F1 ánh sáng
                   + F1 độ phân giải + F1 nhiễu) / 5
```

Bốn phép biến đổi validation dùng mức 2 và seed 7919, cố định qua các epoch.
Clean F1 và từng F1 biến đổi vẫn được lưu riêng. Mỗi điều kiện có trọng số 20%;
điểm tổng cao hơn không đảm bảo F1 ảnh sạch cao hơn, nên phải đọc cả hai.

Test robustness dùng cùng checkpoint trên clean + 4 loại × 3 mức = 13 lượt.
Seed test mặc định 2026; RNG dựa trên hash ảnh, điều kiện, seed và version
`optical-proxies-v1`. Do đó cùng ảnh/điều kiện có cùng biến đổi giữa các model.
Mức 3 chưa dùng trong training/selection; đó là mức mạnh hơn, không phải một
loại corruption hoàn toàn mới. Thông số mỗi mức nằm trực tiếp trong `degrade()`.

Với 180 ảnh, 13 lượt tạo 2.340 dự đoán nhưng vẫn chỉ có **180 ảnh độc lập**.
Baseline→robust thay cả augmentation và tiêu chí chọn checkpoint; muốn khẳng
định riêng augmentation tạo ra lợi ích phải thêm ablation giữ nguyên tiêu chí chọn.

Hướng dẫn vận hành và giới hạn: [ROBUSTNESS_GUIDELINE.md](ROBUSTNESS_GUIDELINE.md).
Đối chiếu phạm vi với [nghiên cứu robustness trong remote sensing](https://arxiv.org/abs/2306.12111);
các phép biến đổi hiện tại là protocol riêng, chưa tái lập benchmark của bài báo.

**10. Đọc metrics như thế nào? — RẤT QUAN TRỌNG**

Đọc [metrics.py](src/satdomain/metrics.py), hàm `classification_metrics()`.

| Metric | Câu hỏi nó trả lời |
|---|---|
| Accuracy | Bao nhiêu phần trăm ảnh được đoán đúng? |
| Precision của lớp | Trong những ảnh bị dự đoán là lớp đó, bao nhiêu ảnh đúng? |
| Recall của lớp | Trong ảnh thật của lớp đó, model tìm đúng bao nhiêu? |
| F1 của lớp | Precision và recall cân bằng ra sao? |
| Macro-F1 | F1 trung bình ngang nhau giữa chín lớp |
| Balanced accuracy | Recall trung bình ngang nhau giữa các lớp |
| Confusion matrix | Những cặp lớp nào bị nhầm với nhau? |
| ECE 15 bins | Confidence và tỷ lệ đúng quan sát được lệch nhau thế nào? |

Ví dụ để hiểu công thức, không phải kết quả thực nghiệm: nếu một lớp có 20 ảnh
thật, nhận ra 15 ảnh và tổng cộng dự đoán 18 ảnh là lớp đó, recall = 15/20,
precision = 15/18. F1 là trung bình điều hòa của hai giá trị. Macro-F1 là trung
bình F1 từng lớp, không phải tính F1 từ macro precision và macro recall.
Xem định nghĩa tại [tài liệu metrics của scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics).

Ma trận trong project: hàng là nhãn thật, cột là nhãn dự đoán. Bản row-normalized
giúp đọc tỷ lệ nhầm của mỗi lớp. Với 20 ảnh/lớp, một ảnh làm recall của lớp đó
thay đổi 5 điểm phần trăm; trên 180 ảnh, một ảnh làm accuracy đổi khoảng 0,56 điểm.

Mean ± standard deviation trong bảng tổng hợp là dao động giữa run, không phải
confidence interval. Chưa triển khai bootstrap confidence intervals hoặc kiểm
định ý nghĩa thống kê. ECE đã có, nhưng chưa có fitting calibration temperature,
NLL hoặc reliability diagram. `temperature=1.0` trong checkpoint không chứng
minh model đã được calibration.

Robustness dùng `drop_pp = 100 × (metric_clean − metric_degraded)`. Đây là mức
giảm do biến đổi ảnh trong cùng target set, **không phải domain gap source→target**.
Mức giảm âm là có thể xảy ra và cần báo cáo trung thực.

**11. Checkpoint, kết quả và tái lập — RẤT QUAN TRỌNG**

Đọc [artifacts.py](src/satdomain/artifacts.py) và [reports.py](src/satdomain/reports.py).

```text
<output-root>/
├── study.json
├── manifests/{source.csv, aid_test.csv, generation_summary.json nếu có}
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

Đây là cấu trúc được sinh khi chạy đủ các bước; output có thể nằm ở ổ D: trên
Windows, không phải trong repo Mac. Các run cũ có thể thiếu những file provenance
được bổ sung về sau.

`best.pt` chứa state_dict, architecture, nhãn/thứ tự lớp, image size, mean/std,
seed, epoch, training arguments, temperature và provenance. Metadata liên kết
weights với manifest/hash/code/environment. `complete.json` ghi hash các artifact
chính để kiểm tra run hoàn tất; không kiểm hash mọi biểu đồ xuất thêm.

`--skip-existing` chỉ bỏ qua run hoàn tất và hợp lệ; **không resume optimizer từ
giữa một epoch/run**. Checkpoint chưa lưu optimizer/scheduler state để resume.
Giữ output dở dang riêng nếu cần điều tra và dùng thư mục mới cho lần train mới.

`reports.py` có thể dựng lại hình từ `history.json` và `metrics.json`. Chỉ có CSV
tổng hợp không đủ để khôi phục từng dự đoán, training curve, manifest hay weights.

`evaluate_robustness.py` tạo output riêng gồm 13 thư mục điều kiện,
`robustness.csv`, `summary.json` và `robustness_curves.png`. Chưa có script tự
tổng hợp robustness trên mọi model/seed; cần chạy từng checkpoint và tổng hợp
các bảng tương ứng, giữ nguyên protocol.

**12. Inference và giới hạn sử dụng**

Đọc [infer.py](src/satdomain/infer.py). Nó tạo lại kiến trúc với
`load_pretrained=False`, nạp state_dict, chuyển sang eval, chuẩn hóa ảnh, tính
softmax và xuất top-k. Không cần tải lại pretrained weights khi đã có checkpoint
đầy đủ. Training pretrained lần đầu có thể cần Internet để lấy weights.

Ảnh ngoài chín lớp vẫn bị ép vào một trong chín lớp: đây là closed-set classifier.
Chưa có out-of-distribution detector, cloud mask hoặc cơ chế tự từ chối dự đoán.
Confidence 94% không có nghĩa xác suất đúng trong mọi domain là 94%.

Preprocessing hiện dùng `build_transforms()` và constants của code đang chạy.
Mặc dù checkpoint lưu normalization, loader hiện không tự khôi phục một pipeline
normalization tùy ý từ metadata. Nếu đổi preprocessing về sau, phải đồng bộ
train/evaluate/infer và kiểm tra tương thích checkpoint cũ.

**13. Các điểm dễ hiểu nhầm trong cấu hình và tái lập**

| Điểm | Tình trạng thực tế và ảnh hưởng |
|---|---|
| `configs/experiment.json` | Chưa được script đọc làm config runtime. Sửa file này không tự đổi lệnh train. Giá trị thực đến từ CLI/defaults và được ghi vào run metadata. |
| Nhãn lớp | Khai báo trong cả `constants.py` và `prepare_manifests.py`; thêm lớp phải đồng bộ mapping, data và classifier rồi train lại. |
| Tham số package | `pyproject.toml` dùng giới hạn phiên bản tối thiểu, chưa khóa chính xác mọi dependency. Lưu phiên bản môi trường thực tế khi chạy. |
| Seed | Có split seed, training seed và corruption seed, mỗi loại điều khiển một việc khác nhau. |
| CPU/CUDA | Cùng seed không đảm bảo weights/kết quả giống hệt giữa backend; code chưa bật toàn bộ deterministic algorithms. |
| `PYTHONHASHSEED` | Được gán lúc gọi hàm seed trong process; không nên coi đây là bảo đảm đã cố định hash randomization ngay từ lúc process khởi động. |
| Gộp kết quả | Aggregator lưu hash nhưng chưa là bộ kiểm tra đầy đủ mọi run có cùng data/protocol; cần kiểm tra provenance trước khi kết luận. |
| Data leakage | Hash bảo vệ nội dung file, không thay thế việc xác minh ảnh gần trùng, cùng địa điểm hoặc target đã dùng để chọn cải tiến. |

**14. Những kết quả đã có và những gì chưa được chứng minh**

Bảng sau được ghi lại từ CSV bạn đã gửi và giải thích CPU/CUDA trong thread
trước; không phải phép đo lại ở phiên lập tài liệu này, cũng không phải kết quả robust.

| Model | Accuracy CUDA | Accuracy CPU |
|---|---:|---:|
| SmallCNN | 68,33% | 68,33% |
| ResNet18 scratch | 71,11% | 77,22% |
| ResNet18 pretrained | 90,37% | 93,52% |
| DeiT-Tiny pretrained | 93,33% | 94,07% |

Chưa xác nhận CPU train lại hay chỉ test checkpoint CUDA; chưa xác minh đầy đủ
cùng manifest/seed/config. Chưa thể quy chênh lệch cho thiết bị hoặc kết luận
DeiT luôn tốt hơn ResNet. Chọn một môi trường nhất quán cho kết quả chính.

| Hạng mục | Trạng thái tại thời điểm tài liệu |
|---|---|
| Code bốn model, train/test/infer | Đã triển khai; có kết quả baseline người dùng chạy trên Windows |
| Manifest/checkpoint provenance và báo cáo | Đã bổ sung; run mới mới có đầy đủ metadata mới |
| `weather_robust` và test 13 điều kiện | Đã triển khai code; chưa có weights/kết quả mới được xác nhận |
| Kiểm thử | Phiên trước có 10 unit test pass và kiểm tra biến đổi trên một ảnh source thật |
| PyTorch/Matplotlib end-to-end bản mới | Chưa chạy trên Mac; cần smoke test Windows |
| Địa lý, mùa, thời tiết thật | Chưa đánh giá riêng do thiếu metadata |
| Nhận diện bão, ngập hoặc thiệt hại | Chưa triển khai |

10 kiểm thử tập trung vào integrity, bảng báo cáo và tính tái lập của biến đổi;
không chứng minh chất lượng model hoặc toàn bộ CUDA training chạy đúng.

180 ảnh AID đã được xem để phân tích và định hướng cải tiến. Nếu dùng chúng cho
quá trình phát triển, phải ghi rõ là kết quả thăm dò và có tập target mới chưa xem
cho đánh giá cuối. Tạo bản blur/haze của ảnh cũ không khôi phục tính độc lập.

**15. Phần nào cần ưu tiên tự học?**

| Ưu tiên | Chủ đề và file đọc | Sau khi đọc cần trả lời được |
|---|---|---|
| **Rất quan trọng — 1** | Methodology + manifest + split | Vì sao AID không được dùng chọn epoch? 5 fold khác full cross-validation thế nào? |
| **Rất quan trọng — 2** | `data.py`, `constants.py` | Ảnh trở thành tensor thế nào? Sai class order/normalization gây hậu quả gì? |
| **Rất quan trọng — 3** | `models.py` | SmallCNN làm gì? So cặp ResNet giúp kiểm soát yếu tố nào? |
| **Rất quan trọng — 4** | `train.py`, `run_study.py` | Loss/backprop/optimizer khác nhau ra sao? `best.pt` development và final khác gì? |
| **Rất quan trọng — 5** | `metrics.py`, `evaluate.py` | Vì sao accuracy cao chưa đủ? Lớp nào yếu? Confidence có đáng tin không? |
| **Rất quan trọng — 6** | `robustness.py`, `evaluate_robustness.py` | Mô phỏng chứng minh được gì? Vì sao 13 lượt không phải 13 lần số mẫu độc lập? |
| Quan trọng | `artifacts.py`, `reports.py` | Làm sao chứng minh kết quả thuộc checkpoint và dataset nào? |
| Quan trọng | `infer.py`, `runtime.py` | Tái sử dụng model thế nào? Vì sao CPU/CUDA có thể khác nhau? |
| Đọc khi vận hành | PowerShell scripts, `pyproject.toml`, hướng dẫn Windows | Cài môi trường, smoke test và lấy output ở đâu? |

Lộ trình gợi ý gồm sáu buổi: (1) dữ liệu/domain/leakage; (2) preprocessing và
SmallCNN; (3) ResNet/pretraining/DeiT; (4) training/validation/refit; (5) metrics và
phân tích ảnh sai; (6) robustness/provenance và lập bảng kết quả báo cáo.

Tự kiểm tra bằng việc giải thích được một ảnh đi từ folder đến nhãn dự đoán,
một checkpoint được chọn như thế nào, và một kết luận nghiên cứu dựa trên bảng nào.
Nếu chưa giải thích được ba việc đó, nên ưu tiên chúng trước khi thêm kiến trúc mới.

**16. Việc tiếp theo có giá trị nhất**

1. Chạy smoke test bản mới trên Windows theo hướng dẫn robustness.
2. Xác minh nguồn source, lưu lại tập ảnh/manifest thực tế và làm rõ hai đợt CPU/CUDA.
3. Khóa protocol baseline/robust, seed/fold, môi trường và tập target chưa xem.
4. Train lại, đánh giá cả ảnh sạch và suy giảm; lưu đủ output trước khi tổng hợp.
5. Đọc F1 từng lớp và confusion matrix, không chỉ so một cột accuracy.
6. Nếu giáo viên yêu cầu bão hoặc địa lý/thời tiết thật, thiết kế thêm dataset có
   metadata và nhãn phù hợp thay vì gọi corruption mô phỏng là bằng chứng thực tế.

Mac hiện dùng để phát triển code/tài liệu; training ở Windows. Tiếp tục tuân thủ
AGENTS.md của máy dùng chung: giới hạn worker, không stress/load test, không
Docker/Colima trên Mac, và dọn process nền sau công việc.
