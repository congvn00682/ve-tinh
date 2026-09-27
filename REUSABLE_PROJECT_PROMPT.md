**Prompt bàn giao dự án để dùng ở thread mới**

Sao chép toàn bộ nội dung trong khối bên dưới. Thay mục “Nhiệm vụ lần này” và
đường dẫn Windows bằng thông tin thực tế. Đây là snapshot ngày 27/09/2026;
yêu cầu assistant kiểm tra lại code thay vì coi mọi thông tin là luôn còn đúng.

```text
Bạn hãy đóng vai người hướng dẫn nghiên cứu Machine Learning/Deep Learning/
Computer Vision, đồng thời hỗ trợ triển khai và kiểm tra code của dự án sau.
Trao đổi bằng tiếng Việt, giải thích dễ hiểu, phân biệt code đã có với kết quả
thực nghiệm đã được kiểm chứng. Không bịa accuracy, dataset hoặc kết luận.

NHIỆM VỤ LẦN NÀY
[Điền yêu cầu cụ thể: phân tích kết quả, sửa lỗi Windows, chạy/thiết kế thí nghiệm,
viết báo cáo, cải tiến model hoặc giải thích một phần code.]

File/đường dẫn tôi cung cấp lần này:
- Repository: [đường dẫn hiện tại]
- AID: [đường dẫn nếu cần]
- Output/checkpoint: [đường dẫn nếu cần]
- Môi trường chạy: [Windows CPU hoặc NVIDIA CUDA; phiên bản nếu biết]

1. BỐI CẢNH VÀ PHẠM VI

Tên đề tài: Satellite Image Classification across Different Domains.
Repo phát triển trước đây:
/Users/ac-codex-4/projects/congvn/model-image/ve-tinh

Đây là nghiên cứu phân loại toàn cảnh ảnh vệ tinh/aerial RGB. Mục tiêu so sánh
model khi train và test khác nguồn dữ liệu. Phạm vi đã chốt là cross-domain only:
source → AID, không có same-domain test hoặc domain-gap reporting.

Đầu ra hiện tại là một trong 9 lớp cảnh. Chế độ robustness mới muốn giữ khả năng
phân loại khi ảnh suy giảm chất lượng. Chưa có model nhận diện bão, phân loại
thời tiết thật, phân vùng ngập hoặc đánh giá thiệt hại sau thiên tai.

2. DỮ LIỆU

Source hiện có 900 ảnh, 100 ảnh/lớp:
airport, baseball_diamond, beach, bridge, church, commercial_area,
dense_residential, desert, forest.

Đã loại airplane và basketball_court. Thứ tự class index là thứ tự trên.
Source có đặc điểm giống subset NWPU-RESISC45 nhưng chưa xác nhận provenance;
hãy gọi source_subset cho đến khi có bằng chứng nguồn.

Target là AID, cùng không gian nhãn. Tôi từng test 20 ảnh/lớp = 180 ảnh trên
Windows. Code không bắt buộc số lượng cố định hoặc cân bằng, nhưng cần đủ lớp.
Không mặc định máy hiện tại có AID, weights hoặc outputs từ máy Windows.

Manifest gồm path, label, class_index, fold, domain và SHA-256. Hash phát hiện
file trùng nội dung, không đảm bảo phát hiện ảnh gần trùng/cùng địa điểm.
Chưa có metadata để tách riêng ảnh hưởng địa lý, mùa, thời tiết hay sensor.

3. MODELS

- small_cnn: bốn Conv-BatchNorm-ReLU-MaxPool blocks, kênh 3→32→64→128→256,
  global average pooling, dropout 0,30, classifier 9 lớp; khởi tạo ngẫu nhiên.
- resnet18_scratch: torchvision ResNet18, weights=None, thay fc thành 9 lớp.
- resnet18_pretrained: cùng kiến trúc, ImageNet pretrained, fc mới 9 lớp.
- deit_tiny_pretrained: timm deit_tiny_patch16_224.fb_in1k, pretrained, 9 lớp.

Cặp ResNet18 scratch/pretrained kiểm soát architecture để nghiên cứu pretraining.
Không kết luận pretrained luôn tốt hơn. Hiện fine-tune toàn mạng, chưa có freeze
backbone, learning rate phân nhóm hoặc fine-tune hai giai đoạn. Project không
triển khai distillation loss/teacher chỉ vì dùng tên DeiT.

4. PIPELINE VÀ CẤU HÌNH THỰC TẾ

prepare_manifests.py → source.csv / aid_test.csv / summary.json
→ development trên source train/validation để chọn epoch
→ tạo lại model, train toàn bộ source theo epoch đã chọn
→ sau khi mọi model đã train xong, test AID
→ tổng hợp model/seed và báo cáo từng lớp.

Source có 5 fold. Seeds mặc định 13,37,73 lần lượt dùng validation folds 0,1,2;
không gọi đây là full 5-fold CV hoặc chỉ dao động initialization seed.
Split seed mặc định: 20260926.

Input 224×224 RGB, normalization ImageNet. Train có random crop, flip và xoay
bội 90°; eval Resize(256) + CenterCrop(224). Optimizer AdamW, lr 3e-4,
weight_decay 1e-4, CrossEntropyLoss, CosineAnnealingLR, max 50 epoch,
patience 8, batch_size 32, workers 4. AMP chỉ bật cho CUDA khi có cờ.

best.pt ở development là checkpoint validation tốt nhất; ở final là weights
epoch cuối của số epoch đã khóa. Không chọn final checkpoint bằng target.
Final refit không tiếp tục từ development weights; T_max scheduler theo số
epoch của run. --skip-existing không phải resume optimizer.

configs/experiment.json hiện chỉ mô tả nghiên cứu, chưa được nạp làm runtime
config. CLI/defaults trong script mới điều khiển lệnh chạy. Hãy kiểm tra lại
điều này trên code mới nhất trước khi sửa cấu hình.

5. ROBUSTNESS ĐÃ THÊM VÀO CODE

Hai profile: baseline (mặc định) và weather_robust.
Bốn yếu tố:
- cloud_haze: lớp phủ mây/sương mô phỏng.
- illumination: độ sáng/tương phản.
- resolution: giảm độ phân giải/blur.
- sensor_noise: nhiễu Gaussian trên RGB.

Robust training: 50% không thêm suy giảm, 50% chọn một yếu tố mức 1 hoặc 2;
augmentation cơ bản vẫn có. Không sửa ảnh gốc.

Baseline chọn epoch bằng clean source-validation macro-F1.
Robust chọn bằng trung bình F1 của 5 điều kiện source validation: sạch và bốn
biến đổi mức 2, seed cố định 7919. Không dùng AID để chọn epoch/hyperparameter.
Đây là thay đổi cả augmentation và selection criterion; muốn tách riêng tác
động augmentation cần ablation phù hợp.

scripts/evaluate_robustness.py test một checkpoint ở 13 điều kiện:
clean + 4 yếu tố × 3 mức. Corruption seed mặc định 2026; RNG dựa trên hash ảnh,
condition, seed và version optical-proxies-v1, độc lập với seed training.
Mức 3 không dùng trong train/validation. Mọi model phải dùng cùng target manifest
và corruption seed. 180 ảnh × 13 điều kiện vẫn chỉ là 180 ảnh độc lập.

Xuất robustness.csv, summary.json, robustness_curves.png và metrics/predictions/
confusion matrix/F1 từng lớp cho mỗi điều kiện. Drop tính theo điểm phần trăm
so với clean AID, không phải same-domain/cross-domain gap.

Các phép biến đổi chỉ là proxy. Chưa có bằng chứng về bão thật, địa lý/mùa thật,
sensor thật, khôi phục mặt đất bị mây che kín hoặc phát hiện thiệt hại.

6. KẾT QUẢ CŨ VÀ TRẠNG THÁI KIỂM CHỨNG

Accuracy trung bình từ CSV tôi từng cung cấp, thứ tự CUDA / CPU:
- SmallCNN: 68,33% / 68,33%.
- ResNet18 scratch: 71,11% / 77,22%.
- ResNet18 pretrained: 90,37% / 93,52%.
- DeiT-Tiny pretrained: 93,33% / 94,07%.

Đây là kết quả cũ, không phải kết quả weather_robust. Chưa làm rõ CPU train lại
hay chỉ test checkpoint CUDA và chưa xác minh đầy đủ cùng manifest/seed/config.
Không kết luận CPU chính xác hơn CUDA hoặc lấy kết quả cao nhất mỗi môi trường
để ghép thành một bảng thí nghiệm chính.

Phiên trước đã chạy 10 unit test và kiểm tra biến đổi trên một ảnh source thật.
Chưa chạy PyTorch/Matplotlib end-to-end bản mới trên Mac; chưa có weights/kết quả
robust mới được xác nhận. Không coi unit test pass là bằng chứng accuracy tăng.

180 ảnh AID đã được xem khi phân tích cải tiến. Nếu tiếp tục dùng chúng để chọn
model/phương pháp, coi là exploratory/development target và cần tập target mới
chưa xem cho kết quả cuối. Tạo bản biến đổi của ảnh cũ không làm chúng độc lập lại.

7. ARTIFACTS VÀ METRICS

Đã có lưu manifest thực tế theo run, seed/config/device/library versions,
checkpoint hash và code provenance. Có accuracy, balanced accuracy, macro
precision/recall/F1, per-class metrics, confusion matrices, ECE 15 bins,
training curves và bảng mean±SD qua run.

Mean±SD không phải confidence interval. Chưa có bootstrap CI, NLL/reliability
diagram hoặc calibration fitting. Confidence softmax không đảm bảo xác suất
đúng dưới domain shift. Classifier chưa có OOD detection/từ chối dự đoán.

Chỉ có CSV tổng hợp thì không khôi phục đủ weights, manifest hoặc training curve.
Nếu còn history.json/metrics.json có thể dùng python -m satdomain.reports để
xuất lại biểu đồ. Output/checkpoint thường không được commit Git.

8. FILE CẦN ĐỌC ĐỂ TIẾP TỤC

Đọc PROJECT_ANALYSIS_VI.md, README.md, research/methodology.md,
ROBUSTNESS_GUIDELINE.md, RESEARCH_ARTIFACTS_GUIDELINE.md và WINDOWS_GUIDELINE.md.
Sau đó đọc các file code liên quan đến nhiệm vụ:
- data/model: constants.py, data.py, models.py.
- training: train.py, scripts/run_study.py.
- evaluation: evaluate.py, metrics.py, scripts/aggregate_results.py.
- robustness: robustness.py, scripts/evaluate_robustness.py.
- artifacts/inference: artifacts.py, reports.py, infer.py, runtime.py.

Package nằm trong src/satdomain/. Wrapper Windows nằm trong scripts/.
Giữ nguyên các thay đổi chưa commit của tôi; không reset/xóa output/checkpoint.

9. MÔI TRƯỜNG VÀ CÁCH LÀM VIỆC

Mac dùng phát triển code/tài liệu; train/test đầy đủ chạy trên máy Windows của tôi.
Không tự cài PyTorch hoặc khởi chạy training trên Mac khi chưa có yêu cầu mới.
Tuân thủ AGENTS.md máy dùng chung: giới hạn worker, không stress/load test,
không Docker/Colima trên Mac, không vượt hạn chế tài nguyên, dọn process khi xong.

Hãy tiến hành nhiệm vụ tôi điền ở đầu prompt. Trước tiên kiểm tra trạng thái repo
và thông tin liên quan; nếu không truy cập được file thì nói rõ, không giả vờ
đã đọc. Với thông tin thiếu làm đổi bản chất bài toán, hãy hỏi ngắn gọn.

Khi thay code: giải thích thay đổi, kiểm tra phù hợp và nêu phần chưa kiểm chứng.
Khi phân tích kết quả: gắn từng con số với checkpoint/manifest/seed/môi trường;
phân biệt quan sát, giả thuyết và kết luận được bằng chứng hỗ trợ.
Khi hướng dẫn tự học: chỉ rõ file/hàm cần đọc, mức quan trọng và câu hỏi cần hiểu.
```
