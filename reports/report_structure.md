# Cấu trúc báo cáo tiểu luận

Tham khảo báo cáo mẫu 34 trang về GCN/superpixel/CIFAR10 để tổ chức: trình bày bài toán và lý thuyết → pipeline → setup thực nghiệm → bảng/ảnh → phân tích/hạn chế → demo. Không chuyển kết quả hoặc kết luận GCN sang sentiment. Mọi lựa chọn model trong dự án này dùng validation; tránh cách pilot chọn bằng test trong mẫu.

## MỞ ĐẦU

Bối cảnh tweet phản hồi hãng hàng không; mục tiêu học biểu diễn chuỗi bằng RNN; phạm vi tiếng Anh/ba sentiments. Hai mục tiêu: audit/adapt source binary pipeline và controlled experiments trên benchmark sạch. Nêu khác biệt LSTM trong nguồn để tránh gọi sai reproduction. Đóng góp: pipeline có checksum/group split, ma trận một factor, metrics đa lớp, error analysis và demo.

## CHƯƠNG 1. TỔNG QUAN VÀ CƠ SỞ LÝ THUYẾT

1. Sentiment Analysis, bài toán hai lớp/ba lớp và domain airline.
2. Sequential data: thứ tự từ, độ dài biến thiên, padding/masking.
3. Text preprocessing, tokenization, OOV, train-only vocabulary, nguy cơ mất phủ định/emoji.
4. Embedding học cùng model: ma trận V×D; không dùng pretrained embedding trong run hiện tại.
5. RNN: hidden state, công thức cập nhật tanh, unrolling/BPTT, output classification.
6. Vanishing/exploding gradient: cơ chế lý thuyết, gradient clipping; phân biệt lý thuyết với vấn đề thực sự đã đo.
7. Cross-entropy/Adam/dropout/early stopping.
8. Accuracy, per-class Precision/Recall/F1, Macro/Weighted-F1, confusion matrix và imbalance.

Trích dẫn tài liệu gốc cho phần lý thuyết khi biên tập; không dẫn notebook Kaggle như bằng chứng duy nhất cho lý thuyết RNN.

## CHƯƠNG 2. PHƯƠNG PHÁP VÀ GIẢI PHÁP

1. Pipeline tổng thể F08: audit → group split → train-only tokenizer → model → validation selection → final test.
2. Dataset/schema; raw→source filter F01–F04; feature text/target sentiment.
3. Duplicate/conflicting-label policy và manifest.
4. P0 và source preprocessing riêng; bảng ví dụ F07; token IDs/OOV/PAD/masking.
5. Kiến trúc RNN và bảng Input/Output/Parameters F09–F10.
6. Training pipeline/checkpoint/ES; class weights train-only; seed/environment/config logging.
7. Định nghĩa A1 adapted/A2 clean/C0 main; ma trận B chỉ đổi một factor.
8. Quy tắc chọn config/seed trước test; majority comparator; inference bundle dùng nguyên pipeline.

## CHƯƠNG 3. THỰC NGHIỆM VÀ ĐÁNH GIÁ

Dùng `experiment_chapter.md` làm bản nháp có số thật; các CSV làm bảng gốc, PNG/PDF làm hình. Không sửa metric chỉ để gần90%.

1. Môi trường thực tế và dependency versions; CPU/GPU/RAM/thời gian.
2. Dataset split/exclusions/class imbalance/group isolation.
3. A1/A2: baseline adapted và clean; khác biệt với nguồn.
4. C0 và controlled experiments: câu hỏi/factor/điều kiện cố định; mean±SD validation F20/F21.
5. Cấu hình cuối chọn bằng validation; test mean±SD và checkpoint seed trung vị.
6. Hội tụ F11–F13, train-fit và inference-mode metrics.
7. Confusion matrix F14/F15; per-class F16; hướng nhầm từng lớp.
8. Error analysis F17–F19/F22, length/negation/OOV slices và support/class mix.
9. Manual review: ghi số case/tiêu chí/agreement sau khi nhóm annotate; để trạng thái chưa hoàn thành nếu chưa review.
10. Đối chiếu nguồn và các khác biệt khiến Accuracy không so trực tiếp.
11. Hạn chế: fixed split, training variability, coverage T40/T60, dropout/budget, chưa calibration/domain shift, chưa preprocessing/gradient ablation.

## CHƯƠNG 4. DEMO VÀ TRIỂN KHAI

1. Artifact model/tokenizer/mapping/checksum.
2. Input tiếng Anh → P0 → sequence/padding → RNN → softmax.
3. Giao diện nhập tweet và sentiment/probabilities; screenshot demo thật.
4. Những câu nhập mới minh họa đúng và sai; không bổ sung chúng vào test metric.
5. Giới hạn ngôn ngữ/domain và softmax chưa calibration.

## KẾT LUẬN

Nhắc lại mục tiêu đã thực hiện, trình bày số thật của benchmark tương ứng, insight được validation/test hỗ trợ và phạm vi giới hạn. Hướng mở rộng tách riêng: manual review sâu, preprocessing/clipping/LR, domain/time holdout; LSTM/GRU/Transformer chỉ là comparator được duyệt cho vòng sau.

## TÀI LIỆU THAM KHẢO VÀ PHỤ LỤC

Dataset/notebook nguồn; tài liệu RNN/Embedding/masking/Adam/metrics; approved plan; frozen matrix; hyperparameter/environment; checksum; hướng dẫn chạy; class report/history; manual annotation protocol và notebook link. Phân công đóng góp thành viên theo công việc thực tế, không tự điền tên/thành tích.
