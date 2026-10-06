# Phân tích lỗi thực tế — B5-balanced, seed 3407

Checkpoint được chọn bằng validation trước test; N=2149. Các tỷ lệ này mô tả một checkpoint, không thay bảng trung bình ba seeds.

## Nhầm lớp

- negative: 215/1368 mẫu sai; cặp nhầm nhiều nhất negative → neutral: 170 mẫu (12.43% support của lớp thật).
- neutral: 194/448 mẫu sai; cặp nhầm nhiều nhất neutral → negative: 157 mẫu (35.04% support của lớp thật).
- positive: 120/333 mẫu sai; cặp nhầm nhiều nhất positive → neutral: 79 mẫu (23.72% support của lớp thật).

Lớp có F1 cao nhất của checkpoint này là **negative**, F1=84.81%. Số lượng mẫu đúng tuyệt đối không được dùng để gọi lớp dễ nhất.

## Độ dài, OOV và negation

Các ngưỡng độ dài lấy từ quartile train: [13.0, 20.0, 24.0]. Bảng `results/tables/error_slices.csv` có support và phân bố class để tránh diễn giải sai do class mix.

| slice | value | support | accuracy | macro_f1 | error_rate | support_negative | support_neutral | support_positive |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| length_bin | (-1.0, 13.0] | 577 | 0.6880 | 0.6909 | 0.3120 | 220 | 213 | 144 |
| length_bin | (13.0, 20.0] | 607 | 0.7496 | 0.6896 | 0.2504 | 386 | 122 | 99 |
| length_bin | (20.0, 24.0] | 499 | 0.7796 | 0.6004 | 0.2204 | 378 | 59 | 62 |
| length_bin | (24.0, inf] | 466 | 0.8133 | 0.5248 | 0.1867 | 384 | 54 | 28 |
| flag_negation | False | 1448 | 0.7265 | 0.7013 | 0.2735 | 777 | 380 | 291 |
| flag_negation | True | 701 | 0.8103 | 0.5200 | 0.1897 | 591 | 68 | 42 |
| flag_any_oov | False | 930 | 0.7516 | 0.7028 | 0.2484 | 560 | 207 | 163 |
| flag_any_oov | True | 1219 | 0.7555 | 0.6716 | 0.2445 | 808 | 241 | 170 |
| truncated | False | 2149 | 0.7538 | 0.6869 | 0.2462 | 1368 | 448 | 333 |

Các chênh lệch là quan sát trên test; chưa chứng minh độ dài/OOV/negation gây ra lỗi. Negation flags dựa trên từ not/no/never trong clean text, cần kiểm tra ngữ cảnh khi phân tích trường hợp cụ thể.

## Cleaning, sarcasm và manual review

Cleaner P0 bỏ emoji/dấu câu, giữ phủ định và nội dung hashtag. Bảng before/after chứng minh phép biến đổi, chưa chứng minh ảnh hưởng F1 vì chưa chạy preprocessing ablation. Tập review có 85 dòng, sampling seed=42, gồm cặp nhầm, mẫu đúng và margin thấp.

`results/predictions/manual_error_review.csv` có cột hai người review và notes để nhóm ghi các nhãn negation, sarcasm, mixed sentiment, slang/typo, missing context, truncation hoặc uncertain. **Chưa có nhãn sarcasm được con người xác minh, nên chưa có kết luận về sarcasm.** Không tự gán sarcasm từ một từ như great.

Softmax và top-2 margin chưa được calibration. Confidence thấp không đồng nghĩa prediction sai; confidence cao cũng có thể sai.

## Giới hạn

Test đã được mở cho phân tích này; không dùng các lỗi để tiếp tục chọn hyperparameter trên cùng holdout. Các cải tiến từ review được ghi future work hoặc kiểm tra trên validation/holdout mới.
