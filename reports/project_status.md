# Trạng thái bàn giao — 06/10/2026

Phần thực nghiệm và tài liệu bàn giao hoàn tất: **22 run RNN, 8 configurations**, báo cáo Word theo mẫu và 16 slide có speaker notes. Ba training seeds cho mỗi configuration sạch; A1 chỉ seed 42. Huấn luyện trên CPU máy cá nhân, tổng training wall time 19,91 phút. Notebook đã chạy từ đầu đến cuối, có output và bản HTML xem nhanh.

| Phase | Trạng thái | Bằng chứng |
| --- | --- | --- |
| 0 — Audit nguồn | Hoàn tất; nguồn thực tế dùng LSTM/binary | `source_audit.md`, `sources/` |
| 1 — Môi trường/dataset | Hoàn tất; dependency pin và raw SHA-256 | `environment/`, raw manifest |
| 2 — EDA/chất lượng/split | Hoàn tất; main 14.353 mẫu, nhóm trùng không chéo split sạch | Split/exclusion logs, F01–F05 |
| 3 — Preprocessing | Hoàn tất; train-only main/A2, source tokenizer khớp Keras trên 6.541 tweet | Representation/tokenizer JSON, F06–F08 |
| 4 — Baseline | A1 chuyển thể/A2 sạch đã chạy; giữ cả run A2 biến thiên mạnh | Baseline runs/history |
| 5 — Đánh giá baseline | Hoàn tất; benchmark riêng, đủ metrics/reports | Test JSON/predictions |
| 6 — Controlled experiments | Hoàn tất: sequence length, hidden units, class weighting | Frozen matrix, validation tables, F20/F21 |
| 7 — Mô hình cuối | B5-balanced, demo seed 3407 chọn bằng validation trước test | Selection/manifest/final bundle |
| 8 — Error analysis | CM/slices/examples và nhận xét riêng của trợ lý cho 85 case hoàn tất; chưa có hai người review | `error_analysis.md`, assistant/manual review CSV |
| 9 — Demo/inference | Website có logo trường/thông tin nhóm và stream tám khâu, trace khớp inference cũ | Demo verification/rehearsal JSON, F23 |
| 10 — Báo cáo/thuyết trình | Word 37 trang, 19 hình/19 bảng; PPTX 16 slide; kịch bản 12 phút 45 giây | `reports/final/`, nguồn authoring, CSV/PNG/PDF |

Kết quả test ba lớp, **mean ± SD qua ba training seeds trên cùng split**: Accuracy **75,18% ± 1,34 điểm phần trăm**, Macro-F1 **0,6914 ± 0,0158**, Weighted-F1 mean **0,7568**, N=2.149. Checkpoint seed 3407 dùng cho demo/CM riêng có Accuracy 0,753839 và Macro-F1 0,686875. Source 90,32% là output LSTM/binary lịch sử, không phải kết quả nhóm tái lập chính xác.

## Kiểm tra đã thực hiện

- 25 tests pass, gồm 11 protocol/holdout-gate tests, 10 trace/input cases và 4 naming/compatibility tests; integrity của 22 run, train-only vocabulary/group isolation và source tokenizer parity.
- Tính lại metrics/CM từ 10 bộ test predictions khớp JSON; thời điểm đánh giá sau frozen selection.
- Inference khớp 12 saved predictions; model cuối chỉ dùng RNN.
- Notebook 12 code cells chạy liên tiếp không lỗi; HTML có 25 ảnh, không ảnh lỗi khi kiểm tra. Có 23 hình riêng ở dạng PNG/PDF.
- Demo API trả probabilities hợp lệ; input rỗng/JSON lỗi trả 400. F23 là ảnh request thật. Ba request tập thuyết trình khớp model/seed và saved predictions.
- Word đã render và kiểm tra từng trang; mục lục/danh mục ánh xạ đúng trang main section. Logo, khung bìa và 31 phần package của mẫu được giữ nguyên; template gốc không bị sửa.
- PPTX có 6 bảng và 6 biểu đồ native, 16 speaker notes; package/layout/import checks pass, đã render file bàn giao để kiểm tra. Chưa thử trên Microsoft PowerPoint trực tiếp.
- ZIP giữ dataset, code, mọi checkpoint/history, notebook có output, hình, báo cáo và slide; loại `.venv`, machine kernel paths, `.build` và `node_modules`.

Các biên nhận ở `results/metrics/`: `artifact_verification.json`, `demo_verification.json`, `presentation_demo_rehearsal.json`, `report_delivery_verification.json`. Nguồn tạo lại Word/slide nằm ở `reports/build_sources/`; quy trình ML không phụ thuộc công cụ authoring.

## Việc nhóm cần làm trước khi nộp

Kiểm tra thông tin bìa theo mẫu, điền đóng góp thực tế của từng thành viên trong phụ lục và tập trình bày bằng kịch bản đã có. Mở PowerPoint trên máy trình chiếu để kiểm tra hiển thị thực tế. Báo cáo và slide đã viết đầy đủ; các ô đóng góp chưa rõ được để trống có chủ đích.

Nếu muốn xác minh sâu các giả thuyết sarcasm/mixed sentiment/cleaning, hai người review độc lập 85 case rồi báo số case/agreement. Hiện chỉ có nhận xét của trợ lý; không công bố human agreement hoặc tỷ lệ sarcasm toàn dataset. Huấn luyện trên Colab/cloud chưa chạy kiểm chứng trong lần này; README có hướng dẫn chuyển môi trường.

Test đã mở. Không chọn tiếp hyperparameter theo test hiện có. Khảo sát thêm preprocessing/LR/clipping hoặc comparator cần thiết kế vòng mới phù hợp và tách khỏi ma trận đã công bố.

Cập nhật theo danh sách nhóm: Trịnh Nguyễn Anh Hào (2611307), Lê Huy Huân (2611308), Nguyễn Nam Triều Tiên (2611323). Website và Word đọc cùng metadata. Kiểm tra streaming/OOV/truncation/input và parity: `results/metrics/demo_analysis_verification.json`.

Tên đề tài đã chốt: **Xây dựng mô hình phân tích cảm xúc khách hàng trong ngành hàng không sử dụng RNN**. Website, bìa Word/slide, notebook và tên file bàn giao đã đồng bộ. Kiểm tra sau đổi tên xác nhận 376 tệp khoa học bất biến; tên API chuẩn và metadata checkpoint lịch sử được giữ để bảo toàn khả năng tải mô hình. Giao diện đã kiểm tra ở desktop 1.000 px và mobile 390 px, không tràn ngang.

## GitHub và triển khai website miễn phí

Mã nguồn, dataset, 22 run/checkpoints, notebook và tài liệu đã được đẩy lên repository công khai:
https://github.com/nguyennamtrieutien/airline-rnn-sentiment-analysis

Demo đã triển khai thành công trên GitHub Pages:
https://nguyennamtrieutien.github.io/airline-rnn-sentiment-analysis/

RNN chạy trực tiếp trong trình duyệt bằng Web Worker, dùng đúng trọng số float32 xuất từ checkpoint B5-balanced/seed3407; không huấn luyện lại. Tokenizer, P0, masking và kiến trúc được giữ nguyên. Đối chiếu 2.149 predictions test đã lưu và 18 câu bổ sung: 2.167 lớp dự đoán khớp, sai lệch xác suất lớn nhất 3,924×10⁻⁷. Đây là kiểm chứng chuyển đổi inference; không thay số liệu thực nghiệm trong báo cáo.

GitHub Actions kiểm tra cú pháp, checksum và parity trước mỗi deploy. Website công khai đã kiểm tra ba câu, đủ tám khâu, đúng checkpoint, logo/tên nhóm, mobile390px không tràn ngang và không có lỗi JavaScript. Nội dung nhập không gửi đến API inference bên ngoài. Biên nhận: `browser_inference_verification.json`, `browser_ui_verification.json`, `github_pages_verification.json` trong `results/metrics/`. Hướng dẫn build, kiểm tra và deploy ở README.
