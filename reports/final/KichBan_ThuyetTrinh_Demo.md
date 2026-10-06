# Kịch bản thuyết trình — Xây dựng mô hình phân tích cảm xúc khách hàng trong ngành hàng không sử dụng RNN

Bản đi kèm PowerPoint 16 slide. Thời lượng đề xuất **12 phút 45 giây**, trong đó demo 70 giây; dành thêm khoảng 5 phút hỏi đáp. Mỗi slide đã có speaker notes. Nội dung dưới là gợi ý tập nói, không phải bản xác nhận các thành viên đã thực hiện từng công việc.

## Chuẩn bị trước buổi trình bày

Danh sách ba thành viên đã điền ở Word; bổ sung phân công thực tế vào phụ lục, kiểm tra thông tin trên bìa theo mẫu. Mở PowerPoint trên máy trình chiếu để kiểm tra font Arial, biểu đồ và chế độ Presenter View; lần bàn giao này đã kiểm tra package và import/render, chưa chạy trực tiếp Microsoft PowerPoint. Có sẵn ảnh F23 và bản HTML notebook nếu máy trình chiếu chưa có Python.

Từ terminal trong thư mục `airline-rnn`, dùng môi trường đã cài theo README:

```bash
source .venv/bin/activate
python -m airline_rnn demo --port 8765
```

Mở http://127.0.0.1:8765. Nếu server đã chạy thì chỉ mở trang, không khởi động bản thứ hai trên cùng port. Không chạy lại training khi thuyết trình. Model dùng B5-balanced, seed 3407, checkpoint epoch 16; manifest đã khóa model/tokenizer và mapping lớp.

## Phân bổ thời gian

| Slide | Nội dung | Thời lượng | Mốc bắt đầu |
| --- | --- | --- | --- |
| 1 | Xây dựng mô hình phân tích cảm xúc khách hàng trong ngành hàng không sử dụng RNN | 20 giây | 00:00 |
| 2 | Notebook nguồn thực tế dùng LSTM và hai lớp | 60 giây | 00:20 |
| 3 | Negative chiếm 62,69% dữ liệu gốc | 40 giây | 01:20 |
| 4 | Nhóm dữ liệu trùng đi trọn vào một split | 55 giây | 02:00 |
| 5 | RNN cập nhật trạng thái theo thứ tự token | 55 giây | 02:55 |
| 6 | Checkpoint chọn bằng validation Macro-F1 | 55 giây | 03:50 |
| 7 | A2 cho thấy biến thiên lớn giữa training seeds | 50 giây | 04:45 |
| 8 | Các đối chứng B chỉ đổi một yếu tố so với C0 | 45 giây | 05:35 |
| 9 | B5-balanced có validation Macro-F1 cao nhất | 40 giây | 06:20 |
| 10 | T60 tăng thời gian nhưng không thêm token thật | 35 giây | 07:00 |
| 11 | Checkpoint demo ở epoch 16 trong 20 epoch | 40 giây | 07:35 |
| 12 | B5 đạt test Macro-F1 trung bình 0,6914 | 55 giây | 08:15 |
| 13 | Neutral thường nhầm negative, positive nhầm neutral | 60 giây | 09:10 |
| 14 | Lỗi chứa cảm xúc pha trộn và thiếu ngữ cảnh | 60 giây | 10:10 |
| 15 | Demo tải đúng checkpoint đã đánh giá | 70 giây | 11:10 |
| 16 | Kết quả có thể truy vết, neutral còn là hạn chế | 25 giây | 12:20 |

## Lời dẫn từng slide

### Slide 1. Xây dựng mô hình phân tích cảm xúc khách hàng trong ngành hàng không sử dụng RNN

Kính chào thầy và các bạn. Đề tài nghiên cứu RNN cho sentiment của tweet hàng không với ba lớp negative, neutral và positive. Trình bày tập trung vào audit nguồn, protocol tái lập, các đối chứng có kiểm soát và kết quả thực đo. Thông tin nhóm ở Word/website đã bổ sung theo danh sách thực tế: Trịnh Nguyễn Anh Hào 2611307, Lê Huy Huân 2611308, Nguyễn Nam Triều Tiên 2611323.

Nguồn đối chiếu: Báo cáo Word; results/tables/test_summary.csv.

### Slide 2. Notebook nguồn thực tế dùng LSTM và hai lớp

Tên notebook ghi RNN nhưng code thực tế dùng LSTM và bỏ toàn bộ neutral, sau đó bỏ 5.000 negative đầu. Nguồn fit tokenizer trên corpus trước chia tập. Audit A1 phát hiện 68 tweet_id và 85 clean text trùng qua splits. Những khác biệt này giới hạn cách so sánh với con số 90,32%. Nhóm giữ yêu cầu RNN, vì thế gọi A1 là baseline chuyển thể, không công bố tái lập chính xác nguồn. Chưa lượng hóa mức Accuracy tăng do overlap.

Nguồn đối chiếu: https://www.kaggle.com/code/chibuzorokocha/airline-sentiment-analysis-90-accuracy-using-rnn ; reports/source_audit.md.

### Slide 3. Negative chiếm 62,69% dữ liệu gốc

Dataset Twitter US Airline Sentiment của CrowdFlower chứa 14.640 dòng với negative 9.178, neutral 3.099 và positive 2.363. Mô hình chỉ nhận văn bản. Airline không làm feature, tweet_id hỗ trợ audit và grouping. Lớp negative chiếm đa số nên Accuracy riêng lẻ dễ che hiệu quả thấp ở neutral/positive. Majority baseline trên test sạch đạt 63,66% Accuracy nhưng Macro-F1 chỉ 0,2593. Các giá trị trong biểu đồ được làm tròn sáu chữ số thập phân; CSV giữ độ chính xác gốc.

Nguồn đối chiếu: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment ; results/tables/dataset.csv.

### Slide 4. Nhóm dữ liệu trùng đi trọn vào một split

Dữ liệu sau audit còn 14.353 tweet. Nhóm trùng được nối bắc cầu bằng tweet_id, canonical raw text hoặc clean text. Mọi thành viên nhóm cùng split nên loại được overlap theo định nghĩa này. Stratification cố gắng giữ tỷ lệ lớp, số dòng thực tế lệch nhẹ tỷ lệ lý tưởng do nhóm không thể tách. Train-only vocabulary và test gate là hai cơ chế riêng. Chúng không loại được tất cả quan hệ ngữ nghĩa hoặc hội thoại gần nhau.

Nguồn đối chiếu: results/tables/dataset_splits.csv ; results/metrics/artifact_verification.json.

### Slide 5. RNN cập nhật trạng thái theo thứ tự token

Mỗi token tra embedding 128 chiều rồi đi qua RNN. Trạng thái hiện tại là tanh của tổng biến đổi tuyến tính embedding và trạng thái trước. Mạng lấy trạng thái cuối của chuỗi đã masking, qua Dense100 và Dense3. Riêng RNN có H(D+H+1)=63.700 tham số, embedding có 512.000 nên chiếm phần lớn tổng. Padding ID0 được mask để không đưa phần đệm vào nội dung. Toàn bộ 22 run đều là RNN.

Nguồn đối chiếu: https://keras.io/api/layers/recurrent_layers/simple_rnn/ ; https://keras.io/api/layers/core_layers/embedding/ ; results/tables/final_architecture.csv.

### Slide 6. Checkpoint chọn bằng validation Macro-F1

Checkpoint dùng Macro-F1 lớn nhất trên toàn validation, tie thì val loss thấp hơn. Early Stopping giữ một mốc riêng với min_delta, nên không nhất thiết cùng checkpoint best. History lưu metric và wall time. Dropout giúp regularization, gradient clipping giới hạn exploding gradient nhưng không chữa hoàn toàn vanishing gradient. Main model có class weighting ở B5, validation metric không nhân class weights. Tổng wall time training của 22 run khoảng 19,91 phút trên CPU này.

Nguồn đối chiếu: results/tables/final_training_configuration.json ; environment/ ; https://arxiv.org/abs/1211.5063 ; https://arxiv.org/abs/1412.6980.

### Slide 7. A2 cho thấy biến thiên lớn giữa training seeds

A1 giữ source pipeline, chỉ chuyển mạng hồi quy sang RNN và dùng learning rate tường minh. A2 sạch hơn nhưng vẫn source cleaner và không masking/clipping; nhiều thành phần khác nên không phải ablation một yếu tố. Seed42 dừng epoch6 với checkpoint1 và thiên negative. Hai seed khác tốt hơn, vì vậy độ lệch chuẩn cao là thông tin chính. Nhóm không loại run xấu và không lấy seed 3407 làm đại diện cho mean A2.

Nguồn đối chiếu: results/tables/test_summary.csv ; results/tables/test_runs.csv ; reports/experiment_chapter.md.

### Slide 8. Các đối chứng B chỉ đổi một yếu tố so với C0

Ma trận chính có sáu configurations, gồm mốc C0 và năm thay đổi độc lập trong ba nhóm nghiên cứu. Tổng tính cả A1/A2 là tám configurations và 22 run. Không khảo sát dropout, learning rate, embedding dimension hoặc vocabulary size trong lần chạy này. B1-60 giữ nguyên mọi thứ trừ T, B3 thay H và B5 thay class weights. Không có test metrics cho các cấu hình B chưa thắng vì test chỉ dành cho đánh giá cuối.

Nguồn đối chiếu: configs/ ; results/tables/experiment_comparison.csv.

### Slide 9. B5-balanced có validation Macro-F1 cao nhất

B5 đạt mean validation Macro-F1 cao nhất 0,6926. Bar biểu diễn mean, SD của từng cấu hình có trong bảng Word/CSV. B3-128 và T20 quanh 0,6724 nhưng thấp hơn B5. Rule chọn xét Macro-F1 trước, nếu gần top dưới 0,005 mới dùng tham số rồi thời gian. B5 thắng nên chưa cần tie-break. Seed demo 3407 có Macro-F1 validation trung vị, không chọn seed tốt nhất trên test. Các giá trị trong biểu đồ được làm tròn sáu chữ số thập phân; CSV giữ độ chính xác gốc.

Nguồn đối chiếu: results/tables/validation_summary.csv ; results/metrics/final_selection.json.

### Slide 10. T60 tăng thời gian nhưng không thêm token thật

Phân bố token sau P0 có median 20, max train 34. T60 không thêm ngữ cảnh so với T40 nên kết quả validation bằng nhau từng seed. Time tăng 42% là cost phần đệm ở triển khai hiện tại. H64 nhanh nhất nhưng mean Macro-F1 gần C0, H128 có chất lượng nhỉnh hơn và ít thời gian hơn H196. Chưa có benchmark latency hoặc peakRAM theo config, số trên là wall time training gồm tính metrics, không phải inference time. Các giá trị trong biểu đồ được làm tròn sáu chữ số thập phân; CSV giữ độ chính xác gốc.

Nguồn đối chiếu: results/tables/validation_summary.csv ; results/figures/F05_train_lengths.png.

### Slide 11. Checkpoint demo ở epoch 16 trong 20 epoch

Đường F1 được tính ở chế độ inference toàn train/validation, không trung bình minibatch. Checkpoint 16 đạt validation Macro-F1 tốt nhất của seed 3407. Accuracy trong fit có dropout còn validation tắt, nên hai đường không cùng chế độ. B5 train loss còn có class weights, vì vậy không so trị số loss với C0 để kết luận model tốt hơn. Xem ba hình loss, Accuracy, F1 riêng trong Word cho phân tích hội tụ đầy đủ. Các giá trị trong biểu đồ được làm tròn sáu chữ số thập phân; CSV giữ độ chính xác gốc.

Nguồn đối chiếu: models/runs/B5-balanced/seed-3407/history.csv ; model_summary.txt.

### Slide 12. B5 đạt test Macro-F1 trung bình 0,6914

B5 đạt test Accuracy trung bình 75,18%, SD 1,34 điểm phần trăm; Macro-F1 trung bình 0,6914, SD 0,0158; Weighted-F1 trung bình 0,7568. C0 đạt Macro-F1 trung bình 0,6515, nên chênh lệch khoảng 0,0399. Đây là trung bình ba training seeds trên cùng split, không phải ensemble. Ba run chưa chứng minh ý nghĩa thống kê trên những split hoặc miền khác. Checkpoint demo riêng đạt Accuracy 75,38% và Macro-F1 0,6869. Confusion matrix tiếp theo thuộc checkpoint này. Các giá trị trong biểu đồ được làm tròn sáu chữ số thập phân; CSV giữ độ chính xác gốc.

Nguồn đối chiếu: results/tables/test_summary.csv ; results/metrics/B5-balanced_seed-3407_test.json.

### Slide 13. Neutral thường nhầm negative, positive nhầm neutral

Hàng của confusion matrix là nhãn thật, cột là dự đoán. Negative đúng 1.153 trên 1.368; neutral đúng 254 trên 448; positive đúng 213 trên 333. F1 lần lượt là 0,8481, 0,5342 và 0,6783. Khi so mean C0 với B5 qua ba seeds, class weighting giảm F1 negative từ 0,8596 xuống 0,8438, tăng neutral từ 0,4754 lên 0,5410 và positive từ 0,6196 lên 0,6895. Sự đánh đổi theo lớp giúp giải thích tại sao Accuracy ít tăng nhưng Macro-F1 cải thiện. Phân biệt các số của checkpoint demo trên hình với trung bình ba seeds. Các giá trị trong biểu đồ được làm tròn sáu chữ số thập phân; CSV giữ độ chính xác gốc.

Nguồn đối chiếu: results/tables/final_per_class.csv ; results/tables/baseline_final_per_class.csv ; results/metrics/B5-balanced_seed-3407_test.json.

### Slide 14. Lỗi chứa cảm xúc pha trộn và thiếu ngữ cảnh

Tweet 9332 dùng awesome ở đầu, nhưng WTH và sự cố gợi ý mỉa mai. Tweet 1537 khen hãng khác trong khi sentiment gán cho hãng được đề cập là negative; P0 làm mất tên hãng trong mentions. Đây là giả thuyết dựa trên văn bản, chưa có attribution hoặc ablation cleaning để xác định nguyên nhân. Trợ lý đã đọc 85 case raw/clean; nhóm chưa có hai người đánh giá độc lập. Accuracy của nhóm tweet dài cao hơn nhưng Macro-F1 thấp và negative chiếm đa số, nên chưa kết luận tweet dài dễ hơn. Giữ nhãn dataset khi tính metric, kể cả case có nhãn đáng xem lại.

Nguồn đối chiếu: results/predictions/assistant_error_review.csv ; results/predictions/manual_error_review.csv ; results/tables/error_slices.csv.

### Slide 15. Demo tải đúng checkpoint đã đánh giá

Chuyển sang demo local, nhập câu cảm ơn như ảnh hoặc dùng một testcase đã lưu. Demo tải model và tokenizer đã đóng băng. Ba giá trị hiển thị là output softmax chưa hiệu chỉnh; giá trị 98% không đảm bảo xác suất dự đoán đúng ngoài thực tế là 98%. Có thể thử một case sai để trình bày hạn chế. Nếu runtime chưa sẵn trên máy trình chiếu, dùng ảnh chụp request thật đã kiểm tra này và nói rõ đang trình bày ảnh. Demo chưa được hosting public. Hướng dẫn khởi động có trong README.

Nguồn đối chiếu: airline_rnn/demo.py ; results/metrics/demo_verification.json ; results/figures/F23_demo.png.

### Slide 16. Kết quả có thể truy vết, neutral còn là hạn chế

Kết quả chính gồm quy trình có thể chạy lại và số liệu thực đo. Cần đặt 90% của nguồn LSTM binary và 75% của RNN ba lớp trong các điều kiện đánh giá khác nhau, đồng thời giữ đầy đủ các training seeds. Class weighting cải thiện các lớp ít mẫu trong miền đã khảo sát, với sự đánh đổi ở negative. Bước tiếp theo là nhóm rà 85 case, điền đóng góp thành viên, chạy lại trên môi trường thứ hai và thiết kế holdout mới nếu khảo sát thêm. LSTM, GRU và BERT chỉ thuộc một phần mở rộng riêng nếu được thực hiện. Xin cảm ơn và sẵn sàng trao đổi.

Nguồn đối chiếu: reports/final/BaoCao_Airline_RNN.docx ; README.md ; reports/project_status.md.

## Demo trong 70 giây

1. Khoảng 10 giây: giới thiệu demo dùng đúng model và preprocessing đã đánh giá, có ba đầu ra softmax.
2. Khoảng 20 giây: nhập `Thank you for the helpful service!`, bấm phân tích. Request kiểm tra trả positive, probabilities negative 0,003222, neutral 0,009164, positive 0,987615. Số hiển thị trên giao diện làm tròn 0,3% / 0,9% / 98,8%; không gọi 98,8% là độ chính xác của mô hình.
3. Khoảng 25 giây: nhập case test **row_id 2856** dưới đây. Nhãn dataset negative nhưng checkpoint dự đoán positive, score 0,9266. Mục đích là trình bày hạn chế cùng ví dụ đúng, không chọn thêm cấu hình từ test.
4. Khoảng 15 giây: chỉ vào ba thanh xác suất và kết luận score cao vẫn có thể sai. Nếu không mở được server trên máy trình chiếu, trình bày ảnh F23 và bảng request đã lưu, nói rõ đây là kết quả request trước buổi trình bày.

Case 2856, giữ nguyên câu để so đúng saved prediction:

> @united thanks for having ground crews that are surprised when flights arrive. #beingsuckontarmacsucks!

Case mỉa mai ứng viên row_id 9332 có thể dùng khi hỏi đáp, không cần đưa thêm vào 70 giây:

> @USAirways awesome... Doors close in 2 minutes, flight leaves in 17 minutes... And the plane just got here. WTH?

Nhãn dataset negative, dự đoán positive, score 0,7395. Trợ lý nhận xét đây là ứng viên mỉa mai; nhóm chưa có hai người xác minh độc lập. Các request tập demo đã chạy và khớp predictions lưu, bằng chứng ở `results/metrics/presentation_demo_rehearsal.json`.

## Câu hỏi phản biện và câu trả lời gợi ý

**1. Vì sao không đạt 90% như tiêu đề notebook?** Nguồn dùng LSTM, hai lớp, tập dữ liệu được lọc mạnh và protocol khác. Output lịch sử 90,32% không thuộc mô hình ba lớp RNN của nhóm. A1 là baseline chuyển thể, test Accuracy 83,80% trong run duy nhất; B5 ba lớp đạt mean Accuracy 75,18% qua ba seeds. Không dùng chênh lệch hai con số để kết luận kiến trúc nào tốt hơn.

**2. Nhóm có thay mô hình chính bằng LSTM không?** Không. Tất cả 22 run đã thực thi dùng RNN. LSTM chỉ có trong file nguồn tham khảo được audit; chưa chạy comparator LSTM/GRU/BERT.

**3. Vì sao chọn Macro-F1?** Ba lớp mất cân bằng. Majority có Accuracy 63,66% nhưng Macro-F1 0,2593. Macro-F1 cho mỗi lớp trọng số bằng nhau; báo cáo vẫn giữ Accuracy, Weighted-F1, Precision/Recall/F1/Support theo lớp.

**4. Làm thế nào hạn chế leakage?** Nhóm trùng ID/raw text chuẩn hóa/clean text cùng split, vocab fit trên train, class weights lấy từ train, checkpoint/cấu hình chọn bằng validation. Test chỉ mở sau frozen selection. Định nghĩa grouping này không chứng minh dữ liệu độc lập về ngữ nghĩa hoặc hội thoại.

**5. B5 cải thiện gì?** Mean test Macro-F1 từ C0 0,6515 lên B5 0,6914 trên cùng split. Mean F1 neutral từ 0,4754 lên 0,5410, positive từ 0,6196 lên 0,6895; negative giảm từ 0,8596 xuống 0,8438. Đây là sự đánh đổi theo lớp trong ba training seeds đã khảo sát.

**6. Ba seeds có chứng minh ý nghĩa thống kê không?** Chưa. Ba lần khởi tạo/huấn luyện dùng cùng một split; mean và SD mô tả biến thiên training, không phải ba split, confidence interval hoặc ensemble.

**7. Vì sao T40 và T60 có kết quả giống nhau?** Cả hai giữ trọn chuỗi thật trong dữ liệu hiện có; phần tăng chỉ là PAD và được masking. T60 tăng median wall time khoảng 42%. Đối chứng này không chứng minh khả năng học phụ thuộc dài hạn.

**8. Checkpoint và Early Stopping có giống nhau không?** Checkpoint lưu mọi cải thiện validation Macro-F1, tie theo validation loss. Early Stopping dùng mốc riêng với min_delta 0,001 và patience 5. Model demo ở best epoch 16 của seed 3407 dù run có 20 epoch. Seed này được chọn theo validation trung vị trước test.

**9. Sarcasm hoặc cleaning có gây lỗi không?** Có các trường hợp gợi ý nhưng chưa xác định nhân quả. 85 case có nhận xét riêng của trợ lý; human review còn trống. Muốn kết luận cần hai người review và preprocessing ablation với protocol/holdout mới phù hợp; không sửa nhãn dataset để tăng metric hiện có.

**10. Tại sao tweet dài Accuracy cao nhưng Macro-F1 thấp?** Class mix khác: nhóm dài có tỷ lệ negative cao. Đây là liên hệ quan sát được, chưa chứng minh tweet dài dễ hơn hay RNN xử lý dài tốt hơn.

**11. Xác suất softmax 98,8% có nghĩa đúng 98,8% không?** Chưa có calibration, nên đó chỉ là output của model. Ví dụ 2856 có positive 92,66% vẫn sai so nhãn dataset. Dataset tiếng Anh hàng không Mỹ chưa hỗ trợ kết luận cho tiếng Việt hoặc miền khác.

**12. Nhóm có chạy trên GPU/Colab không?** Kết quả bàn giao chạy CPU macOS arm64, RAM 16 GiB, không có TensorFlow GPU. Tổng training wall time khoảng 19,91 phút gồm callback metrics. Có hướng dẫn Colab/Drive, chưa chạy kiểm chứng cloud; không gọi số wall time này là GPU time hay inference latency.

## Rút gọn nếu chỉ có 8 phút

Dùng 10 slide: 1, 2, 3, 4, 5, 8, 9, 12, 13, 15. Giữ ghi chú A1/A2, hội tụ và lỗi cho hỏi đáp. Không bỏ slide audit nguồn hoặc nhầm mean ba seeds với một checkpoint. Kết thúc bằng một câu về khả năng tái lập và giới hạn neutral.

## Trình bày website có tám khâu

Website hiển thị logo trường, tên trường/khoa, đề tài/giảng viên và ba thành viên. Khi nhập câu, chỉ lần lượt vào tám khâu: tiếp nhận văn bản, cleaning, token hóa, ánh xạ ID, padding/masking, forward pass RNN, đọc softmax và argmax. Chọn “Xem chuỗi đầy đủ và mask” hoặc “Xem giá trị vector thật” nếu cần giải thích sâu. Thời gian là của request hiện tại; giá trị vector không phải attribution của từng từ.
