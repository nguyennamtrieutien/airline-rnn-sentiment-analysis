# EXPERIMENTAL PLAN: SIMPLE RNN CHO AIRLINE SENTIMENT ANALYSIS

**Trạng thái:** bản kế hoạch để nhóm duyệt, ngày 05/10/2026. Chưa triển khai notebook, chưa huấn luyện mô hình và chưa sinh kết quả Accuracy/F1 của nhóm. Những con số kết quả được ghi dưới đây đều được đánh dấu là output có sẵn của nguồn Kaggle. Các con số dataset là thống kê kiểm tra trực tiếp file nguồn, không phải kết quả mô hình.

**Tên đề tài đề xuất:** “Phân tích cảm xúc tweet về hãng hàng không bằng RNN: tái dựng pipeline và thực nghiệm có kiểm soát”.

## 1. Phát hiện quyết định phạm vi dự án

Notebook được chỉ định có tiêu đề RNN nhưng **lớp hồi quy thực tế là LSTM, không phải RNN**. Nó cũng **bỏ toàn bộ neutral**, rồi bỏ 5.000 tweet negative đầu tiên. Output test đã lưu là **Accuracy 0,9032094, tương đương khoảng 90,32%**, trên bài toán hai lớp đã lọc này.

Vì vậy, hai yêu cầu “giữ đúng kiến trúc gốc” và “mô hình chính phải là RNN” không thể đồng thời được đáp ứng với notebook này. Plan ưu tiên ràng buộc RNN của nhóm và ghi rõ baseline là **bản thích nghi từ notebook gốc**. Không gọi kết quả RNN là tái lập chính xác kết quả LSTM.

| Thành phần | Vai trò và phạm vi | Quyết định trong plan |
|---|---|---|
| Nguồn Kaggle | LSTM, hai lớp; dùng làm nguồn audit và đối chiếu | Đọc mã và output có sẵn; không đưa LSTM vào thực nghiệm chính |
| Experiment A — Reproduce Baseline | Tái dựng pipeline gốc sát nhất trong giới hạn RNN | A1: thay duy nhất loại lớp hồi quy bằng RNN, ghi riêng các thay đổi tương thích môi trường/seed; A2: baseline RNN hai lớp có protocol sạch |
| Experiment B — Controlled Experiments | Khảo sát RNN bằng các thay đổi một yếu tố | Dùng baseline C0 ba lớp và một split cố định; giữ neutral để đáp ứng phần phân tích lỗi ba lớp |
| Tái lập nguyên trạng LSTM | Mới trả lời được câu hỏi “cùng dataset, preprocessing và kiến trúc có đạt kết quả nguồn không?” | Chỉ là đề xuất mở rộng, cần nhóm chấp nhận phạm vi LSTM phụ lục trước khi triển khai; không nằm trong plan mặc định |

**Câu hỏi nghiên cứu được sử dụng:**

1. **A:** Với pipeline gần notebook gốc và RNN, kết quả thực tế là bao nhiêu; khác biệt kiến trúc/protocol giới hạn việc đối chiếu 90,32% như thế nào?
2. **B:** Trên bài toán ba lớp và protocol sạch, sequence length, vocabulary, hidden units, SpatialDropout và class weighting ảnh hưởng đến validation Macro-F1, chất lượng từng lớp và chi phí huấn luyện ra sao?

Nếu môn học bắt buộc tái lập *đúng nguyên trạng* notebook, cần duyệt lại phần A hoặc tìm một notebook gốc thực sự dùng RNN. Không được âm thầm đổi nghĩa “reproduce”.

## 2. Nguồn đã nghiên cứu và cách kế thừa báo cáo mẫu

- [Notebook Kaggle của Chibu](https://www.kaggle.com/code/chibuzorokocha/airline-sentiment-analysis-90-accuracy-using-rnn), Version 1; bản hiển thị có scriptVersionId 120745701. Đã đối chiếu mã tải qua API công khai với code và output hiển thị. Bản mã tải về không chứa output; các chỉ số nguồn trong plan được đọc từ bản hiển thị.
- [Dataset Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment), chủ nguồn CrowdFlower; đã kiểm tra trực tiếp `Tweets.csv` trong gói dataset công khai.
- [Báo cáo mẫu của Nguyễn Nam Triều Tiên](/Users/johnnycu/Downloads/2611323_NguyenNamTrieuTien_CuoiKy_ThiGiacMayTinh.pdf), 34 trang PDF: Mở đầu, Chương 1 lý thuyết, Chương 2 giải pháp, Chương 3 thực nghiệm, Chương 4 demo, Kết luận và tài liệu tham khảo.

Plan kế thừa mức độ trình bày của mẫu: sơ đồ pipeline và kiến trúc; bảng cấu hình; các nhóm thực nghiệm; loss/accuracy/Macro-F1 theo epoch; confusion matrix và bảng từng lớp; ví dụ đúng/sai; bảng đối chiếu nguồn; hạn chế; demo dùng cùng checkpoint. Trong mẫu, các mục 3.3–3.6 minh họa rõ cách gắn bảng, hình với nhận xét.

**Điểm cần điều chỉnh về phương pháp:** phần pilot ở mục 3.5 của báo cáo mẫu có ưu tiên kết quả test khi chọn cấu hình. Dự án này chỉ chọn bằng validation. Báo cáo mẫu là tài liệu tham khảo cấu trúc và mức độ chi tiết; các hướng dẫn, lựa chọn mô hình và kết luận trong đó không thay thế yêu cầu của nhóm.

## 3. Audit notebook gốc — thông tin đã xác minh

### 3.1. Dataset, input, output và tập dữ liệu thực sự được dùng

| Thuộc tính | Thông tin xác minh được | Căn cứ/trạng thái |
|---|---|---|
| Dataset | Twitter US Airline Sentiment, `crowdflower/twitter-airline-sentiment` | Metadata notebook và file dataset |
| File đọc | `Tweets.csv`; đường dẫn trong mã là `../input/Tweets.csv` | Code cell đọc dữ liệu; đường dẫn hiện tại cần cấu hình lại |
| Số dòng/cột file nguồn kiểm tra | 14.640 dòng, 15 cột | Đếm trực tiếp file CSV hiện tải |
| Input mô hình | Nội dung tweet tiếng Anh từ cột `text`, chuyển thành chuỗi token ID | Code |
| Target | Cột `airline_sentiment` | Code |
| Nhãn dataset gốc | negative, neutral, positive | File CSV |
| Nhãn thực nghiệm nguồn | negative và positive; loại neutral | Code lọc dữ liệu |
| Output nguồn | Hai xác suất softmax; nhãn lấy argmax | Dense đầu ra và cell inference |
| Cột dùng cho học mô hình | `text` và `airline_sentiment` | Code lấy hai cột |
| Cột dùng EDA | `airline`, dùng countplot | Không được đưa vào input mô hình |
| Cột khác | Không được dùng trong mô hình | Không có code trích xuất thêm feature |
| Encode label | negative → 0; positive → 1 | Vòng lặp thủ công |
| Bất cập encode | Vòng lặp cố định 6.541 thay vì lấy độ dài dữ liệu | Cần kiểm tra kích thước; thay bằng mapping tương đương khi triển khai và ghi thay đổi |

Phân bố đã kiểm tra từ file nguồn và suy ra theo đúng thao tác lọc của code:

| Giai đoạn | Negative | Neutral | Positive | Tổng |
|---|---:|---:|---:|---:|
| CSV gốc | 9.178 (62,69%) | 3.099 (21,17%) | 2.363 (16,14%) | 14.640 |
| Sau bỏ neutral | 9.178 | 0 | 2.363 | 11.541 |
| Sau bỏ 5.000 negative đầu tiên | 4.178 (63,87%) | 0 | 2.363 (36,13%) | 6.541 |

Dataset có 6 hãng: Virgin America 504, United 3.822, Southwest 2.420, Delta 2.222, US Airways 2.913, American 2.759. Đây là phân bố CSV gốc, không phải phân bố sau thao tác bỏ 5.000 negative.

SHA-256 của `Tweets.csv` đã kiểm tra: `ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e`. Triển khai phải kiểm tra checksum và row order; chưa khẳng định file lịch sử của tác giả có cùng checksum vì notebook không công bố checksum lịch sử.

### 3.2. Preprocessing và sequence representation

| Thành phần | Notebook thực sự thực hiện | Giới hạn/điểm cần kiểm tra |
|---|---|---|
| Xử lý mention đầu câu | Gọi `lstrip` với một chuỗi chứa tên nhiều hãng, rồi `rstrip('@')` | `lstrip` xóa theo tập ký tự, không xóa chính xác một prefix/tên hãng; có nguy cơ cắt mất ký tự từ đầu tweet |
| Lowercase | Có | Sau thao tác strip |
| Regex cleaning | Pattern `[^a-zA-z0-9\s]`, thay ký tự khớp bằng chuỗi rỗng | Giữ chữ/số/khoảng trắng theo pattern; `A-z` có phạm vi ASCII rộng hơn chữ cái nên không nên mô tả là regex chữ cái chuẩn |
| URL/mention | Không có regex URL/mention chuyên biệt | Xóa dấu không đồng nghĩa với loại bỏ toàn bộ URL hay username |
| Stopwords | Không có bước loại stopword trong code | Markdown có đề cập chung, nhưng không được xem là bước đã triển khai |
| Stemming/lemmatization | Không có | Không tự bổ sung vào A1 |
| Emoji, dấu câu, apostrophe | Nhiều ký tự bị xóa bởi regex | Cần lưu before/after; có thể mất tín hiệu cảm xúc hoặc biến đổi contractions |
| Tokenization | Keras Tokenizer, word-level, split bằng khoảng trắng | Không phải subword tokenizer |
| Vocabulary cap | `num_words=4000` | Tổng `word_index` chưa xác định; cap không có nghĩa CSV có đúng 4.000 từ khác nhau |
| Fit tokenizer | Fit toàn bộ dữ liệu đã lọc, trước train/test split | Vocabulary biết cả văn bản validation/test |
| OOV | Không đặt OOV token | Theo hành vi Tokenizer cần kiểm tra đúng phiên bản: token không biết/vượt cap bị bỏ khỏi sequence |
| Token IDs | PAD dùng 0; từ có ID nhỏ hơn cap được giữ | Với cap 4.000, không nên diễn giải là 4.000 từ nội dung cộng PAD |
| Padding huấn luyện | `pad_sequences` không truyền `maxlen` | Độ dài theo sequence dài nhất của toàn bộ dữ liệu được token hóa; số cụ thể chưa có output shape để xác minh trực tiếp |
| Padding/truncation | Không truyền hướng; theo mặc định API là pre/pre, giá trị PAD 0 | Ghi riêng đây là hành vi mặc định API, kiểm tra phiên bản nguồn |
| Inference trong nguồn | Truyền độ dài 31 tường minh | Output ví dụ inference chạy được; không thay thế việc kiểm tra `X_train.shape[1]` lúc tái dựng |

Không được đưa Word2Vec, GloVe, cross-validation hay Early Stopping vào phần mô tả “đã dùng trong nguồn”: chúng xuất hiện trong phần giới thiệu chung nhưng code không thực hiện.

Tài liệu kiểm tra hành vi API: [Tokenizer](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/text/Tokenizer) và [pad_sequences](https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences). Đây là tài liệu API hiện có, không phải bằng chứng phiên bản TensorFlow/Keras lịch sử của notebook.

### 3.3. Split, kiến trúc và huấn luyện nguồn

| Thuộc tính | Giá trị trong nguồn | Trạng thái |
|---|---|---|
| Train/test split | 70%/30%; shuffle; stratify theo nhãn; random_state=1 | Code xác minh |
| Validation | 20% của phần train qua `validation_split` | Không chia stratified riêng |
| Kích thước thực tế | Train dùng để cập nhật trọng số 3.662; validation 916; test 1.963 | Train/val từ log; test = 6.541 − 4.578 |
| Tỷ lệ trên tổng dữ liệu đã lọc | Xấp xỉ 56% train / 14% validation / 30% test | Không phải 70% train thực tế cộng thêm 20% validation độc lập |
| Embedding | Input dimension 4.000; output dimension 128 | Không load pretrained embedding; các default khởi tạo/trainable cần ghi rõ khi dựng lại |
| SpatialDropout1D | 0,5 | Code xác minh |
| Lớp hồi quy | **LSTM**, 196 units | Không phải RNN |
| Dropout trong lớp hồi quy | Input dropout 0,3; recurrent dropout 0,3 | Code xác minh |
| Sau lớp hồi quy | Dropout 0,2; Dense 100/ReLU; Dropout 0,4 | Code xác minh |
| Đầu ra | Dense 2/softmax | Bài toán hai lớp |
| Activation hồi quy | Không đặt tường minh | Tài liệu LSTM hiện tại mặc định tanh và recurrent sigmoid; phải kiểm tra phiên bản lịch sử trước khi ghi là cấu hình lịch sử xác định |
| Loss | Sparse categorical cross-entropy | Label số nguyên, không one-hot |
| Optimizer | Adam bằng tên optimizer | Code xác minh |
| Learning rate | Không đặt tường minh | **Chưa xác định chắc chắn phiên bản chạy gốc/default thực tế**; tài liệu Adam hiện tại mặc định 0,001 |
| Batch size | 32 | Code xác minh |
| Epoch | 20 | Code và log xác minh |
| Metric trong compile | Accuracy | Không có F1/Precision/Recall trong compile |
| Early Stopping/checkpoint | Không có trong code | Evaluate mô hình sau epoch cuối |
| Seed | Split seed=1; không đặt global seed cho NumPy/TensorFlow | Không bảo đảm tái tạo cùng initialization/dropout |
| Hardware/version | Output có đường dẫn Python 3.6; metadata hiện tải cho biết GPU tắt | CPU/GPU/RAM và phiên bản TensorFlow/Keras của output lịch sử không được công bố đầy đủ |

Theo [Keras fit](https://keras.io/api/models/model_training_apis/), `validation_split` lấy phần cuối của các array trước bước shuffle bên trong fit; không bảo đảm stratification của validation. Split train/test trước đó có shuffle nhưng vẫn không tương đương một validation split stratified tường minh.

### 3.4. Kết quả nguồn đã xác minh — không phải kết quả của nhóm

| Chỉ số | Output có sẵn của notebook |
|---|---:|
| Test Accuracy | 0,9032094 ≈ **90,32%** |
| Test loss | 0,6523001417498685 |
| Train Accuracy epoch 20 | 99,62% |
| Train loss epoch 20 | 0,0140 |
| Validation Accuracy epoch 20 | 90,61% |
| Validation loss epoch 20 | 0,5855 |
| Validation Accuracy cao nhất trong log | 92,58%, epoch 11 |
| Validation loss thấp nhất trong log | 0,2383, epoch 3 |
| Precision/Recall/F1/Macro-F1/Weighted-F1 | Không có số báo cáo tường minh trong các output đã đọc |
| Confusion matrix | Có cell tạo hình hai lớp |

Các chỉ số trên hỗ trợ nhận xét rằng log có chênh lệch train/validation và validation loss tăng sau giai đoạn đầu; không đủ để kết luận nguyên nhân duy nhất là overfitting hay gán nguyên nhân cho một lớp dropout cụ thể. Các lựa chọn checkpoint khác nhau có thể tạo kết quả khác; nguồn đang evaluate epoch cuối, không phải epoch có validation accuracy cao nhất.

### 3.5. Rủi ro leakage và diễn giải “90%”

| Vấn đề | Bằng chứng | Ảnh hưởng và cách xử lý trong plan |
|---|---|---|
| Vocabulary contamination | Fit tokenizer trước khi split | Test text tham gia tạo từ vựng/tần suất; không phải nhãn test trực tiếp được đưa vào loss. A1 giữ để mô tả pipeline nguồn; A2/C0 fit train-only |
| Sequence length dùng toàn bộ corpus | Padding trước split, không đặt maxlen | Thông tin độ dài test/validation tham gia xác định representation; A2 lấy độ dài từ train; C0 dùng độ dài đã định trước |
| Duplicate leakage | Code không kiểm tra duplicate tweet ID/text | **Nguy cơ cần kiểm tra**, chưa kết luận số duplicate của nguồn hay mức tăng metric |
| Sampling bias | Bỏ negative theo thứ tự, không random | Có thể làm thay đổi phân bố hãng/thời gian/nội dung; kiểm tra bảng trước/sau, không mặc định việc này nâng accuracy |
| Task scope bị thu hẹp | Bỏ neutral và phần lớn negative | 90,32% không phải kết quả trên toàn bộ 14.640 tweet với ba lớp |
| Loại mô hình | Code dùng LSTM | Không được trình bày 90,32% là thành tích RNN |
| Accuracy | Loss sparse + softmax hai lớp; inference argmax | Không thấy bằng chứng sai công thức accuracy trong code đã đọc; vấn đề chủ yếu là phạm vi và protocol |
| Khả năng tái lập | Chỉ cố định split seed | Khác initialization, version và hardware có thể dẫn đến khác kết quả |

**Không tuyên bố “90% là sai” hoặc “leakage chắc chắn làm tăng X%”.** Nguồn có output test tương ứng tiêu đề; phạm vi là LSTM/hai lớp/tập lọc và protocol có giới hạn. Chưa có thực nghiệm sạch để định lượng ảnh hưởng leakage.

## 4. Các phase theo thứ tự triển khai

Các đường dẫn dưới đây là **cấu trúc đầu ra dự kiến**, chưa được tạo trong bước lập plan.

### Phase 0 — Audit notebook gốc

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 0 | Chốt Version 1/scriptVersionId; lưu bản source và bằng chứng output; tách markdown mô tả khỏi code thực hiện; xác nhận LSTM/hai lớp và giới hạn 90%; lập danh sách unknown | Notebook, metadata, output hiển thị, báo cáo mẫu | `reports/source_audit.md`, snapshot nguồn, `source_manifest.json`, bảng khác biệt A1/A2/C0 | Mỗi hyperparameter có căn cứ hoặc ghi chưa xác định; nhóm duyệt cách gọi A là adapted baseline và phạm vi ba lớp của B |

### Phase 1 — Chuẩn bị môi trường và dataset

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 1 | Tạo môi trường Colab; lấy dataset đúng slug; lưu CSV nguyên bản; kiểm tra checksum/row order; khóa dependencies sau smoke test API; ghi phần cứng và version; cấu hình lưu output bền vững | Dataset nguồn, plan đã duyệt | `data/raw/Tweets.csv`, manifest/checksum, environment lock, `environment.json`, hướng dẫn lấy dữ liệu | Nạp được CSV đúng schema/count; nhận diện RNN/Tokenizer/save-load tương thích; môi trường được ghi đầy đủ; chưa huấn luyện toàn bộ |

### Phase 2 — EDA và kiểm tra dữ liệu

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 2 | Thống kê sentiment/airline; kiểm tra thiếu text/label, nhãn ngoài miền, duplicate ID/text, nhãn xung đột, tweet rỗng; so sánh trước/sau lọc nguồn; lập nhóm duplicate; khóa split A2/C0 trước việc fit bất kỳ vocabulary nào | CSV nguyên bản | Dataset profile, bảng class count theo split, duplicate/conflict report, split manifest, hình EDA | Các dòng bị loại/giữ đều có lý do; không trùng nhóm duplicate giữa train/val/test; phân bố lớp từng split được kiểm tra; số liệu train/test thực tế không chỉ là tỷ lệ dự kiến |

EDA phục vụ chọn sequence length/vocabulary chỉ dùng train: histogram/percentile token length, tỷ lệ OOV và tỷ lệ bị truncation. Profile tổng corpus dùng để mô tả dữ liệu, không dùng để chọn hyperparameter từ test. Nội dung/nhãn test khóa đến lần đánh giá cuối.

### Phase 3 — Preprocessing

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 3 | Định nghĩa P-source cho A; định nghĩa P0 cho C0; xử lý Unicode/HTML/mention/URL/contractions theo quy tắc đã ghi; tokenizer train-only cho protocol sạch; padding/truncation và mask thống nhất; mapping nhãn bất biến | Split manifests, raw text, cleaning specification | `preprocessing_spec.md`, before/after samples, tokenizer, label map, representation config, train token statistics | A2/C0 không có dữ liệu val/test trong fit; A1 ghi rõ contamination giữ từ nguồn; PAD/OOV/label có mapping rõ; không bỏ token phủ định theo stopword; inference dùng đúng artifact, không refit tokenizer |

### Phase 4 — Reproduce RNN baseline

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 4 | Dựng A1 RNN theo pipeline nguồn; chạy kiểm tra nhỏ về shape/label/loss/save-load; train A1 đủ 20 epoch; dựng A2 protocol sạch; dựng C0 ba lớp làm mốc B | A1/A2/C0 configs, dữ liệu tương ứng | Model summaries/architecture, run configs, history, last checkpoint A1, best checkpoints A2/C0, timing | Tất cả có RNN; loss hữu hạn; output 2 hoặc 3 lớp đúng scope; save/load giữ prediction; mọi khác biệt với nguồn được ghi; không đặt tiêu chí “phải đạt 90%” |

A1 là đối chứng pipeline có giới hạn, không phải mô hình cuối được dùng để tuyên bố chất lượng tổng quát. Không chuyển weights/tokenizer A1 sang A2/C0. Các quyết định B đã khóa trước khi đọc kết quả test A1; nếu cần hiển thị test A1, hoãn đến Phase 7 khi toàn bộ cấu hình so sánh đã khóa.

### Phase 5 — Đánh giá baseline

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 5 | Kiểm tra hội tụ train/val; tính Accuracy, macro/weighted và per-class P/R/F1 trên validation; majority-class reference; kiểm tra support/label order; ghi giới hạn khi đối chiếu nguồn | Histories, checkpoint A2/C0, validation | `validation_metrics.json`, classification report, validation confusion matrix, learning curves | Macro-F1 tính trên toàn val; không lỗi đảo label; không dùng test chọn epoch/config; C0 vượt qua kiểm tra pipeline kể cả metric chưa tốt |

### Phase 6 — Controlled experiments

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 6 | Chạy ma trận B sau khi C0 chạy được; mỗi variant chỉ đổi một yếu tố so với C0; khởi tạo mới mỗi run; cùng split/seeds/budget/callbacks; so sánh validation trung bình và SD | Frozen matrix, C0, ba training seeds | Mỗi run có config/history/checkpoint/timing; `validation_comparison.csv`, biểu đồ ablation | Không cấu hình thiếu log; không dùng test; kết luận giới hạn theo mức đã thử; chọn ứng viên theo quy tắc validation đã đăng ký |

### Phase 7 — Train mô hình cuối

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 7 | Khóa C0 và cấu hình B* thắng validation; chọn checkpoint dùng demo theo quy tắc seed; dùng checkpoint đã huấn luyện hoặc train lại đúng config nếu cần kiểm tra tái lập; sau khi khóa mới đánh giá test cho A1/A2/C0/B* theo benchmark tương ứng | Frozen configs, validation table, checkpoints, test manifest | `final_selection.json`, best model, test metrics/report/CM/predictions, hashes | Test chỉ được mở trong một đợt đánh giá cuối; A1 ghi giới hạn protocol nguồn; không sửa config sau khi thấy test; checkpoint cuối chọn bằng val; demo dùng cùng artifact đánh giá |

Mặc định **không gộp train+validation để retrain**, vì sẽ mất validation dùng chọn checkpoint. Cũng không train lại chỉ để tìm seed có test tốt hơn. Nếu có nhiều checkpoint của cùng cấu hình từ ba seed, báo cáo trung bình ± SD; checkpoint demo chọn seed có validation Macro-F1 trung vị, trước khi xem test. Lặp lại train seed đã định chỉ là kiểm tra tái lập, không phải thêm ứng viên để chọn.

### Phase 8 — Error analysis

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 8 | Phân tích nhầm lớp và độ dài; xem đúng/sai/confidence thấp; rà negation/sarcasm/mixed sentiment/cleaning; gắn nhãn nhóm lỗi có tiêu chí và bằng chứng | Test predictions cố định, raw/clean text, tokenizer statistics | `error_analysis.md`, `error_cases.csv`, slice metrics/support, bảng mẫu minh họa | Mọi nhận xét có số liệu hoặc ví dụ thật; sarcasm có kiểm tra ngữ cảnh; không biến test errors thành vòng tuning trên chính test |

### Phase 9 — Demo/inference

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 9 | Notebook inference bắt buộc; demo nhỏ tùy chọn; nhập tweet, trả nhãn và xác suất; cho xem raw→clean→tokens; xử lý input rỗng/toàn OOV/quá dài | Frozen checkpoint, tokenizer, preprocessing config, label map | Inference examples, demo local, screenshot giao diện, hướng dẫn sử dụng | Không refit; xác suất tổng ≈1; class order đúng; prediction demo khớp notebook trên cùng input; công bố phạm vi tiếng Anh/airline |

### Phase 10 — Chuẩn bị số liệu cho báo cáo

| Phase | Việc cần làm | Đầu vào | Đầu ra cần lưu | Tiêu chí hoàn thành |
|---|---|---|---|---|
| 10 | Xuất bảng/hình trực tiếp từ metrics; soạn chương Thực nghiệm; đối chiếu nguồn theo scope; viết limitations; chuẩn bị slide và checklist tái chạy; rà mọi claim | All frozen artifacts, source audit, error analysis | Tables CSV/Markdown, figures PNG/PDF, report chapter outline/text, presentation outline, README | Số trong báo cáo/slide khớp artifact; mỗi hình có split/run/seed/caption; không gọi B ba lớp là tái lập 90%; một thành viên khác có thể Run All theo README |

## 5. Protocol thực nghiệm và representation

### 5.1. Ba cấu hình nền

| Thành phần | A1 — pipeline nguồn thích nghi RNN | A2 — binary baseline sạch | C0 — baseline B ba lớp |
|---|---|---|---|
| Population | Bỏ neutral; bỏ 5.000 negative đầu theo source row order | Cùng tập binary đã lọc, kiểm soát dữ liệu không hợp lệ/duplicate | Giữ cả ba lớp và mọi dòng hợp lệ; không bỏ 5.000 negative |
| Cleaning | P-source: giữ hành vi strip/lowercase/regex nguồn | P-source để gần A1; không tự sửa cleaner như một phần vô hình | P0 được mô tả bên dưới |
| Tokenizer fit | Toàn population như nguồn; ghi rõ contamination | Chỉ train | Chỉ train |
| OOV | Không đặt như nguồn | Không đặt để gần A1; ghi tỷ lệ token bị bỏ | OOV token riêng; không bỏ từ không biết |
| Sequence length | Giá trị suy ra từ pipeline nguồn; kiểm tra khi triển khai | Max sequence length tính từ train; pre-padding/pre-truncation | 40 token; post-padding/post-truncation; PAD được mask |
| Label mapping | negative=0, positive=1 | negative=0, positive=1 | negative=0, neutral=1, positive=2 |
| Recurrent layer | RNN 196, tanh; input/recurrent dropout 0,3/0,3 | Như A1 | Như A1 |
| Embedding và dense head | 4.000×128; SpatialDropout 0,5; Dropout 0,2; Dense 100/ReLU; Dropout 0,4; output 2/softmax | Như A1 | Như A1 nhưng output 3/softmax; mask PAD |
| Split | Nguồn: train/test seed=1 + validation_split 0,2 | Explicit, xấp xỉ 56/14/30; seed=1; stratified có kiểm soát nhóm duplicate | Explicit, mục tiêu 70/15/15; split seed=42; stratified có kiểm soát nhóm duplicate |
| Optimization | Adam, default source chỉ chốt nếu xác minh được; nếu dùng 0,001 phải ghi là lựa chọn triển khai | Adam learning rate 0,001, ghi tường minh | Adam learning rate 0,001; global gradient norm clipping=1,0 |
| Training | 20 epoch, batch 32; không ES; checkpoint cuối để đối chiếu pipeline nguồn | Max 20 epoch, batch 32; best val Macro-F1; ES | Như A2 |
| Training seeds | 42: lựa chọn của nhóm, nguồn không có global seed | 42, 2026, 3407 ở plan khuyến nghị | 42, 2026, 3407 |
| Ý nghĩa so sánh | Mô tả baseline thích nghi; không gọi là exact reproduction | Đối chiếu protocol sạch có ghi thay đổi gộp; không suy ra riêng ảnh hưởng mỗi sửa đổi | Mốc cố định cho mọi ablation B |

**Không so trực tiếp A2 với C0 để kết luận “mô hình tốt hơn”:** population, số lớp, representation và split khác nhau. Không giải thích chênh lệch A1/A2 như hiệu ứng duy nhất của leakage vì nhiều yếu tố protocol thay đổi.

### 5.2. Cleaner P0 đề xuất cho C0/B

P0 là policy được khóa trước B, không phải kết quả tối ưu đã chứng minh:

1. Chuẩn hóa Unicode và HTML entities; lowercase; chuẩn hóa khoảng trắng.
2. URL → token URL; mention → token USER bằng quy tắc nhận diện đầy đủ, không dùng `lstrip` theo tập ký tự.
3. Chuẩn hóa apostrophe; mở rộng contractions theo danh sách cố định, giữ rõ `not`, `no`, `never`. Danh sách và thứ tự thay thế được lưu.
4. Bỏ ký hiệu `#` nhưng giữ nội dung hashtag; giữ số. Các marker phải được tokenize thành token ổn định.
5. Cleaner cơ sở dùng token từ, bỏ dấu câu/emoji còn lại; ghi nhận mất thông tin để phân tích lỗi. Không stemming/lemmatization hoặc loại stopword. Biến thể giữ emoji/dấu câu chỉ ở mở rộng preprocessing ablation.
6. Tweet trở thành chuỗi token rỗng dùng marker EMPTY; không âm thầm xóa val/test vì rỗng hoặc OOV. Lưu số lượng và đánh giá riêng.

Vocabulary C0 có tổng cap 4.000 **kể cả các ID đặc biệt**: PAD=0, OOV=1; các marker khác chỉ chiếm ID trong cap theo mapping được lưu. Tokenizer vocabulary train-only, cùng tokenizer/mapping qua mọi seed của cùng config. Khi đổi vocabulary cap, giữ đúng quy tắc ranking và reserved tokens; không thay luôn cleaner.

### 5.3. Split, duplicate và leakage guard

- Lưu raw data bất biến; dùng row index + tweet ID + text hash cho định danh vì tweet ID cần kiểm tra uniqueness, không mặc định là khóa duy nhất.
- Kiểm tra duplicate theo ID, raw text và text đã chuẩn hóa bằng quy tắc không học từ corpus. Khóa nhóm duplicate trước split.
- Với bản sạch, không để một nhóm duplicate xuất hiện ở nhiều split. Dòng thiếu text/label hoặc nhãn ngoài miền bị loại với log; nhãn xung đột trong nhóm phải được ghi, xử lý theo quy tắc đã duyệt, không chọn nhãn dựa trên kết quả mô hình.
- Chốt quy tắc: loại duplicate hoàn toàn giống nhau; các bản cùng nội dung còn giữ để mô tả đơn vị tweet phải cùng một nhóm. Nhóm nhãn xung đột bị cách ly khỏi benchmark chính và đếm rõ, không bỏ chỉ những mẫu mô hình đoán sai.
- Dùng stratified split nếu dữ liệu không cần group; khi có nhóm còn lại dùng cách chia có kiểm soát group và gần tỷ lệ lớp. Nếu không đạt tỷ lệ 70/15/15 chính xác, báo số thực tế và lý do.
- Với 14.640 dòng và **trước mọi kiểm tra loại dòng/group**, tỷ lệ C0 tương ứng 10.248/2.196/2.196. Đây là số kế hoạch; không điền làm support cuối khi chưa tạo manifest.
- Không dùng `airline_sentiment_confidence`, `negativereason`, gold labels hay các thông tin annotation để làm feature. `airline` chỉ dùng EDA/slice analysis; input chính vẫn là text.
- Không oversample/class-weight cả val/test; mọi weight lấy từ train. Tokenizer, thống kê chọn representation và bất kỳ thao tác học tham số nào đều chỉ fit train.
- Test manifest chỉ dùng khi config/checkpoint policy đã khóa. Không có test metric trong quá trình chọn B.

### 5.4. Seed, checkpoint, training history và chọn mô hình

| Quy tắc | Đề xuất cố định |
|---|---|
| Seed split C0/B | 42; giữ cùng manifest qua mọi run |
| Training seeds | 42, 2026, 3407; cố định Python/NumPy/TensorFlow và seed shuffle/dropout |
| Determinism | Bật deterministic operations khi được hỗ trợ; ghi giới hạn khác hardware/version, không hứa bitwise identical trên mọi máy |
| Budget sạch | Max 20 epoch, batch 32; nếu có underfitting ở mọi config, chỉ sửa budget trước khi chạy lại toàn bộ matrix và ghi revision, không nới riêng ứng viên |
| Checkpoint | Validation Macro-F1 cao nhất; tie chính xác ưu tiên validation loss thấp hơn, sau đó epoch sớm hơn |
| Early Stopping | Theo val Macro-F1, mode max; patience=5, min_delta=0,001; khôi phục best checkpoint theo quy tắc trên |
| History | Epoch, train loss/accuracy, val loss/accuracy/Macro-F1/Weighted-F1, learning rate, epoch time, best epoch, stop reason |
| Train F1 curve | Tính prediction trên train trong inference mode tại cuối epoch; cùng cách với val. Ghi riêng `train_eval_F1` để không nhầm với metric batch khi dropout hoạt động |
| F1 computation | Toàn split, đủ label order cố định; không lấy trung bình F1 của mini-batch; `zero_division=0` và ghi khi lớp không được dự đoán |
| Cấu hình thắng B | Mean validation Macro-F1 qua ba seeds; nếu chênh <0,005, coi là gần nhau theo quy tắc thực dụng, ưu tiên ít parameters hơn, rồi train time trung vị thấp hơn, rồi thứ tự config đã định |
| Ý nghĩa threshold tie | 0,005 là quy tắc ra quyết định đặt trước, không phải kiểm định thống kê hay tuyên bố hai mô hình tương đương |
| Final config | Chọn một config thực sự đã chạy. Không tự ghép “winner” của từng yếu tố rồi gọi là tối ưu |
| Test release | Một đợt cuối cho các baseline/config đã đăng ký; kết quả không tác động chọn checkpoint, seed hoặc config |

Không đặt ReduceLROnPlateau riêng cho một variant; không tự đổi optimizer/loss/gradient clipping giữa các ablation. Mọi run khởi tạo model và optimizer mới, không fine-tune nối tiếp.

Ghi `environment.json`: OS, Python, TensorFlow/Keras, scikit-learn, NumPy; CPU/model/core, GPU/model/VRAM/CUDA/cuDNN nếu có, tổng RAM; runtime precision/deterministic settings. Ghi wall time training, thời gian tính metric mỗi epoch và tổng pipeline riêng; GPU time cần đồng bộ phù hợp. Thời gian chỉ so trực tiếp khi cùng loại hardware; không ghi thời gian dự kiến như kết quả đã đo.

Mỗi run lưu toàn bộ hyperparameter, preprocessing policy, tokenizer hash, split hash, dataset hash, seed, version/revision mã, checkpoint, history và trạng thái hoàn tất. Sau mất phiên Colab, resume từ checkpoint không được gọi là run mới độc lập.

## 6. Experiment Matrix

### 6.1. Ma trận configuration

Giá trị dưới đây là **đề xuất sẽ thử**, không phải cấu hình tốt nhất đã được chứng minh. C0 sử dụng E=128, H=196, V=4.000, T=40 và các dropout kế thừa nguồn. V tính cả reserved IDs. B4 chỉ thay SpatialDropout1D, giữ input/recurrent dropout 0,3 và các Dropout 0,2/0,4.

| ID | Population/mốc | Biến thay đổi | Giá trị | Giữ cố định | Seeds | Test trong giai đoạn chọn |
|---|---|---|---|---|---|---|
| A1 | Binary đã lọc/source pipeline | LSTM nguồn → RNN | H=196 | Cleaning/sampling/tokenizer/padding/source split/batch/epoch/head | 42 | Khóa, chỉ mở sau khi đóng lựa chọn |
| A2 | Binary baseline sạch | Gói sửa protocol, không phải ablation một yếu tố | Train-only tokenizer, explicit/group-aware split, checkpoint theo val | Population/cleaner/kiến trúc RNN binary gần A1 | 42, 2026, 3407 | Không |
| C0 | Ba lớp/protocol sạch | Reference B | T=40, V=4.000, H=196, SpatialDropout=0,5, weight=None | P0, E=128, Dense100, optimizer/LR/batch/budget/mask/split | 42, 2026, 3407 | Không |
| B1-20 | C0 | Sequence length | 20 token | Toàn bộ config C0 trừ T | Cùng ba seeds | Không |
| B1-60 | C0 | Sequence length | 60 token | Toàn bộ config C0 trừ T | Cùng ba seeds | Không |
| B2-2k | C0 | Vocabulary cap | 2.000 | Toàn bộ config C0 trừ V | Cùng ba seeds | Không |
| B2-8k | C0 | Vocabulary cap | 8.000 | Toàn bộ config C0 trừ V | Cùng ba seeds | Không |
| B3-64 | C0 | Hidden units | 64 | Toàn bộ config C0 trừ H | Cùng ba seeds | Không |
| B3-128 | C0 | Hidden units | 128 | Toàn bộ config C0 trừ H | Cùng ba seeds | Không |
| B4-0 | C0 | SpatialDropout1D | 0 | Toàn bộ config C0 trừ một dropout này | Cùng ba seeds | Không |
| B4-025 | C0 | SpatialDropout1D | 0,25 | Toàn bộ config C0 trừ một dropout này | Cùng ba seeds | Không |
| B5-balanced | C0 | Train class weighting | Balanced từ train | Toàn bộ config C0 trừ class weights; không resampling | Cùng ba seeds | Không |
| B* | C0 hoặc một variant B đã chạy | Chọn bằng validation | Config nguyên vẹn đã thắng | Frozen protocol và các seeds | Checkpoint đã có | Chỉ đợt cuối |
| D-majority | Từng benchmark tương ứng | Baseline tham chiếu không neural | Luôn dự đoán lớp phổ biến nhất trong train | Cùng val/test tương ứng | Không cần training seed | Chỉ đợt cuối cho test |

C0 cung cấp luôn mức 40/4.000/196/0,5/None nên không chạy trùng mức cơ sở cho từng nhóm. Full matrix B có **10 config khác nhau × 3 seeds = 30 run**; cộng A1 một run và A2 ba run là **34 run huấn luyện**. B* tái sử dụng checkpoint; không cộng thêm run khi không cần train lại.

### 6.2. Câu hỏi, ghi nhận và diễn giải từng nhóm

Metric chung: **validation Macro-F1 chính**, Accuracy và Weighted-F1 phụ; P/R/F1 từng lớp; số parameter, best epoch và wall time. Không chọn theo test.

| Nhóm | Câu hỏi nghiên cứu | Biến thay đổi → biến giữ cố định | Kết quả phải ghi thêm | Cách diễn giải khi có kết quả |
|---|---|---|---|---|
| B1 | Sequence ngắn/dài ảnh hưởng việc giữ ngữ cảnh và khả năng học của RNN như thế nào? | T=20/40/60 → V/H/E/cleaner/dropout/split/optimizer/batch giữ C0 | Tỷ lệ truncation trên train/val; token length trước OOV/padding; metric theo nhóm độ dài; thời gian | Nếu T ngắn giảm F1 cùng truncation tăng, có bằng chứng phù hợp với mất ngữ cảnh; nếu T dài không tốt hơn, chỉ kết luận trong mức thử. Không tự kết luận vanishing gradient khi chưa đo gradient |
| B2 | Tăng vocabulary có giúp giảm OOV và cải thiện chất lượng không? | V=2.000/4.000/8.000 → T/H/E và phần còn lại giữ C0 | OOV rate theo số token và số tweet; embedding parameters; train/val gap | Đối chiếu OOV/F1/cost; giảm OOV không tự động chứng minh tăng F1. V thay đổi số parameter là hệ quả của yếu tố này, cần công bố |
| B3 | Sức chứa recurrent state có đáng với số parameter/chi phí tăng? | H=64/128/196 → T/V/E/dropout/head/protocol giữ C0 | Tổng parameters; recurrent parameters; gap train/val; mean±SD | Model lớn có train F1 cao nhưng val không tăng là dấu hiệu cần xem overfitting; model nhỏ tương đương val có thể được chọn theo quy tắc tie |
| B4 | SpatialDropout trên embedding ảnh hưởng generalization ra sao? | SpatialDropout=0/0,25/0,5 → giữ mọi dropout khác | Train-eval/val F1 gap, val loss, best epoch, số epoch chạy | Chỉ diễn giải tác động SpatialDropout; không gán cho toàn bộ regularization hay recurrent dropout |
| B5 | Class weighting có cải thiện cân bằng giữa các lớp không? | None/balanced → giữ đầy đủ dữ liệu và các config khác | Weight từng class; macro/per-class recall/F1; weighted-F1; unweighted val loss | Có thể đổi tradeoff giữa Accuracy và Macro-F1; chỉ kết luận minority class được lợi nếu per-class metrics hỗ trợ; không cân bằng test |

Class weight đề xuất: với C lớp, `w_c = N_train / (C × n_train,c)`. Tính từ số mẫu train sau xử lý dữ liệu, không lấy count toàn CSV. Loss train B5 có weighting nhưng metric và validation loss so sánh dùng dữ liệu không weighting; cần ghi rõ để tránh so weighted train loss với baseline như cùng một đại lượng.

## 7. Metric và đánh giá cuối

| Chỉ số/đầu ra | Định nghĩa và vai trò |
|---|---|
| Accuracy | Số dự đoán đúng/tổng mẫu; luôn ghi support và scope binary/three-class |
| Precision từng class | TP/(TP+FP); lớp được dự đoán có đáng tin cậy không |
| Recall từng class | TP/(TP+FN); mô hình bỏ sót bao nhiêu mẫu của lớp |
| F1-score từng class | Trung bình điều hòa Precision/Recall |
| Macro-Precision/Macro-Recall | Trung bình không weighting giữa các lớp; báo để tránh một số Precision/Recall không rõ cách average |
| Macro-F1 | Trung bình F1 từng lớp, mỗi lớp cùng trọng số; **tiêu chí chọn chính** |
| Weighted-F1 | Trung bình F1 với trọng số support; bổ sung góc nhìn chất lượng trên phân bố thực tế |
| Confusion matrix | Bản số đếm và bản chuẩn hóa theo nhãn thật; trục hàng=true, cột=predicted; đúng class order |
| Classification report | P/R/F1/support từng lớp, macro avg và weighted avg; lưu text lẫn structured metrics |
| Cross-entropy loss | Theo dõi hội tụ và mức phạt prediction sai/tự tin; bổ sung, không thay Macro-F1 |
| Mean±SD | Báo qua ba training seeds trên cùng split; không diễn giải là biến thiên do thay dataset split |

CSV gốc có 62,69% negative: predictor luôn negative có accuracy 62,69% trên toàn CSV mà không nhận diện neutral/positive. Đây là **ví dụ tính từ phân bố**, không phải test metric đã chạy. Trên split cuối, majority baseline lấy lớp phổ biến trong train và tính metric thật trên val/test. Weighted-F1 cũng có thể chịu ảnh hưởng lớp lớn; Macro-F1 và recall từng lớp cần được xem đồng thời.

Không dùng Precision/Recall dạng binary mặc định cho C0 ba lớp. Không dùng raw softmax probability để tính confusion matrix trước argmax. Trong classification report ghi đủ các lớp ngay cả khi một lớp không được model dự đoán. [Tài liệu metric scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html) là tham chiếu công thức/averaging khi triển khai.

**Test table policy:** mặc định chỉ mở test cho A1, A2, C0 và B* đã khóa, trong các benchmark tương ứng. A1 được ghi riêng là pipeline nguồn có contamination, không phải ước lượng sạch của hiệu quả tổng quát. Các hàng ablation khác để “không đánh giá theo protocol”. Nếu môn học cần test từng config, đăng ký danh sách trước khi mở test và đánh giá tất cả trong cùng đợt cuối; cấm dùng bảng đó để đổi winner.

## 8. Danh mục hình và bằng chứng cho báo cáo

Mỗi hình lưu PNG đủ chất lượng đưa vào báo cáo (khuyến nghị 300 dpi cho biểu đồ) và PDF/SVG khi công cụ hỗ trợ; đồng thời lưu bảng dữ liệu gốc. Caption phải có population, split, config, seed và đơn vị. Tên file không chứa “best” trước khi chọn xong.

| Mã | Hình/bảng trực quan cần sinh | Dùng để chứng minh/giải thích điều gì? | Dữ liệu và lưu ý |
|---|---|---|---|
| F01 | Phân bố sentiment CSV gốc | Quy mô, ba lớp và class imbalance | Counts và percentages; ghi trước xử lý |
| F02 | Phân bố sentiment qua các bước lọc nguồn | Nguồn đã chuyển population và mất neutral như thế nào | Raw → binary → drop5k; không trộn với C0 |
| F03 | Số tweet theo airline | Dataset không đồng đều theo hãng | Hãng chỉ là metadata EDA |
| F04 | Sentiment theo airline, trước/sau drop5k | Kiểm tra sampling có thay đổi cấu trúc dataset không | Counts và tỷ lệ; chưa kết luận bias nếu chưa thấy |
| F05 | Phân bố độ dài tweet trên train | Cơ sở giải thích sequence representation | Characters, whitespace words và tokenizer tokens phải tách rõ; đánh dấu T=20/40/60 |
| F06 | Tỷ lệ truncation/OOV theo config | Liên hệ representation với thông tin bị mất | Train/val; dùng raw token length trước khi PAD |
| F07 | Bảng mẫu trước/sau preprocessing | Quy tắc thực sự làm gì; phát hiện xóa negation/emoji/mention | Mẫu thật, có row ID; che username khi đưa lên slide nếu không cần |
| F08 | Pipeline tổng thể | Vị trí split, train-only fit, checkpoint và test evaluation | Có nhánh inference dùng artifact đã lưu |
| F09 | Sơ đồ RNN triển khai và minh họa unroll | Vai trò Embedding/recurrent state/dense head | Không vẽ LSTM gates cho RNN; hiển thị shape |
| F10 | Model summary/bảng parameter | Chứng minh kiến trúc và độ lớn thực tế | Xuất text + bảng, không chỉ screenshot chữ nhỏ |
| F11 | Train/Validation Loss | Hội tụ và khoảng cách generalization | B5 cần giải thích train loss weighted; đánh dấu best epoch |
| F12 | Train/Validation Accuracy | Diễn biến học và giới hạn accuracy | Ghi khác train fit-mode và train inference-mode nếu có |
| F13 | Train-eval/Validation Macro-F1 | Hội tụ trên tiêu chí chọn; tác động mất cân bằng | Full-split prediction, dropout off; có thể tính mỗi epoch cho dataset này |
| F14 | Confusion matrix count | Số nhầm cụ thể và support từng class | Tên sentiment, không chỉ nhãn 0/1/2 |
| F15 | Confusion matrix row-normalized | Tỷ lệ lớp thật bị nhầm sang lớp nào | Ghi phần trăm theo hàng; không nhầm với precision |
| F16 | Bar chart F1 từng class | Lớp mạnh/yếu và sự không đồng đều | Có support kèm theo; không dự đoán trước negative dễ nhất |
| F17 | Bảng/vùng minh họa prediction đúng | Model xử lý thành công những trường hợp nào | Chọn theo quy tắc, có true/pred và xác suất |
| F18 | Bảng/vùng minh họa prediction sai | Những cặp nhầm/khó khăn thực tế | Có raw/clean tweet; không chỉ chọn ví dụ hấp dẫn |
| F19 | Prediction confidence thấp | Những trường hợp decision không rõ ràng | Xếp theo max probability và top-2 margin; softmax chưa được calibration |
| F20 | Bảng config + validation Macro-F1 mean±SD | So sánh một yếu tố và độ ổn định giữa seeds | B dùng validation; trục/config và error bars rõ |
| F21 | Macro-F1 so với train time/parameters | Tradeoff chất lượng–chi phí | Chỉ so timing cùng hardware |
| F22 | Metric/error rate theo length bins | Tweet ngắn/dài có khác biệt quan sát được không | Bins khóa theo train; báo support/per-class mix |
| F23 | Giao diện demo và raw→clean→tokens→prediction | Demo tái sử dụng đúng pipeline và checkpoint | Kết quả của một input không phải metric tổng quát |

F01, F05, F07–F18 và F20 là bộ hình trọng tâm; F03/F04/F06/F19/F21/F22 bổ sung khi có câu chuyện được dữ liệu hỗ trợ. Hình gốc LSTM nếu trích trong báo cáo phải ghi là nguồn tham khảo, không hòa vào hình thực nghiệm nhóm.

## 9. Format các bảng kết quả

Ký hiệu **“Chưa chạy”** là ô chờ kết quả. **“Không đánh giá theo protocol”** là test không được mở cho config đó. Không điền số minh họa trông như metric thực nghiệm.

### Bảng Dataset

| Thuộc tính | Giá trị |
|---|---|
| Dataset/source/version/checksum | Twitter US Airline Sentiment; ghi manifest |
| Tổng dòng raw / số cột | 14.640 / 15, đã kiểm tra file hiện tải |
| Cột input / target | text / airline_sentiment |
| Counts raw negative / neutral / positive | 9.178 / 3.099 / 2.363 |
| Population benchmark | A: binary đã lọc; B: ba lớp hợp lệ |
| Missing/invalid/duplicates/conflicts | Chờ audit lúc triển khai và log xử lý |
| Train / val / test actual | Chờ manifest cuối, không lấy số dự kiến thay actual |
| Counts từng class theo split | Chờ manifest cuối |
| Token length median/P90/P95/P99 trên train | Chưa tính trong bước plan |
| OOV / truncation rate | Chưa tính theo configuration |

### Bảng cấu hình huấn luyện — C0 đề xuất

| Hyperparameter | Giá trị |
|---|---|
| Model | Một lớp RNN, tanh |
| Vocabulary cap / embedding dimension | 4.000 gồm reserved IDs / 128 |
| Sequence length | 40, post-padding/post-truncation |
| PAD / OOV / mask | 0 / 1 / mask PAD |
| Hidden units | 196 |
| SpatialDropout / input dropout / recurrent dropout | 0,5 / 0,3 / 0,3 |
| Dropout sau RNN / sau Dense | 0,2 / 0,4 |
| Dense hidden / output | 100, ReLU / 3, softmax |
| Loss | Sparse categorical cross-entropy |
| Optimizer / learning rate | Adam / 0,001 tường minh |
| Global gradient clipping | Norm 1,0 |
| Batch size / max epoch | 32 / 20 |
| Checkpoint / Early Stopping | Val Macro-F1 / patience 5, min_delta 0,001 |
| Class weighting | None, trừ B5 |
| Split seed / training seeds | 42 / 42, 2026, 3407 |
| Hardware / versions / training time | Ghi thực tế từng run; chưa có số đo |

### Bảng kiến trúc RNN — C0 đề xuất

Quy ước B=batch size, T=40, V=4.000, E=128, H=196, C=3. Parameters dưới đây là **tính lý thuyết từ cấu hình**, không phải model summary đã chạy; cần đối chiếu model summary khi triển khai. RNN gồm input kernel, recurrent kernel và một bias; khác công thức LSTM có nhiều gates.

| Layer | Input | Output | Parameters |
|---|---|---|---:|
| Token IDs | B×T | B×T | 0 |
| Embedding, mask PAD | B×T | B×T×128 | V×E = 512.000 |
| SpatialDropout1D, 0,5 | B×T×128 | B×T×128 | 0 |
| RNN 196, tanh | B×T×128 | B×196 | H(E+H+1) = 63.700 |
| Dropout, 0,2 | B×196 | B×196 | 0 |
| Dense 100, ReLU | B×196 | B×100 | (196+1)×100 = 19.700 |
| Dropout, 0,4 | B×100 | B×100 | 0 |
| Dense 3, softmax | B×100 | B×3 | (100+1)×3 = 303 |
| Tổng dự kiến | | | 595.703 |

A1/A2 có Dense2 nên tổng dự kiến của kiến trúc RNN tương ứng là 595.602 parameters nếu cùng embedding dimensions; đây không phải parameter count của LSTM nguồn. Mask không làm mất một row khỏi embedding matrix.

### Bảng các thực nghiệm

| Experiment | Thay đổi | Val Accuracy | Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---|---|---|---|---|
| Nguồn Kaggle — LSTM/binary, tham khảo | Source protocol | 90,61% ở epoch 20 | Không báo cáo | 90,32% | Không báo cáo |
| A1 — RNN/source pipeline | Thích nghi kiến trúc | Chưa chạy | Chưa chạy | Chưa chạy; protocol nguồn có giới hạn | Chưa chạy |
| A2 — RNN/binary sạch | Gói sửa protocol | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| C0 — RNN/ba lớp | Reference B | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |
| B1-20 / B1-60 | T=20 / 60; tách thành hai hàng khi xuất | Chưa chạy | Chưa chạy | Không đánh giá theo protocol | Không đánh giá theo protocol |
| B2-2k / B2-8k | V=2.000 / 8.000; tách hai hàng | Chưa chạy | Chưa chạy | Không đánh giá theo protocol | Không đánh giá theo protocol |
| B3-64 / B3-128 | H=64 / 128; tách hai hàng | Chưa chạy | Chưa chạy | Không đánh giá theo protocol | Không đánh giá theo protocol |
| B4-0 / B4-025 | SpatialDropout=0 / 0,25; tách hai hàng | Chưa chạy | Chưa chạy | Không đánh giá theo protocol | Không đánh giá theo protocol |
| B5-balanced | Balanced train weights | Chưa chạy | Chưa chạy | Không đánh giá theo protocol | Không đánh giá theo protocol |
| B* cuối | Config thắng validation, ghi ID thật | Chưa chạy | Chưa chạy | Chưa chạy | Chưa chạy |

Bảng A và B nên tách trong bản báo cáo cuối để tránh so scope khác nhau. Bảng tổng hợp mean±SD cần đi kèm bảng từng run/seed. Thêm cột Weighted-F1, parameters, best epoch, epochs trained, train time và delta validation Macro-F1 so C0 trong phụ lục. Chênh lệch dùng **điểm phần trăm**, ghi rõ đơn vị.

### Bảng kết quả từng lớp — mô hình cuối ba lớp

| Sentiment | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Negative | Chưa chạy | Chưa chạy | Chưa chạy | Chờ test manifest |
| Neutral | Chưa chạy | Chưa chạy | Chưa chạy | Chờ test manifest |
| Positive | Chưa chạy | Chưa chạy | Chưa chạy | Chờ test manifest |
| Macro average | Chưa chạy | Chưa chạy | Chưa chạy | Tổng test |
| Weighted average | Chưa chạy | Chưa chạy | Chưa chạy | Tổng test |

Bản binary không thêm dòng neutral giả; ghi neutral không thuộc phạm vi.

### Bảng đối chiếu notebook gốc

| Yếu tố | Nguồn | A1 | A2 | C0/B* |
|---|---|---|---|---|
| Loại recurrent layer | LSTM | RNN | RNN | RNN |
| Lớp/population | Binary/drop5k | Theo nguồn | Binary/protocol sạch | Ba lớp/không drop5k |
| Tokenizer fit | Toàn dữ liệu lọc | Theo nguồn | Train-only | Train-only |
| Sequence length | Auto theo corpus; demo31 | Kiểm tra khi chạy | Auto train | Config cố định |
| Checkpoint đánh giá | Epoch20 | Epoch20 | Best val Macro-F1 | Best val Macro-F1 |
| Kết quả | Test90,32%; F1 không báo | Chưa chạy | Chưa chạy | Chưa chạy |
| Mức tương đương | Nguồn tham khảo | Adapted, không exact | Sửa protocol, không exact | Bài toán rộng hơn, không exact |

## 10. Kế hoạch phân tích lỗi

Mọi prediction export giữ: run ID, seed, row/tweet ID, true label, predicted label, đủ xác suất các lớp, max probability, top-2 margin, raw/clean text, token lengths, OOV/truncation indicators và metadata airline cho slice. Không có nhãn thật thì không gán đúng/sai.

| Câu hỏi | Phương pháp và bằng chứng cần thu | Điều kiện được phép kết luận |
|---|---|---|
| Positive hay bị nhầm sang lớp nào? | Đọc hàng positive của CM count/row-normalized; xem mẫu từng cặp nhầm | Nêu cả số lượng và tỷ lệ trên support positive; binary không có lựa chọn neutral |
| Neutral hay bị nhầm sang lớp nào? | C0/B* ba lớp; hàng neutral và P/R/F1 neutral | Chỉ kết luận với benchmark ba lớp; không suy từ A |
| Negative có dễ nhất không? | So recall/F1 từng lớp, không so số đúng tuyệt đối | Negative có nhiều mẫu không chứng minh dễ hơn; chỉ phát biểu theo metric thực |
| Tweet nào khó? | Gắn nhóm lỗi: nhiều sentiment, ngữ cảnh thiếu, slang/typo, OOV, truncation, negation, sarcasm | Nhóm có định nghĩa, counts/support, ví dụ thật; chưa đủ mẫu thì mô tả case study |
| Ngắn/dài ảnh hưởng thế nào? | Bins đặt theo phân bố train; đối chiếu accuracy/error rate/Macro-F1 và class composition | Khác biệt quan sát không mặc định là quan hệ nhân quả; báo support tránh bins rất ít mẫu |
| Negation có gây lỗi? | Tìm marker not/no/never và contractions trước/sau clean; manual review; đối chiếu error rates | Có nhóm không negation làm tham chiếu; không kết luận nguyên nhân chỉ từ keyword |
| Sarcasm có gây lỗi? | Hai thành viên đánh dấu subset thật, đọc ngữ cảnh, ghi tiêu chí và bất đồng | Keyword như “great” không đủ xác nhận sarcasm; dùng cách nói “phù hợp giả thuyết” khi chỉ có vài trường hợp |
| Cleaning có mất thông tin? | So raw→clean, chú ý apostrophe/emoji/hashtag/URL/strip; ghi removed content | Có thể kết luận transformation đã bỏ thông tin; chỉ khẳng định tác động hiệu quả khi có preprocessing ablation tương ứng |
| Confidence thấp biểu thị gì? | Xếp các mẫu theo max probability và margin; kiểm tra đúng/sai/nhóm khó | Gọi là điểm softmax của model, không phải xác suất đúng đã calibration |

Chọn tập review có quy tắc: tối đa 10 mẫu mỗi cặp nhầm có đủ dữ liệu, 5 mẫu đúng mỗi lớp và 10 mẫu có margin thấp nhất; sampling seed cố định. Ghi tổng số review và số mẫu thực tế nếu không đủ. Manual review ưu tiên hai thành viên độc lập; không đưa kết luận ngôn ngữ học rộng từ tập nhỏ.

Error analysis test là phân tích sau đánh giá. Cải tiến gợi ra từ đó được ghi là future work; nếu tiếp tục tuning cần một holdout mới hoặc chỉ tuning trên validation, không dùng lại test như chưa từng xem.

## 11. Môi trường chính và tổ chức dự án

**Môi trường chính đề xuất: Google Colab, TensorFlow/Keras, lưu artifact bền vững vào thư mục Google Drive của nhóm.** Phù hợp làm việc notebook theo nhóm, chia sẻ kết quả và chạy CPU/GPU mà không bắt mọi thành viên cài môi trường ML. Drive dùng lưu trữ; không thay manifest/version control. Trong lần chạy, copy dữ liệu sang disk runtime nếu cần giảm I/O, đồng bộ checkpoint/history định kỳ.

GPU và RAM Colab được ghi theo runtime thực cấp; không hứa có T4 hoặc một thời lượng phiên cố định. [Colab FAQ](https://research.google.com/colaboratory/faq.html) nêu tài nguyên không được bảo đảm và giới hạn có thể thay đổi. Vì dataset nhỏ, phải smoke test/timing CPU và GPU trước khi chọn accelerator; RNN không mặc định có cùng tối ưu kernel như LSTM. Tất cả run so timing dùng cùng hardware profile; đổi profile thì ghi nhóm timing riêng.

Fallback: Kaggle Notebook nếu cần data attachment/runtime thay Colab; máy cá nhân cho inference/demo hoặc tái chạy với dependencies lock. Không thay seed/split/config khi đổi nền tảng. Không cài lại môi trường Python 3.6 lỗi thời chỉ để giấu các thay đổi tương thích; nếu không dựng được môi trường lịch sử, báo rõ là reproduction ở môi trường mới.

```text
airline-rnn/
├── data/
│   ├── raw/                    # CSV bất biến + manifest/checksum
│   ├── interim/                # cleaning audit, duplicate groups
│   ├── processed/              # representation theo config
│   └── splits/                 # manifests A1, A2, C0
├── notebooks/
│   ├── 00_source_audit.ipynb
│   ├── 01_data_and_eda.ipynb
│   ├── 02_baselines.ipynb
│   ├── 03_controlled_experiments.ipynb
│   └── 04_final_evaluation_and_inference.ipynb
├── configs/                    # base + variant + frozen matrix
├── models/
│   ├── runs/<config>/<seed>/
│   └── final/                  # model + tokenizer + label_map + config
├── results/
│   ├── histories/
│   ├── figures/
│   ├── metrics/
│   ├── predictions/
│   └── tables/
├── reports/
│   ├── source_audit.md
│   ├── error_analysis.md
│   ├── experiment_chapter.md
│   └── presentation_outline.md
├── demo/                       # tùy chọn, inference-only
├── environment/                # dependency lock + hardware/version logs
└── README.md
```

Đây là cấu trúc **dự kiến** sau khi duyệt. Bộ tối thiểu có thể gộp các bước vào một notebook “Run All”; các bản notebook tách phải được chạy đúng thứ tự qua README. Có thể thêm module dùng chung khi triển khai để tránh cleaning/inference logic trùng nhau, nhưng chưa tạo code ở bước này.

## 12. Sản phẩm cuối và tiêu chí nghiệm thu

| STT | Sản phẩm | Tiêu chí |
|---:|---|---|
| 1 | Notebook hoàn chỉnh | Restart/Run All từ raw data đến evaluation theo config, không dựa vào biến từ phiên cũ |
| 2 | Dataset hoặc hướng dẫn lấy | Đúng slug/file, checksum, version/ngày tải và schema |
| 3 | Baseline RNN thích nghi | A1/A2 có nguồn gốc và khác biệt được công bố; không gọi thành LSTM tái lập chính xác |
| 4 | Experimental configurations | Matrix đóng băng, một yếu tố mỗi variant, đủ seed logs |
| 5 | Best checkpoint | Chọn bằng val, cùng tokenizer/label/preprocessing; load được |
| 6 | Training history | CSV/JSON đủ epoch/metric/time/stop reason |
| 7 | Bảng kết quả | Tự xuất từ metrics, có support/scope/mean±SD |
| 8 | Confusion matrix | Count và row-normalized, labels/trục đúng |
| 9 | Classification report | P/R/F1/support và averaging rõ |
| 10 | Hình báo cáo | Figure files và bảng số liệu gốc, caption và run provenance |
| 11 | Inference examples | Dùng artifact đã lưu; có cả mẫu rỗng/OOV/quá dài và mẫu thường |
| 12 | Error analysis | Case thật, counts/support, giới hạn kết luận |
| 13 | README | Cài/load data/chạy theo thứ tự/đọc outputs/khôi phục run và tái inference |
| 14 | Nội dung chương Thực nghiệm | Môi trường, protocol, baseline, ablation, final result, error analysis, source comparison, limitations |
| 15 | Demo nhỏ nếu thời gian cho phép | Input tiếng Anh; sentiment + xác suất từng lớp; inference khớp notebook |

README phải ghi cách chạy lại cùng split và seeds; đường dẫn artifacts; config ID; dependency lock; expected dataset counts. Tái chạy không đòi hỏi cùng Accuracy/F1 đến từng chữ số trên phần cứng khác; sai lệch được đối chiếu trong giới hạn deterministic settings đã ghi.

## 13. Cấu trúc báo cáo và trình bày trên lớp

Cấu trúc theo báo cáo mẫu, thay nội dung cho NLP/RNN; có mục lục, danh mục hình/bảng, từ viết tắt và tài liệu tham khảo. Không áp dụng các lựa chọn GCN hay kết quả của mẫu vào đề tài này.

| Phần | Nội dung dự kiến | Bằng chứng/đầu ra gắn vào |
|---|---|---|
| MỞ ĐẦU | Bối cảnh airline sentiment; mục tiêu tái dựng và kiểm soát; phạm vi tiếng Anh; ba lớp ở benchmark chính; giới hạn nguồn dùng LSTM/binary | Source audit và định nghĩa A/B |
| CHƯƠNG 1. TỔNG QUAN VÀ CƠ SỞ LÝ THUYẾT | Sentiment Analysis; sequential data; preprocessing/tokenization; embedding học từ dữ liệu; RNN, hidden state và BPTT; vanishing/exploding gradient; P/R/F1/Macro-F1/Weighted-F1/CM | Công thức RNN, metric; minh họa unroll; nguồn học thuật và tài liệu API được kiểm tra |
| CHƯƠNG 2. PHƯƠNG PHÁP VÀ GIẢI PHÁP | Pipeline; dataset và population; P-source/P0; train-only vocabulary; padding/OOV/mask; kiến trúc RNN; loss/optimizer; train/checkpoint/inference; lý do thiết kế ablation | F07–F10, bảng dataset/config/architecture |
| CHƯƠNG 3. THỰC NGHIỆM VÀ ĐÁNH GIÁ | Môi trường và versions; quality/split manifest; baseline A; reference C0; controlled experiments; selection bằng val; final test; per-class; error analysis; source comparison; limitations | F01–F06, F11–F22; toàn bộ result tables thật |
| CHƯƠNG 4. DEMO / TRIỂN KHAI | Mục tiêu/chức năng; artifact loading; input cleaning/tokens/probabilities; giao diện; smoke tests; hạn chế phạm vi; triển khai local nếu đủ | F23; checkpoint/config ID; inference consistency |
| KẾT LUẬN | Trả lời A/B trong giới hạn bằng chứng; đóng góp pipeline; kết quả thực; hạn chế; hướng phát triển | Kết quả thực nghiệm đã khóa; không mục tiêu “bắt buộc90%” |
| TÀI LIỆU THAM KHẢO / PHỤ LỤC | Kaggle, dataset, nguồn lý thuyết/API; configs; full classification reports; seed tables; README | Link/version/ngày truy cập; bảng chi tiết và raw metrics |

Chương 3 có thể tổ chức như mẫu: **3.1** môi trường/dataset; **3.2** cấu hình/protocol; **3.3** baseline/source audit; **3.4** sequence/vocabulary; **3.5** capacity/dropout/weighting; **3.6** final result/hội tụ/per-class/lỗi; **3.7** đối chiếu nguồn/hạn chế. Đối với bộ tối thiểu không chạy ba lớp, bỏ các kết luận neutral và ghi scope binary ngay tên bảng/hình.

Mỗi nhóm thí nghiệm viết theo trình tự: câu hỏi → thiết kế và biến cố định → bảng thật → hình → nhận xét có số liệu → giới hạn. Phân biệt “quan sát” với “giả thuyết giải thích”. Ví dụ, T=60 kém hơn T=40 không đủ chứng minh vanishing gradient; cần kiểm tra gradient hoặc diễn đạt thận trọng.

Outline thuyết trình khoảng 10–12 slide: bài toán; phát hiện nguồn; dataset/imbalance; pipeline train-only; kiến trúc RNN; protocol; baseline; ablations; final CM/per-class; error cases; demo; kết luận/hạn chế. Không dùng slide toàn screenshot notebook.

## 14. Ba mức triển khai

| Mức | Phạm vi đề xuất | Số run huấn luyện dự kiến | Nội dung có thể kết luận |
|---|---|---:|---|
| Tối thiểu | Audit; A1 một seed; A2 một seed; đầy đủ metrics binary/hội tụ/CM; inference; README; bảng đối chiếu và limitations | 2 | Hoàn thành một thực nghiệm RNN thích nghi có tái chạy; không gọi là exact reproduction, không phân tích neutral, không khẳng định robust qua nhiều seeds |
| Khuyến nghị | A1 một seed; A2 ba seeds; C0 ba seeds; **B1 sequence, B3 hidden, B5 class weighting**; final three-class evaluation; error analysis; demo nhỏ | 22 | Có câu chuyện giữ ngữ cảnh/sức chứa/cân bằng lớp; đủ nội dung báo cáo và thuyết trình; chọn mô hình bằng val |
| Nâng cao | Toàn bộ năm nhóm B và ba seeds; phân tích sai số/slice sâu hơn; thêm kiểm định hoặc ablation riêng khi đủ nguồn lực | 34 cho matrix cơ sở; phần thêm tính riêng | Khảo sát thêm vocabulary/regularization, mean±SD, tradeoff chi phí; không suy rộng ngoài population đã kiểm tra |

Bộ khuyến nghị có 6 config B khác nhau (C0 + hai sequence + hai hidden + một weighting) × 3 seeds = 18 run, cộng A1 một và A2 ba = 22. Thời lượng thực tế chỉ ước lượng sau timing một run C0; không cam kết tổng thời gian trước benchmark.

**Nếu nhóm chỉ muốn tái hiện notebook nguyên trạng để hoàn thành môn:** dưới ràng buộc RNN, phương án tối thiểu chỉ là tái dựng có thích nghi. Nếu cần đúng nguyên trạng phải chấp nhận chạy LSTM như thực nghiệm nguồn riêng; đó là thay đổi phạm vi cần được duyệt, không phải một hành động mặc định.

Các mở rộng phù hợp, mỗi phần có ngân sách và protocol riêng:

- Preprocessing ablation giữ emoji/negation/dấu câu, thay một policy cụ thể; đo tác động bằng val, không chỉ dựa trên ví dụ lỗi.
- Learning rate hoặc gradient clipping ablation, kèm gradient-norm logging để kiểm tra giả thuyết ổn định tối ưu của RNN.
- Một cấu hình kết hợp các lựa chọn tốt, chỉ sau các ablation; train và đánh giá val như một config mới, không cộng cơ học các mức cải thiện.
- Paired bootstrap trên predictions final để ước lượng uncertainty của delta Macro-F1; giữ duplicate groups khi lấy mẫu lại; phân biệt uncertainty test sampling với SD qua training seeds. Không dùng p-value để chọn tiếp sau test.
- Benchmark theo airline/time để khảo sát domain shift, **tách khỏi** random split chính và có protocol mới.
- Reliability diagram/calibration bằng validation nếu muốn diễn giải confidence; giữ test để kiểm tra cuối.
- TF-IDF + Logistic Regression có thể làm comparator phi hồi quy. LSTM/GRU/BERT/Transformer chỉ được đề xuất như nghiên cứu mở rộng tách biệt; không thay RNN làm model chính và không thêm vào ma trận mặc định.

## 15. Các quyết định cần được nhóm duyệt trước bước code

1. Chấp nhận A là **RNN adapted baseline**, cùng tuyên bố rằng exact reproduction của LSTM nguồn không thuộc phạm vi hiện tại.
2. Chấp nhận **benchmark ba lớp C0/B** để có phân tích neutral; giữ hai lớp A trong bảng riêng.
3. Chọn mức tối thiểu/khuyến nghị/nâng cao; khuyến nghị bắt đầu 22 run, mở B2/B4 khi còn nguồn lực.
4. Chốt Colab + lưu artifacts bền vững; configs, split, seed và test release policy như plan.
5. Chấp nhận mục tiêu chất lượng quy trình và phân tích; không đặt một Accuracy/F1 chưa chạy làm điều kiện hoàn thành.

Sau khi duyệt mới tạo notebook/code/cấu trúc dự án và triển khai từng phase. Tài liệu này không chứa kết quả mô hình do nhóm tự chạy.
