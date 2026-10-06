# Xây dựng mô hình phân tích cảm xúc khách hàng trong ngành hàng không sử dụng RNN

Dự án thực nghiệm môn Deep Learning, từ dataset audit đến huấn luyện, controlled experiments, đánh giá cuối, phân tích lỗi và demo nhập tweet. Mô hình chính và mọi run thực thi đều dùng **RNN**. Notebook đã được chạy có output; kết quả số được lưu ở `results/`, không điền trước.

**Điểm cần phân biệt:** [notebook Kaggle nguồn](https://www.kaggle.com/code/chibuzorokocha/airline-sentiment-analysis-90-accuracy-using-rnn) dùng **LSTM và hai lớp**, sau khi bỏ neutral cùng 5.000 negative. Test Accuracy 90,32% là output lịch sử của nguồn. A1 của dự án giữ source pipeline và thay bằng RNN theo phạm vi đã duyệt; không tuyên bố tái lập chính xác LSTM hoặc đạt 90%. Benchmark chính giữ ba lớp.

## GitHub và website miễn phí

- Mã nguồn, notebook, báo cáo và các artifacts thực nghiệm: [GitHub repository](https://github.com/nguyennamtrieutien/airline-rnn-sentiment-analysis).
- Demo trực tuyến: [Airline RNN](https://nguyennamtrieutien.github.io/airline-rnn-sentiment-analysis/).

Demo được triển khai bằng **GitHub Pages**, chạy RNN trực tiếp trong trình duyệt bằng Web Worker. Cùng checkpoint `B5-balanced`, seed3407; không huấn luyện lại và không thay kiến trúc. Trọng số float32 xuất từ model gốc (~2,3MB), cùng tokenizer/P0/masking và các lớp Embedding → RNN/tanh → Dense/ReLU → softmax. Dropout tắt khi inference. Tám khâu và vector thật được hiển thị như bản Python; thời gian xử lý phụ thuộc thiết bị. Các câu nhập được phân tích trong trình duyệt, không gửi đến dịch vụ inference bên ngoài.

Đối chiếu bản JavaScript với 2.149 predictions test đã lưu và 18 câu bổ sung: **2.167/2.167 lớp dự đoán khớp**, sai lệch xác suất lớn nhất <0,0000004. Đây là kiểm tra chuyển đổi inference, không phải một thực nghiệm huấn luyện mới. Số liệu báo cáo vẫn lấy từ TensorFlow. Bằng chứng: `results/metrics/browser_inference_verification.json`.

Mỗi push vào `main` tự chạy kiểm tra cú pháp, checksum, prediction parity rồi deploy qua `.github/workflows/pages.yml`. Repository công khai dùng GitHub Pages miễn phí; không cần máy chủ Python, API key hoặc thông tin thanh toán để mở demo. Lần mở đầu cần tải trọng số; browser phải hỗ trợ Web Worker, module JavaScript và Web Crypto.

Chạy bản website tĩnh tại máy:

```bash
python3 scripts/build_static_site.py
python3 -m http.server 8080 --directory .build/site
```

Mở `http://localhost:8080`. Kiểm tra parity bằng Node.js và thư viện chuẩn Python:

```bash
python3 scripts/build_browser_fixtures.py
node scripts/verify_browser_model.mjs .build/browser/verification-input.json
```

Muốn xuất lại từ checkpoint, dùng môi trường ML đã cài ở phần dưới:

```bash
python scripts/export_browser_model.py
python scripts/verify_browser_model.py
```

`deploy/browser-model/` là bundle triển khai đã kiểm tra checksum; `deploy/browser-reference-cases.json` lưu output TensorFlow của 18 câu bổ sung. Các model/run, dataset splits và kết quả gốc được kèm trong repository để xác minh và chạy lại. `.venv`, cache, file môi trường chứa bí mật và các bản build tạm được loại khỏi Git.

## Mở kết quả

- `reports/final/BaoCao_Airline_RNN.docx`: báo cáo Word theo mẫu, 37 trang, 19 hình và 19 bảng; mục lục và trang bìa đã hoàn thiện.
- `reports/final/ThuyetTrinh_Airline_RNN.pptx`: 16 slide có speaker notes, bảng và biểu đồ chỉnh sửa được.
- `reports/final/KichBan_ThuyetTrinh_Demo.md`: lịch thuyết trình khoảng 13 phút, thao tác demo và câu hỏi phản biện dự kiến.
- `notebooks/Airline_RNN_Complete.ipynb`: notebook đầy đủ, có output thực thi.
- `notebooks/Airline_RNN_Complete.html`: bản xem notebook không cần chạy kernel.
- `reports/experiment_chapter.md`: chương Thực nghiệm và đánh giá, có số thật.
- `reports/error_analysis.md`: confusion pairs, per-class metrics, slices và giới hạn diễn giải.
- `results/predictions/assistant_error_review.csv`: nhận xét của trợ lý cho 85 case; các giả thuyết được tách khỏi nhãn của hai người review trong `manual_error_review.csv`.
- `reports/source_audit.md`: đối chiếu với notebook Kaggle.
- `reports/report_structure.md`, `reports/presentation_outline.md`: cấu trúc báo cáo và thuyết trình.
- `reports/project_status.md`: trạng thái từng phase, kiểm tra và việc nhóm còn thực hiện.
- `results/tables/experiment_comparison.csv`: so sánh cấu hình; test của ablation không được chọn để trống có chủ đích.
- `models/final/`: model, tokenizer, mapping lớp, config, representation metadata và manifest checksum.

## Môi trường và cài đặt

Python **3.12** được kiểm tra; TensorFlow 2.20.0, Keras 3.11.3, NumPy 2.2.6, scikit-learn 1.7.2. Dependency trực tiếp pin ở `requirements.txt`; freeze toàn bộ lần chạy ở `environment/lock-local.txt`. Các run bàn giao được chạy bằng CPU máy cá nhân macOS arm64, RAM16GiB; chi tiết từng run nằm trong `models/runs/.../environment.json`. GPU TensorFlow không khả dụng trong lần này. Không gán kết quả CPU cho Colab GPU.

Từ terminal ở thư mục `airline-rnn`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
python -m pytest -q
```

Không cần cài TensorFlow Metal. TensorFlow CPU đủ cho dataset này. Trên Windows, kích hoạt `.venv\Scripts\activate` và dùng các lệnh Python tương tự. Hỗ trợ notebook trên Colab được chuẩn bị; chưa thực thi kiểm thử cloud trong lần bàn giao này.

## Dataset và provenance

Dataset: [Twitter US Airline Sentiment — CrowdFlower](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment). File `data/raw/Tweets.csv` được kèm theo. Nếu thiếu, `prepare` tải public dataset API và kiểm tra checksum trước khi dùng.

```text
SHA-256: ea94b23f41892b290dec3330bb8cf9cb6b8bc669eaae5f3a84c40f7b0de8f15e
Raw: 14.640 rows, 15 columns
negative: 9.178 | neutral: 3.099 | positive: 2.363
Input: text | target: airline_sentiment
Main label IDs: negative=0, neutral=1, positive=2
Binary label IDs: negative=0, positive=1
```

Nếu public endpoint thay đổi hoặc bị chặn, tải dataset từ trang trên, đặt đúng file `Tweets.csv` vào `data/raw/` rồi chạy lại. File khác checksum bị từ chối; không tự trộn phiên bản dataset. Nguồn code tham khảo nằm trong `sources/original.ipynb`; file này có LSTM lịch sử và **không được pipeline thực thi**.

Main clean benchmark có 14.353 mẫu: train10.043 / validation2.161 / test2.149. Exact duplicates bị loại; nhóm trùng tweet ID/raw text chuẩn hóa/clean text giữ cùng split; nhóm nhãn mâu thuẫn được cách ly. Manifest và exclusion logs lưu đầy đủ. Tỷ lệ≈70/15/15 do group split, seed42. A2 có population/split binary riêng; A1 giữ population/split legacy với các rủi ro của nguồn. Không so trực tiếp ba benchmark như cùng một test set.

## Ma trận đã đăng ký

| Configuration | Benchmark | Factor | Training seeds |
|---|---|---|---|
| A1 | Source binary adapted | Source preprocessing, RNN196; epoch20 cuối | 42 |
| A2 | Clean binary | Train-only tokenizer, group split, val checkpoint | 42, 2026, 3407 |
| C0 | Clean three-class | T40, H196, embedding128, no class weight | 42, 2026, 3407 |
| B1-20 | Clean three-class | Chỉ T=20 | 42, 2026, 3407 |
| B1-60 | Clean three-class | Chỉ T=60 | 42, 2026, 3407 |
| B3-64 | Clean three-class | Chỉ H=64 | 42, 2026, 3407 |
| B3-128 | Clean three-class | Chỉ H=128 | 42, 2026, 3407 |
| B5-balanced | Clean three-class | Chỉ balanced class weights từ train | 42, 2026, 3407 |

Tổng **22 run**, ba seeds trên cùng fixed split. Không có run LSTM/GRU/BERT/Transformer. Advanced có thêm vocabulary/spatial-dropout variants, **chưa chạy** trong kết quả bàn giao; chỉ thực hiện thành vòng đăng ký mới trong output root riêng, có holdout mới nếu dùng kết quả để đưa ra kết luận confirmatory.

P0 main: NFKC/HTML/lowercase, mention/URL markers, contractions giữ phủ định, bỏ punctuation; không bỏ stopwords. Vocab4000 gồm PAD0/OOV1/marker2–4, train-only; post-padding/truncation, mask PAD. RNN/tanh → Dense100/ReLU → softmax; dropout kế thừa nguồn. Adam0.001, batch32,max20epochs, global clipnorm1.0 main. Checkpoint theo full-val Macro-F1, ES patience5/min_delta0.001. Best checkpoint và ES dùng hai quy tắc riêng: lưu mọi cải thiện Macro-F1 (tie theo loss), ES đợi cải thiện đáng kể theo min_delta.

Trong dataset hiện có, T40/T60 cùng giữ trọn chuỗi; B1-60 không phải bằng chứng về lợi ích thêm context. T20 có truncation. Thời gian gồm training và metrics toàn train/validation mỗi epoch; không phải inference latency.

## Chạy pipeline

Các lệnh dưới có thể chạy tuần tự:

```bash
python -m airline_rnn prepare
python -m airline_rnn smoke
python -m airline_rnn train --tier recommended
python -m airline_rnn evaluate
python -m airline_rnn report
python -m airline_rnn status
python scripts/verify_artifacts.py
```

Hoặc một lệnh:

```bash
python -m airline_rnn all --tier recommended
```

`evaluate` chỉ mở test sau khi **toàn bộ** ma trận hoàn tất và selection được khóa. Cấu hình cuối chọn mean validation Macro-F1; các config cách max dưới0.005 ưu tiên ít parameters, rồi median time/thứ tự matrix. Checkpoint demo lấy seed có validation F1 trung vị trước test. Test chỉ đánh giá A1/A2/C0/winner; những B khác là validation-only. Không refit trên train+val sau test.

**Cache:** Run All/lệnh `all` xác minh config/code/data/artifacts rồi dùng lại run đã hoàn tất. Không tự huấn luyện đè checkpoint đã chọn. Thay code/config/data khiến fingerprint khác và bị từ chối. Checkpoint thiếu/sửa sau khi selection cũng bị chặn. Nếu bị ngắt trước selection, `train --restart-incomplete` archive run lỗi và train lại đúng seed; các run hoàn tất được giữ.

**Huấn luyện lại từ đầu**, lưu riêng artifacts (không cần xóa bản bàn giao):

```bash
python -m airline_rnn all --tier recommended --root fresh_reproduction
```

Đây là reproduction cùng benchmark đã công bố, không phải một vòng tuning mới được phép sử dụng holdout cũ. Muốn thay matrix/tune theo insight từ test cần thiết kế vòng mới với test chưa từng xem. Minimum chỉ A1/A2 seed42: `all --tier minimum --root minimum_reproduction`; các báo cáo kết quả đầy đủ bàn giao dựa trên recommended. Tier khác phải dùng root mới.

## Chạy notebook/Colab/Drive

Máy cá nhân: mở `.ipynb`, chọn kernel `.venv`; notebook tự xác định project root. Nếu cần Jupyter UI, cài thêm JupyterLab trong environment. `scripts/execute_notebook.py` dùng nbclient để chạy top-to-bottom và export HTML mà không cần JupyterLab.

Colab:

1. Mở `.ipynb` trong Colab.
2. Chạy cell đầu, upload `airline-rnn-project.zip` nếu chưa có thư mục project; cài pinned dependencies theo cell.
3. Chọn `RUN_ROOT=ROOT` để xem/xác minh run đã có, hoặc một thư mục mới để train từ đầu.
4. Nếu lưu Drive, mount Drive trước cell `RUN_ROOT`, đặt `RUN_ROOT` thành **thư mục mới** trên Drive. Code vẫn đọc từ project đã upload; artifacts ghi ra Drive. Không ghi đè bản test đã mở để thử hyperparameter khác.
5. Run All. Nếu đã import ML dependency sai version trước khi cài, restart runtime và Run All.

Google Drive chỉ lưu artifacts, không bắt buộc đăng nhập để chạy local. GPU Colab là tùy chọn, seeds/deterministic ops không đảm bảo bit-identical giữa CPU/GPU hoặc phiên bản thư viện khác. Ghi environment mới và đối chiếu độ biến thiên, không sửa số để giống nguồn.

## Inference và demo

```bash
python -m airline_rnn infer --text '@united Thank you for the helpful service!'
python -m airline_rnn demo --port 8765
```

Mở `http://127.0.0.1:8765`. Server chỉ bind máy local. Nhập tiếng Anh; output sentiment và softmax từng lớp, clean text/OOV/truncation. Empty input bị từ chối; inference dùng cùng tokenizer/cleaner/mapping/checkpoint, không fit mới. Website có logo trường từ mẫu Word, thông tin tiểu luận/giảng viên và ba thành viên. Mỗi lần chạy hiển thị tám khâu thực tế: tiếp nhận → cleaning → token/ID/OOV → padding/masking → forward pass Embedding/RNN/Dense/softmax → đọc xác suất và argmax. Có thể mở chi tiết vector thật, chuỗi ID và mask. Các thời gian trên web thuộc request hiện tại, không phải số benchmark. Score chưa calibration; chưa đánh giá tiếng Việt/miền khác. Ctrl+C để tắt server.

## Artifacts và báo cáo

```text
airline-rnn/
├── airline_rnn/             # companion modules: data/model/train/eval/report/inference/demo
├── scripts/                 # notebook authoring/execution, validation/export utilities
├── tests/                   # protocol, holdout gate, model serialization
├── configs/frozen_matrix.json
├── data/
│   ├── raw/                 # original CSV + download/checksum manifest
│   ├── interim/             # benchmark populations + exclusion/conflict logs
│   ├── processed/           # train/val arrays, tokenizer, representation stats (no test arrays during tuning)
│   └── splits/              # row/group/split manifests
├── notebooks/               # executed notebook + HTML
├── models/
│   ├── runs/<config>/seed-*/ # best/last models, config/seed/environment/history/val metrics
│   ├── smoke/               # isolated smoke checks; excluded from scientific metrics
│   └── final/               # selected inference bundle
├── results/
│   ├── figures/             # PNG300dpi and PDF + caption/scope manifest
│   ├── metrics/             # per-run JSON/reports, selection/test release
│   ├── predictions/         # held-out predictions/probabilities, inference examples/manual review
│   ├── histories/           # training histories per config/seed
│   └── tables/              # report-ready CSVs
├── reports/                 # audit/chapter/error analysis/outline/approved plan
│   ├── final/               # Word, PowerPoint, kịch bản và nguồn nội dung
│   └── build_sources/       # mã tạo Word/slide, page map và hướng dẫn authoring
├── environment/             # pinned freeze/runtime; local kernel paths excluded from ZIP
├── sources/                 # Kaggle code reference + historical output evidence
├── requirements.txt
└── README.md
```

Metrics: Accuracy, macro Precision/Recall, per-class Precision/Recall/F1/Support, Macro-F1, Weighted-F1, count/normalized CM, classification report. Predictions giữ row IDs/true/pred/softmax để tính lại. Majority baseline từ train giúp đọc Accuracy trong dataset imbalance.

23 hình dự kiến được xuất theo tier/case availability; manifest là danh sách thực tế. Hình ví dụ không thay metric tổng thể. Slices length/OOV/negation mô tả liên hệ, không chứng minh nguyên nhân. `manual_error_review.csv` có cột cho hai người annotate; chưa có nhãn sarcasm được xác minh thì không kết luận sarcasm gây lỗi. Preprocessing ablation và gradient diagnostics chưa chạy, không bịa kết quả.

Mọi notebook output, bảng và phần chương thực nghiệm ghi rõ nguồn, benchmark, split, config và seed/mean±SD. Ba training seeds không phải ba split và SD không phải confidence interval. Báo cáo Word và slide đã hoàn thiện theo mẫu. Nhóm bổ sung đóng góp thực tế, kiểm tra thông tin bìa và tập trình bày. Hai người có thể review độc lập 85 case nếu muốn xác minh các giả thuyết định tính; chưa có agreement được đo.

Thông tin trường/khoa, giảng viên và sinh viên trên bìa được lấy theo mẫu theo xác nhận của người cung cấp. File mẫu gốc không bị sửa. Word giữ công thức, bảng, mục lục liên kết và footer; sau khi biên tập làm đổi số trang, cần cập nhật số trang của mục lục hoặc tạo lại bằng nguồn authoring. PowerPoint được kiểm tra bằng package validators và import/render; chưa thử trực tiếp trên Microsoft PowerPoint. Mã tạo tài liệu dùng dependency riêng, không cần cài để chạy notebook/model/demo.

Thông tin nhóm dùng chung ở `airline_rnn/web/project-info.json`: 2611307 Trịnh Nguyễn Anh Hào, 2611308 Lê Huy Huân, 2611323 Nguyễn Nam Triều Tiên. Word đã cập nhật trang bìa và phụ lục. `/api/analyze` trả NDJSON gồm trạng thái/kết quả từng khâu; `/api/predict` vẫn trả JSON prediction như trước. Có 25 test đã pass; kiểm tra HTTP và trace parity ở `results/metrics/demo_analysis_verification.json`.

Cách gọi và tên file đã thống nhất là RNN. Khi tạo mạng, alias `RNN` gọi lớp RNN thuần có sẵn của Keras; tên API chuẩn của thư viện và định danh trong checkpoint/provenance lịch sử vẫn được giữ để tải đúng trọng số. Thay đổi cách gọi không huấn luyện lại hoặc sửa số liệu. `sources/naming_migration.json` cùng `sources/training_source_before_naming.zip` xác thực nguồn cũ và nguồn hiện tại; cache chỉ được chấp nhận khi cấu hình, dữ liệu, seed và checksum artifact khớp. Dùng thư mục output mới để huấn luyện lại từ đầu.
