"""Author the portable notebook. Does not train or invent numerical outputs."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip()))


md(r"""
# Xây dựng mô hình phân tích cảm xúc khách hàng trong ngành hàng không sử dụng RNN

**Bài thực nghiệm Deep Learning có thể chạy lại — ma trận khuyến nghị 22 run.**

Input: tweet tiếng Anh; output chính: `negative / neutral / positive`. Mọi recurrent layer thực thi trong dự án là **RNN**, không có LSTM/GRU/Transformer.

Notebook Kaggle [Airline Sentiment Analysis (90% Accuracy) using RNN](https://www.kaggle.com/code/chibuzorokocha/airline-sentiment-analysis-90-accuracy-using-rnn) thực tế dùng **LSTM và chỉ hai lớp**. Vì vậy A1 dưới đây là *source pipeline adapted to RNN*, không được gọi là tái lập chính xác mô hình nguồn. A2 kiểm tra binary bằng protocol sạch; C0 và B là benchmark ba lớp trọng tâm.

Notebook này đi từ audit → chuẩn bị → huấn luyện → selection validation → final test → phân tích lỗi → inference. File đã có output thực thi. `Run All` kiểm tra cache theo checksum; để huấn luyện mới hoàn toàn, đặt `RUN_ROOT` thành thư mục mới trước khi chạy. Thư viện đồng hành `airline_rnn/` là phần của notebook; upload **toàn bộ project ZIP** khi dùng Colab, không chỉ một file `.ipynb`.

Không dùng lại holdout đã mở để chọn hyperparameter. Softmax trong demo chưa calibration. Một số case review cần nhóm tự annotate; notebook không tự gán nhãn sarcasm.
""")
md("""
## Chuẩn bị runtime — máy cá nhân hoặc Colab

Colab: upload project ZIP qua cell sau nếu chưa có thư mục code. ZIP chứa dataset public, checkpoint và artifacts; không chứa `.venv`. Dependencies pin ở `requirements.txt`. Cell cài đặt chỉ chạy khi package thiếu hoặc sai phiên bản. Nếu đã import ML package sai phiên bản trước khi cài, cần restart runtime rồi Run All.

Máy cá nhân: tạo virtual environment theo README và chọn kernel của environment đó. Các số thời gian/GPU dưới đây được đọc từ run thực tế, không gán cấu hình Colab cho kết quả chạy CPU.
""")
code(r"""
from pathlib import Path
import os, sys, json, importlib.metadata, subprocess, zipfile

def locate_project():
    for path in [Path.cwd(), *Path.cwd().parents]:
        if (path / "airline_rnn" / "training.py").exists():
            return path
    found = sorted(Path.cwd().glob("**/airline_rnn/training.py"))
    return found[0].parents[1] if found else None

ROOT = locate_project()
if ROOT is None:
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError("Mở notebook từ thư mục project hoặc cài project theo README.")
    uploaded = files.upload()
    archives = [name for name in uploaded if name.endswith(".zip")]
    if len(archives) != 1:
        raise RuntimeError("Upload đúng một project ZIP.")
    destination = Path.cwd() / "airline_project"
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archives[0]) as archive:
        for item in archive.infolist():
            if not (destination / item.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("ZIP chứa đường dẫn không hợp lệ.")
        archive.extractall(destination)
    ROOT = locate_project()
if ROOT is None:
    raise RuntimeError("Không tìm thấy package airline_rnn trong project.")
sys.path.insert(0, str(ROOT))
print("Project:", ROOT)

requirements = (ROOT / "requirements.txt").read_text().splitlines()
mismatches = []
for line in requirements:
    if "==" not in line:
        continue
    package, wanted = line.split("==", 1)
    try:
        actual = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        actual = None
    if actual != wanted:
        mismatches.append((package, actual, wanted))
if mismatches:
    print("Install pinned dependencies:", mismatches)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    if any(name in sys.modules for name in ("numpy", "tensorflow", "keras", "sklearn")):
        raise RuntimeError("Đã cài dependencies; restart runtime để tránh dùng package cũ đã import.")
else:
    print("All pinned dependency versions match.")
""")
code("""
import pandas as pd
import numpy as np
from IPython.display import display, Image, Markdown
from airline_rnn.__main__ import bootstrap
from airline_rnn.common import read_json
from airline_rnn.config import configurations, configuration_by_id
from airline_rnn.data import prepare_data, load_benchmark, prepare_representation
from airline_rnn.training import run_matrix, run_experiment, environment_snapshot
from airline_rnn.model import RNN, configure_runtime
from airline_rnn.evaluation import freeze_selection, evaluate_final, verified_runs
from airline_rnn.reporting import generate_eda, generate_results, write_report

TIER = "recommended"  # minimum=2 runs; recommended=22; advanced=34 (separate NEW root)
RUN_ROOT = ROOT       # fresh training: ROOT / "fresh_reproduction"; keep existing holdout frozen
# Colab + Drive, optional: mount Drive and use a NEW path such as /content/drive/MyDrive/airline-rnn-run-01
RUN_ROOT = bootstrap(RUN_ROOT)
configure_runtime(42)

def figure(name):
    path = RUN_ROOT / "results/figures" / f"{name}.png"
    if path.exists():
        display(Image(filename=str(path), width=950))
    else:
        print("Figure unavailable for this tier:", name)

display(pd.DataFrame([{"Configuration": config.id, "Benchmark": config.benchmark,
                      "T": config.sequence_length or "auto/train (A1: all source)", "H": config.hidden_units,
                      "Class weights": config.class_weighting, "Seeds": list(seeds)}
                     for config, seeds in configurations(TIER)]))
print("Current notebook runtime:")
display(environment_snapshot())
""")
md("""
## Phase 0 — Audit nguồn và định nghĩa reproduction

Dataset public [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment): 14.640 tweet, 15 cột. Nguồn bỏ toàn bộ neutral và 5.000 negative đầu, còn 6.541 tweet. Chỉ text và sentiment dùng cho học.

Source cleaning dùng `lstrip` theo character set, lowercase và regex xóa punctuation; tokenizer Keras cap 4.000 fit **trước split trên cả 6.541 tweet**, không OOV token; auto maxlen=31, pre-padding/pre-truncation. Label binary negative=0, positive=1. Split nguồn: sklearn train/test 70/30 stratified random_state=1, sau đó validation_split=0.2 lấy cuối training array. Embedding128 → SpatialDropout0.5 → **LSTM196** (input/recurrent dropout0.3) → Dropout0.2 → Dense100/ReLU → Dropout0.4 → Dense2/softmax. Sparse CE, Adam (không ghi LR), batch32,20epochs, Accuracy.

Nguồn báo test Accuracy90.32094%, không phải số đo của nhóm. Train-only vocabulary và duplicate isolation chưa được thực hiện ở nguồn. A1 giữ các vấn đề này để audit pipeline, thay recurrent cell bằng RNN/tanh và ghi Adam0.001 tường minh. Đây là khác biệt có chủ đích do phạm vi môn học; không có một run LSTM ẩn trong project.
""")
code("""
display(read_json(RUN_ROOT / "sources/historical_outputs.json"))
print("Historical source output above is NOT our RNN result.")
""")
md("""
## Phase 1–2 — Dataset, checksum, EDA và split sạch

Raw CSV bất biến được kiểm tra SHA-256. Bỏ exact duplicate records; nối nhóm theo tweet ID/raw text chuẩn hóa/clean text; cách ly toàn bộ nhóm có nhãn mâu thuẫn. A2 và C0/B giữ nhóm trong cùng split và stratify trên nhóm. Main split≈70/15/15, seed42; A2≈56/14/30, seed1; training seeds không thay split. A1 giữ split nguồn riêng, có nguy cơ overlap.

EDA metadata airline phục vụ kiểm tra phân bố, không đưa vào input của mô hình. Cột confidence/gold/negative reason không dùng làm feature hay lọc theo chất lượng nhãn tùy tiện. Các thống kê bị loại/split được lưu để kiểm tra.
""")
code("""
profile = prepare_data(RUN_ROOT)
for config, _ in configurations(TIER):
    prepare_representation(RUN_ROOT, config)
generate_eda(RUN_ROOT)
display(profile)
display(pd.read_csv(RUN_ROOT / "results/tables/dataset_splits.csv"))
display(pd.read_csv(RUN_ROOT / "results/tables/exclusions.csv"))
for name in ("F01_raw_sentiment", "F02_source_filtering", "F03_airlines", "F04_airline_filtering", "F05_train_lengths"):
    figure(name)
""")
md("""
## Phase 3 — Preprocessing và biểu diễn chuỗi

P0: chuẩn hóa Unicode/HTML/apostrophe, lowercase, URL→`urlmarker`, mention→`usermarker`, mở contractions giữ phủ định (`can't`→`can not`), bỏ punctuation còn lại; không stopword removal/stemming. Text rỗng sau cleaning→`emptymarker`. Tokenizer fit train-only, cap4.000 gồm PAD0,OOV1,marker2–4. Không học vocabulary từ validation/test. C0/B post-padding, post-truncation, mask PAD; Embedding được học cùng model.

Ảnh preprocessing và CSV chứa ví dụ thật. Emoji/punctuation bị bỏ trong P0 là hạn chế đã biết; chưa khẳng định tác động lên Accuracy nếu chưa có ablation riêng. T40/T60 có thể cùng coverage; chỉ diễn giải thêm context khi thống kê truncation hỗ trợ.
""")
code("""
display(pd.read_csv(RUN_ROOT / "results/tables/preprocessing_examples.csv"))
for name in ("F07_preprocessing", "F08_pipeline"):
    figure(name)
display(read_json(RUN_ROOT / ("data/processed/C0/representation.json" if TIER != "minimum" else "data/processed/A2/representation.json")))
""")
md(r"""
## Phase 4–6 — Baseline và ma trận controlled experiments

Với embedding $x_t$, RNN cập nhật $h_t=\tanh(W_xx_t+W_hh_{t-1}+b)$; hidden state cuối → Dense/ReLU → softmax. Không có gate. Vanishing/exploding gradient là động cơ lý thuyết; không kết luận đã quan sát từ learning curve nếu chưa đo gradient. C0/B clip global norm1.0; A1/A2 giữ Adam không clipping.

- A1: source binary pipeline adapted, seed42, epoch20 cuối; không ES.
- A2: clean binary, train-only tokenizer/group split, 3 seeds.
- C0: main3class, T40/H196/embedding128, 3 seeds.
- B1-20/B1-60: chỉ đổi sequence length. B3-64/B3-128: chỉ đổi hidden units. B5-balanced: chỉ đổi class weights `N_train/(K*n_class_train)`.

Những factor còn lại giữ cố định. Mỗi run lưu config, code/data/split hashes, environment, class weights, history, best/last checkpoint, validation predictions. Best checkpoint theo **full-validation Macro-F1**, tie theo validation loss; ES patience5/min_delta0.001/max20epochs. Metrics full-split trong inference mode, không lấy trung bình F1 theo minibatch. Smoke test nhỏ dưới đây tách khỏi run khoa học.
""")
code("""
smoke = run_experiment(RUN_ROOT, configuration_by_id("C0"), 42, smoke=True, max_epochs=2)
print("SMOKE ONLY — excluded from selection/test/report comparisons:")
display({key: smoke[key] for key in ("status", "smoke", "epochs_trained", "parameters")})
""")
code("""
# This is the computationally expensive cell on a fresh root.
# Cached runs are accepted only if config/code/data and artifact hashes match.
if (RUN_ROOT / "results/metrics/final_selection.json").exists():
    verified_runs(RUN_ROOT)  # block incomplete/modified cache after selection
runs = run_matrix(RUN_ROOT, tier=TIER)
print("Completed scientific runs:", len(runs))
display(pd.read_csv(RUN_ROOT / "results/tables/validation_runs.csv"))
""")
md("""
## Phase 7 — Khóa mô hình cuối bằng validation, rồi mở test

Selection dùng mean validation Macro-F1. Những configuration cách max dưới0.005 được ưu tiên ít parameters, sau đó median training time và thứ tự matrix. Tolerance là quy tắc thực dụng đặt trước, không phải kiểm định tương đương. Demo dùng seed có validation Macro-F1 trung vị, chọn trước khi đọc test.

Mô hình cuối giữ checkpoint đã train trên train và chọn epoch trên validation; không refit train+validation sau khi xem test. Test chỉ đánh giá A1/A2/C0/winner; test của ablation không được chọn để trống. Không ghép nhiều factor thắng thành một model chưa đăng ký. Ba seeds dùng cùng split; mean±SD phản ánh training variability, không phải confidence interval.
""")
code("""
selection = freeze_selection(RUN_ROOT)
display(selection)
display(pd.read_csv(RUN_ROOT / "results/tables/validation_summary.csv"))
# All model selection above is frozen before this call first reads test features for prediction.
test_summary = evaluate_final(RUN_ROOT)
display(test_summary)
""")
md(r"""
## Phase 5/7 — Metrics và hình kết quả thực tế

Accuracy=$\#correct/N$. Precision/Recall/F1 báo riêng mỗi lớp; Macro-F1 là trung bình không trọng số của F1 lớp, Weighted-F1 theo support. Negative chiếm đa số, nên một bộ dự đoán toàn negative vẫn có Accuracy đáng kể trong khi bỏ qua neutral/positive. Majority comparator lấy lớp nhiều nhất từ train. Classification report có đủ Precision/Recall/F1/Support/Macro/Weighted.

Confusion matrix: hàng=true, cột=predicted. Hình count và chuẩn hóa theo hàng có cùng checkpoint. Per-class/curves/examples của seed trung vị phải phân biệt với bảng mean±SD 3 seeds.
""")
code("""
generate_results(RUN_ROOT)
write_report(RUN_ROOT)
display(pd.read_csv(RUN_ROOT / "results/tables/experiment_comparison.csv").fillna("withheld"))
display(pd.read_csv(RUN_ROOT / "results/tables/final_architecture.csv"))
print((RUN_ROOT / "models/final/model_summary.txt").read_text().replace(RNN.__name__, "RNN").replace("simple_rnn", "rnn"))
for name in ("F09_architecture", "F10_model_summary", "F11_loss", "F12_accuracy", "F13_macro_f1", "F14_confusion_count", "F15_confusion_normalized", "F16_per_class_f1", "F20_configuration_comparison", "F21_quality_cost", "F06_representation_rates"):
    figure(name)
metric_path = RUN_ROOT / f"results/metrics/{selection['winner_config']}_seed-{selection['demo_seed']}_test.json"
print(read_json(metric_path)["classification_report_text"])
""")
md("""
## Phase 8 — Phân tích lỗi dựa trên evidence

Đọc các hướng nhầm positive/neutral/negative từ CM; dùng recall/F1 xác định lớp nào dễ hơn trong run này, không chỉ support. Slices độ dài dùng ngưỡng quartile **từ train**, phủ định dùng rule đánh dấu `not/no/never`; đây là phân nhóm mô tả, chưa chứng minh nguyên nhân. So sánh cùng support/class mix, không gán mọi lỗi ở slice cho length/negation.

CSV `manual_error_review.csv` có case sampling theo confusion pair, đúng từng class, low margin. Hai người annotate sarcasm/negation mất nghĩa/cleaning artifacts; các cột ban đầu để trống có chủ đích. Sau khi annotate, báo agreement và số case thật; không bịa nhãn. Không dùng kết quả error analysis trên test để quay lại chọn model trong vòng này.
""")
code("""
display(Markdown((RUN_ROOT / "reports/error_analysis.md").read_text()))
display(pd.read_csv(RUN_ROOT / "results/tables/error_slices.csv"))
for name in ("F17_correct_examples", "F18_error_examples", "F19_low_confidence", "F22_length_errors"):
    figure(name)
review = pd.read_csv(RUN_ROOT / "results/predictions/manual_error_review.csv")
print("Manual review cases:", len(review), "— human annotation fields remain pending.")
display(review.head(5))
""")
md("""
## Phase 9 — Inference dùng nguyên preprocessing/tokenizer/checkpoint

Không fit tokenizer ở inference. Mapping lớp đọc từ bundle; probabilities tổng≈1. Ví dụ nhập mới không phải thêm test metric. Demo local có form và bar xác suất; lệnh trong README. Mô hình học tiếng Anh/tweet airline, chưa kiểm tra tiếng Việt/miền khác. Softmax chưa calibration; confidence thấp dùng margin để mô tả, không khẳng định xác suất đúng đã kiểm chứng.
""")
code("""
from airline_rnn.inference import SentimentPredictor
predictor = SentimentPredictor(RUN_ROOT)
examples = [
    "@united Thank you for the helpful service!",
    "@AmericanAir My flight was delayed again and nobody helped me.",
    "@Delta What time does flight 123 depart?",
]
inference_examples = [{"text": text, **predictor.predict(text)} for text in examples]
display(pd.DataFrame(inference_examples))
from airline_rnn.common import save_json
save_json(RUN_ROOT / "results/predictions/inference_examples.json", inference_examples)

# Change this text and re-run this cell for an interactive notebook demo.
USER_TEXT = "@united I can't believe how helpful the staff were today!"
display(predictor.predict(USER_TEXT))
figure("F23_demo")  # actual browser screenshot if captured for this project
""")
md("""
## Phase 10 — Artifacts cho báo cáo và thuyết trình

- `reports/experiment_chapter.md`: chương thực nghiệm chứa số thật, nguồn dữ liệu, protocol, kết quả và hạn chế.
- `reports/source_audit.md`: đối chiếu kiến trúc/population và overlap nguồn.
- `reports/error_analysis.md`: phân tích mô tả CM/slices, giới hạn causal interpretation.
- `results/figures/`: PNG300dpi + PDF; `manifest.json` ghi mục đích và scope từng hình.
- `results/tables/`: Dataset, split/exclusion, training config, architecture Input/Output/Parameters, comparison, per-class, error slices.
- `results/metrics/`: full JSON/classification reports/frozen selection; `results/histories/`: history từng run.
- `models/final/`: checkpoint, tokenizer, labels, config, representation metadata và checksum manifest.
- README: môi trường, cách chạy mới/cached, Colab/Drive, demo, chính sách holdout.

Cấu trúc báo cáo đầy đủ và outline thuyết trình lưu trong `reports/`. Báo cáo mẫu chỉ định hướng cách tổ chức; không lấy kết luận CIFAR10/GCN của mẫu làm kết quả sentiment.
""")
code("""
display(Markdown((RUN_ROOT / "reports/experiment_chapter.md").read_text()))
manifest = read_json(RUN_ROOT / "results/figures/manifest.json")
display(pd.DataFrame(manifest).T[["png", "purpose", "scope"]])
print("All outputs saved under:", RUN_ROOT)
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12.14"},
                     "colab": {"provenance": []}}
nbf.validate(notebook)
path = ROOT / "notebooks/Airline_RNN_Complete.ipynb"
nbf.write(notebook, path)
print(path)
