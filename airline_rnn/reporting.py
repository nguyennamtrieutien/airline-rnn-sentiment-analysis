from __future__ import annotations

import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import initialize, read_json, save_json
from .data import load_benchmark, load_raw
from .model import RNN
from .text import clean_text, source_clean

COLORS = {"negative": "#454545", "neutral": "#CC9C38", "positive": "#306DA6"}
BLUE, GOLD = "#306DA6", "#CC9C38"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11, "figure.dpi": 110,
                         "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": False, "legend.frameon": False})


def export_figure(root, figure, name, purpose, scope):
    folder = root / "results/figures"
    figure.savefig(folder / f"{name}.png", bbox_inches="tight", facecolor="white")
    figure.savefig(folder / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    path = root / "results/figures/manifest.json"
    manifest = read_json(path) if path.exists() else {}
    manifest[name] = {"png": f"{name}.png", "pdf": f"{name}.pdf", "purpose": purpose, "scope": scope}
    save_json(path, manifest)


def md_table(frame, decimals=4):
    frame = frame.copy()
    for column in frame:
        if pd.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].map(lambda value: f"{value:.{decimals}f}" if pd.notna(value) else "—")
    escape = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(map(escape, frame.columns)) + " |", "| " + " | ".join(["---"] * len(frame.columns)) + " |"]
    lines += ["| " + " | ".join(map(escape, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(lines)


def readable_summary(table, split):
    def value(row, metric, percentage=False):
        mean, std = row[f"{split}_{metric}_mean"], row.get(f"{split}_{metric}_std", 0)
        scale, suffix = (100, "%") if percentage else (1, "")
        result = f"{mean * scale:.2f}{suffix}" if percentage else f"{mean:.4f}"
        if row.seeds > 1:
            result += f" ± {std * scale:.2f}" if percentage else f" ± {std:.4f}"
        return result
    result = pd.DataFrame({"Configuration": table.config_id, "Benchmark": table.benchmark, "Seeds": table.seeds})
    result[f"{split.title()} Accuracy"] = [value(row, "accuracy", True) for _, row in table.iterrows()]
    result[f"{split.title()} Macro-F1"] = [value(row, "macro_f1") for _, row in table.iterrows()]
    return result


def table_figure(root, frame, name, title, purpose, scope, widths=None):
    frame = frame.copy().astype(str)
    column_widths = widths or [1 / len(frame.columns)] * len(frame.columns)
    for column, relative_width in zip(frame.columns, column_widths):
        if column in ("Tweet", "P0"):
            # DejaVu Sans does not contain astral emoji. Show explicit code points
            # in the figure, while keeping original Unicode intact in the CSV.
            def wrap(text):
                text = "".join(f"[U+{ord(char):X}]" if ord(char) > 0xFFFF else char for char in text)
                return "\n".join(textwrap.wrap(" ".join(text.split()), width=max(15, int(relative_width * 96))))
            frame[column] = frame[column].map(wrap)
    figure, axis = plt.subplots(figsize=(13, max(2.3, 1.4 + len(frame) * 1.0)))
    axis.axis("off")
    axis.set_title(title, loc="left", pad=16)
    table = axis.table(cellText=frame.values, colLabels=frame.columns, loc="center",
                       cellLoc="left", colLoc="left", colWidths=widths)
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#D6D6D6")
        cell.set_linewidth(0.4)
        cell.set_height(0.09 if row == 0 else min(0.85 / max(len(frame), 1), 0.35))
        cell.set_facecolor("#EAF0F5" if row == 0 else "white")
    export_figure(root, figure, name, purpose, scope)


def generate_eda(root):
    root = initialize(root)
    style()
    raw = load_raw(root)
    profile = read_json(root / "results/metrics/dataset_profile.json")
    names = ["negative", "neutral", "positive"]
    counts = raw.airline_sentiment.value_counts().reindex(names)
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    bars = axis.bar(names, counts, color=[COLORS[n] for n in names])
    axis.bar_label(bars, labels=[f"{count:,}\n{count / len(raw):.1%}" for count in counts], padding=5)
    axis.set_ylim(0, counts.max() * 1.2)
    axis.set_ylabel("Tweets (count)")
    axis.set_title("Raw dataset: sentiment distribution (N=14,640)")
    export_figure(root, figure, "F01_raw_sentiment", "Class imbalance và population ba lớp gốc", "raw CSV")
    stages = pd.DataFrame([[9178, 3099, 2363], [9178, 0, 2363], [4178, 0, 2363]],
                          index=["Raw", "Remove neutral", "Remove first 5k negative"], columns=names)
    stages.to_csv(root / "results/tables/source_filter_counts.csv")
    figure, axis = plt.subplots(figsize=(9, 4.5))
    bottom = np.zeros(3)
    for name in names:
        axis.bar(stages.index, stages[name], bottom=bottom, color=COLORS[name], label=name)
        bottom += stages[name].to_numpy()
    axis.set_ylabel("Tweets (count)"); axis.set_title("Source notebook changes the benchmark population")
    axis.legend(ncols=3)
    export_figure(root, figure, "F02_source_filtering", "Thể hiện tác động lọc population trong nguồn", "raw/source binary")
    airline_counts = raw.airline.value_counts().sort_values()
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.barh(airline_counts.index, airline_counts.values, color=BLUE)
    axis.bar_label(bars, padding=4); axis.set_xlim(0, airline_counts.max() * 1.15)
    axis.set_xlabel("Tweets (count)"); axis.set_title("Raw dataset: airline distribution")
    export_figure(root, figure, "F03_airlines", "Độ không đồng đều theo airline", "raw CSV")
    source = load_benchmark(root, "binary_source")
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for axis, frame, title in zip(axes, (raw, source), ("Raw: 14,640 tweets", "Source subset: 6,541 tweets")):
        cross = pd.crosstab(frame.airline, frame.airline_sentiment).reindex(columns=names, fill_value=0)
        cross.to_csv(root / f"results/tables/airline_sentiment_{'raw' if len(frame) == len(raw) else 'source'}.csv")
        left = np.zeros(len(cross))
        for name in names:
            axis.barh(cross.index, cross[name], left=left, color=COLORS[name], label=name)
            left += cross[name].to_numpy()
        axis.set_xlabel("Tweets (count)"); axis.set_title(title)
    axes[1].legend(ncols=3, loc="lower right")
    export_figure(root, figure, "F04_airline_filtering", "Kiểm tra thay đổi phân bố airline sau sampling", "raw/source binary")
    stats = pd.read_csv(root / "data/processed/C0/train_representation_stats.csv") if (root / "data/processed/C0/train_representation_stats.csv").exists() else None
    if stats is not None:
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.hist(stats.token_length, bins=np.arange(0, 64, 2), color=BLUE, edgecolor="white")
        for length, line in ((20, "--"), (40, "-"), (60, ":")):
            axis.axvline(length, linestyle=line, color="#454545", label=f"T={length}")
        axis.set_xlabel("Tokens before padding/OOV replacement"); axis.set_ylabel("Train tweets (count)")
        axis.set_title(f"C0 training lengths (N={len(stats):,}); thresholds registered before training")
        axis.legend()
        export_figure(root, figure, "F05_train_lengths", "Cơ sở giải thích sequence coverage", "C0 train only")
    examples = raw.loc[raw.row_id.isin([1, 17, 18, 21, 28]), ["row_id", "text"]].copy()
    examples["source_clean"] = examples.text.map(source_clean)
    examples["P0_clean"] = examples.text.map(clean_text)
    examples.to_csv(root / "results/tables/preprocessing_examples.csv", index=False)
    display = pd.DataFrame({"Row ID": examples.row_id, "Tweet": examples.text,
                            "P0": examples.P0_clean.map(lambda t: "\n".join(textwrap.wrap(t, 55)))})
    table_figure(root, display, "F07_preprocessing", "Raw vs P0 (astral emoji shown as explicit U+ codes)",
                 "Các biến đổi preprocessing có thể kiểm tra", "raw examples; no model prediction", [0.06, 0.5, 0.44])
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.axis("off")
    nodes = [(0.04, 0.73, "Raw CSV\nchecksum + audit"), (0.29, 0.73, "Group-aware split\ntrain / val / test"),
             (0.54, 0.73, "Fit tokenizer\nTRAIN ONLY"), (0.79, 0.73, "Tokens → PAD / mask\nEmbedding → RNN"),
             (0.79, 0.25, "Train / validate\ncheckpoint: val Macro-F1"), (0.54, 0.25, "Freeze config + seed\nrelease holdout"),
             (0.29, 0.25, "Final test\nmetrics + error analysis"), (0.04, 0.25, "Saved inference bundle\nsame preprocessing")]
    from matplotlib.patches import FancyBboxPatch
    for x, y, label in nodes:
        axis.add_patch(FancyBboxPatch((x + .085 - .106, y - .096), .212, .192,
                       boxstyle="round,pad=0.004", transform=axis.transAxes,
                       facecolor="#EAF0F5", edgecolor="#A3B7C9"))
        axis.text(x + 0.085, y, label, transform=axis.transAxes, ha="center", va="center", fontsize=10)
    for first, second in zip(nodes, nodes[1:]):
        direction = np.sign(second[0] - first[0])
        start = (first[0] + .085 + direction * .113, first[1])
        end = (second[0] + .085 - direction * .113, second[1])
        if not direction:
            start = (first[0] + .085, first[1] - .11)
            end = (second[0] + .085, second[1] + .11)
        axis.annotate("", xy=end, xytext=start, xycoords="axes fraction",
                      arrowprops={"arrowstyle": "->", "color": "#6F6F6F", "shrinkA": 0, "shrinkB": 0})
    axis.set_title("Main three-class RNN pipeline — A1 legacy protocol is separate")
    export_figure(root, figure, "F08_pipeline", "Split-before-fit và test-release policy", "C0/B")
    split_rows = [{"benchmark": benchmark, "split": split, **counts, "total": sum(counts.values())}
                  for benchmark, info in profile["benchmarks"].items() for split, counts in info["counts"].items()]
    pd.DataFrame(split_rows).fillna(0).to_csv(root / "results/tables/dataset_splits.csv", index=False)
    exclusion_summary = []
    for name in ("binary", "three_class"):
        table = pd.read_csv(root / f"data/interim/exclusions_{name}.csv")
        for reason, count in table.reason.value_counts().items():
            exclusion_summary.append({"benchmark": name, "reason": reason, "rows": int(count)})
    pd.DataFrame(exclusion_summary).to_csv(root / "results/tables/exclusions.csv", index=False)
    return profile


def confusion_figure(root, matrix, names, normalized, scope):
    matrix = np.asarray(matrix)
    shown = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1) if normalized else matrix
    figure, axis = plt.subplots(figsize=(6.5, 5.5))
    image = axis.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalized else None)
    figure.colorbar(image, ax=axis, label="Proportion within true class" if normalized else "Tweets (count)")
    for row in range(len(names)):
        for column in range(len(names)):
            text = f"{shown[row, column]:.1%}" if normalized else str(matrix[row, column])
            axis.text(column, row, text, ha="center", va="center", fontsize=12,
                      color="white" if shown[row, column] > shown.max() * .5 else "#202020")
    axis.set_xticks(range(len(names)), names); axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted sentiment"); axis.set_ylabel("True sentiment")
    axis.set_title("Final test confusion matrix" + (" — row normalized" if normalized else " — count"))
    export_figure(root, figure, "F15_confusion_normalized" if normalized else "F14_confusion_count",
                  "Phân bố nhầm lớp theo tỷ lệ" if normalized else "Số mẫu theo từng cặp nhầm", scope)


def generate_results(root):
    root = initialize(root); style()
    selection = read_json(root / "results/metrics/final_selection.json")
    if not (root / "results/metrics/test_release.json").exists():
        raise RuntimeError("Final results require released test evaluation")
    config_id, seed = selection["winner_config"], selection["demo_seed"]
    directory = root / f"models/runs/{config_id}/seed-{seed}"
    run = read_json(directory / "run.json")
    metrics = read_json(root / f"results/metrics/{config_id}_seed-{seed}_test.json")
    predictions = pd.read_csv(root / f"results/predictions/{config_id}_seed-{seed}_test.csv", keep_default_na=False)
    scope = f"{config_id}, seed={seed}, {run['config']['benchmark']}, test N={len(predictions)}"
    architecture = pd.DataFrame(read_json(directory / "architecture.json"))
    # Display labels are RNN; immutable run metadata retains its original framework identity.
    recurrent = architecture.type == RNN.__name__
    architecture.loc[recurrent, ["layer", "type"]] = ["rnn", "RNN"]
    input_shapes = [f"(None, {read_json(directory / 'representation.json')['sequence_length']})"] + architecture.output_shape.tolist()[:-1]
    architecture.insert(2, "input_shape", input_shapes)
    architecture.to_csv(root / "results/tables/final_architecture.csv", index=False)
    save_json(root / "results/tables/final_training_configuration.json", run["config"])
    table_figure(root, architecture.rename(columns={"layer": "Layer", "type": "Type", "input_shape": "Input", "output_shape": "Output", "parameters": "Parameters"}),
                 "F10_model_summary", "Actual RNN architecture and parameters", "Kiến trúc model đã chạy", scope)
    figure, axis = plt.subplots(figsize=(11, 7)); axis.axis("off")
    for i, row in enumerate(architecture.itertuples()):
        y = 0.94 - i * 0.125
        axis.text(.5, y, f"{row.type} — {row.layer}\n{row.output_shape}; parameters={row.parameters:,}",
                  ha="center", va="center", transform=axis.transAxes, fontsize=10,
                  bbox={"boxstyle": "round,pad=.4", "facecolor": "#EAF0F5", "edgecolor": "#A3B7C9"})
        if i:
            axis.annotate("", xy=(.5, y + .049), xytext=(.5, y + .077), xycoords="axes fraction", arrowprops={"arrowstyle": "->"})
    axis.set_title(f"Implemented model: {config_id}, seed={seed}, RNN/tanh (no gates)")
    export_figure(root, figure, "F09_architecture", "Pipeline layer shapes và parameter count", scope)
    history = pd.read_csv(directory / "history.csv")
    for name, train_key, val_key, title, ylabel in (
        ("F11_loss", "train_loss", "val_loss", "Training and validation loss", "Cross-entropy"),
        ("F12_accuracy", "train_accuracy", "val_accuracy", "Training and validation accuracy", "Accuracy (%)"),
        ("F13_macro_f1", "train_eval_macro_f1", "val_macro_f1", "Train-eval and validation Macro-F1", "Macro-F1 (%)")):
        figure, axis = plt.subplots(figsize=(9, 4.5)); scale = 1 if name == "F11_loss" else 100
        axis.plot(history.epoch, history[train_key] * scale, color=BLUE, marker="o", markersize=3, label="Train" if name != "F13_macro_f1" else "Train inference mode")
        axis.plot(history.epoch, history[val_key] * scale, color=GOLD, linestyle="--", marker="s", markersize=3, label="Validation")
        axis.axvline(run["best_epoch"], color="#555555", linestyle=":", label=f"Best epoch={run['best_epoch']}")
        axis.set_xlabel("Epoch"); axis.set_ylabel(ylabel); axis.set_title(f"{title} — {config_id}, seed={seed}")
        axis.set_xticks(history.epoch); axis.legend()
        if scale == 100: axis.set_ylim(0, 100)
        else: axis.set_ylim(bottom=0)
        export_figure(root, figure, name, "Hội tụ và generalization gap; train fit có dropout, val không có", scope.replace("test", "train/validation"))
    confusion_figure(root, metrics["confusion_matrix"], metrics["label_names"], False, scope)
    confusion_figure(root, metrics["confusion_matrix"], metrics["label_names"], True, scope)
    report = pd.DataFrame(metrics["classification_report"]).T
    per_class = report.loc[metrics["label_names"], ["precision", "recall", "f1-score", "support"]].reset_index(names="sentiment")
    per_class["support"] = per_class.support.astype(int)
    per_class.to_csv(root / "results/tables/final_per_class.csv", index=False)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar(per_class.sentiment, per_class["f1-score"] * 100, color=[COLORS[n] for n in per_class.sentiment])
    axis.bar_label(bars, labels=[f"{f1:.1%}\nN={int(support)}" for f1, support in zip(per_class["f1-score"], per_class.support)], padding=4)
    axis.set_ylim(0, 110); axis.set_ylabel("F1-score (%)"); axis.set_title("Final model: per-class test F1 and support")
    export_figure(root, figure, "F16_per_class_f1", "Chất lượng giữa sentiment classes", scope)
    for correct, name, title in ((True, "F17_correct_examples", "Correct predictions — seeded class samples"),
                                 (False, "F18_error_examples", "Incorrect predictions — seeded confusion-pair samples")):
        cases = []
        subset = predictions[predictions.correct == correct]
        grouping = ["true_label"] if correct else ["true_label", "predicted_label"]
        for _, group in subset.groupby(grouping, sort=True):
            cases.append(group.sample(1, random_state=42))
        if cases:
            samples = pd.concat(cases).head(6)
            display = pd.DataFrame({"True": samples.true_label, "Predicted": samples.predicted_label,
                                    "Score": samples.confidence.map(lambda x: f"{x:.1%}"), "Tweet": samples.raw_text})
            table_figure(root, display, name, title, "Ví dụ prediction thực; không thay metric tổng thể", scope, [.10, .11, .09, .70])
    low = predictions.sort_values(["top2_margin", "row_id"]).head(5)
    display = pd.DataFrame({"True": low.true_label, "Predicted": low.predicted_label,
                            "Margin": low.top2_margin.map(lambda x: f"{x:.3f}"), "Tweet": low.raw_text})
    table_figure(root, display, "F19_low_confidence", "Smallest top-2 probability margins (uncalibrated softmax)",
                 "Các quyết định có softmax margin thấp", scope, [.10, .11, .09, .70])
    summary = pd.read_csv(root / "results/tables/validation_summary.csv")
    comparison = summary[summary.benchmark == "three_class"]
    if not comparison.empty:
        figure, axis = plt.subplots(figsize=(10, 5))
        bars = axis.bar(comparison.config_id, comparison.val_macro_f1_mean * 100,
                        yerr=comparison.val_macro_f1_std * 100, capsize=4, color=BLUE)
        axis.bar_label(bars, labels=[f"{value:.1%}" for value in comparison.val_macro_f1_mean], padding=6)
        axis.set_ylim(0, 100); axis.set_ylabel("Validation Macro-F1 (%)")
        axis.set_title("Registered RNN configurations — mean ± SD across 3 training seeds")
        export_figure(root, figure, "F20_configuration_comparison", "So sánh validation cùng split; SD không phải CI", "C0/B validation")
        figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
        positions = np.arange(len(comparison))
        axes[0].barh(positions, comparison.val_macro_f1_mean * 100, xerr=comparison.val_macro_f1_std * 100, capsize=3, color=BLUE)
        axes[0].set_yticks(positions, comparison.config_id); axes[0].set_xlim(0, 100)
        axes[0].set_xlabel("Validation Macro-F1 (%) — mean ± SD")
        bars = axes[1].barh(positions, comparison.training_seconds_median, color=GOLD)
        axes[1].bar_label(bars, labels=[f"{seconds:.1f}s | {parameters:,} params" for seconds, parameters in zip(comparison.training_seconds_median, comparison.parameters)], padding=5, fontsize=9)
        axes[1].set_xlim(0, comparison.training_seconds_median.max() * 1.65)
        axes[1].set_xlabel("Median training wall time incl. epoch metrics (s)")
        axes[0].invert_yaxis(); figure.suptitle("Quality and training cost — same CPU/runtime profile")
        figure.tight_layout()
        export_figure(root, figure, "F21_quality_cost", "Tradeoff chất lượng và wall time/parameter count", "C0/B validation; same hardware")
    representation_rows = []
    for path in (root / "data/processed").glob("*/representation.json"):
        metadata = read_json(path)
        if metadata["benchmark"] == "three_class":
            for split, stats in metadata["stats"].items():
                representation_rows.append({"config_id": metadata["config_id"], "split": split,
                                            "oov_rate": stats["oov_token_rate"], "truncation_rate": stats["truncation_rate"]})
    representation_table = pd.DataFrame(representation_rows)
    representation_table.to_csv(root / "results/tables/representation_comparison.csv", index=False)
    if not representation_table.empty:
        reference = representation_table[representation_table.split == "validation"]
        figure, axis = plt.subplots(figsize=(10, 4.5)); x = np.arange(len(reference))
        axis.bar(x - .18, reference.oov_rate * 100, .36, color=BLUE, label="OOV token rate")
        axis.bar(x + .18, reference.truncation_rate * 100, .36, color=GOLD, label="Truncated tweet rate")
        axis.set_xticks(x, reference.config_id); axis.set_ylabel("Rate (%)"); axis.set_ylim(bottom=0)
        axis.set_title("Validation representation: OOV and truncation"); axis.legend()
        export_figure(root, figure, "F06_representation_rates", "Thông tin bị thay thế/cắt ở từng configuration", "C0/B validation")
    return generate_error_analysis(root, predictions, metrics, run, selection)


def generate_error_analysis(root, predictions, metrics, run, selection):
    from .training import classification_metrics
    predictions = predictions.copy()
    train_stats = pd.read_csv(root / f"data/processed/{run['config']['id']}/train_representation_stats.csv")
    thresholds = sorted(set(float(train_stats.token_length.quantile(q)) for q in (.25, .5, .75)))
    predictions["length_bin"] = pd.cut(predictions.token_length, [-1, *thresholds, float("inf")]).astype(str)
    predictions["flag_negation"] = predictions.clean_text.str.contains(r"\b(?:not|no|never)\b", regex=True)
    predictions["flag_url"] = predictions.clean_text.str.contains(r"\burlmarker\b", regex=True)
    predictions["flag_any_oov"] = predictions.unknown_tokens > 0
    slices = []
    names = metrics["label_names"]
    for feature in ("length_bin", "flag_negation", "flag_any_oov", "truncated"):
        for value, frame in predictions.groupby(feature, sort=True):
            probabilities = frame[[f"prob_{name}" for name in names]].to_numpy()
            labels = frame.true_label.map({name: i for i, name in enumerate(names)}).to_numpy(dtype="int32")
            measured = classification_metrics(labels, probabilities, names)
            slices.append({"slice": feature, "value": str(value), "support": len(frame),
                           "accuracy": measured["accuracy"], "macro_f1": measured["macro_f1"],
                           "error_rate": 1 - measured["accuracy"],
                           **{f"support_{name}": int((frame.true_label == name).sum()) for name in names}})
    slices = pd.DataFrame(slices)
    slices.to_csv(root / "results/tables/error_slices.csv", index=False)
    length = slices[slices.slice == "length_bin"]
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(length.value, length.error_rate * 100, color=BLUE)
    axis.bar_label(bars, labels=[f"{rate:.1%}\nN={n}" for rate, n in zip(length.error_rate, length.support)], padding=4)
    axis.set_ylim(0, 110); axis.set_ylabel("Test error rate (%)"); axis.set_xlabel("Token length bin (train quartiles)")
    axis.set_title("Observed test errors by length — class mix also varies")
    export_figure(root, figure, "F22_length_errors", "Sai khác theo độ dài, có support và class mix đi kèm", f"{run['config']['id']} test")
    reviews = []
    for _, frame in predictions[~predictions.correct].groupby(["true_label", "predicted_label"]):
        reviews.append(frame.sample(min(10, len(frame)), random_state=42).assign(review_reason="confusion_pair"))
    for _, frame in predictions[predictions.correct].groupby("true_label"):
        reviews.append(frame.sample(min(5, len(frame)), random_state=42).assign(review_reason="correct_class_sample"))
    reviews.append(predictions.sort_values(["top2_margin", "row_id"]).head(10).assign(review_reason="low_margin"))
    review = pd.concat(reviews).drop_duplicates("row_id").copy()
    review["reviewer_1_group"] = ""; review["reviewer_2_group"] = ""; review["review_notes"] = ""
    review_path = root / "results/predictions/manual_error_review.csv"
    if review_path.exists():
        existing = pd.read_csv(review_path, keep_default_na=False).set_index("row_id")
        # Re-running reports must never erase human annotation already supplied.
        for column in ("reviewer_1_group", "reviewer_2_group", "review_notes"):
            if column in existing:
                review[column] = review.row_id.map(existing[column]).fillna("")
    review.to_csv(review_path, index=False)
    matrix = np.asarray(metrics["confusion_matrix"])
    notes = []
    for i, name in enumerate(names):
        row = matrix[i].copy(); row[i] = -1
        j = int(row.argmax()); total = int(matrix[i].sum()); errors = total - int(matrix[i, i])
        if errors:
            notes.append(f"- {name}: {errors}/{total} mẫu sai; cặp nhầm nhiều nhất {name} → {names[j]}: {matrix[i,j]} mẫu ({matrix[i,j]/total:.2%} support của lớp thật).")
        else:
            notes.append(f"- {name}: không có nhầm lớp trên {total} mẫu test này.")
    best_class = max(names, key=lambda name: metrics["classification_report"][name]["f1-score"])
    analysis = f"""# Phân tích lỗi thực tế — {run['config']['id']}, seed {run['seed']}

Checkpoint được chọn bằng validation trước test; N={len(predictions)}. Các tỷ lệ này mô tả một checkpoint, không thay bảng trung bình ba seeds.

## Nhầm lớp

{chr(10).join(notes)}

Lớp có F1 cao nhất của checkpoint này là **{best_class}**, F1={metrics['classification_report'][best_class]['f1-score']:.2%}. Số lượng mẫu đúng tuyệt đối không được dùng để gọi lớp dễ nhất.

## Độ dài, OOV và negation

Các ngưỡng độ dài lấy từ quartile train: {thresholds}. Bảng `results/tables/error_slices.csv` có support và phân bố class để tránh diễn giải sai do class mix.

{md_table(slices)}

Các chênh lệch là quan sát trên test; chưa chứng minh độ dài/OOV/negation gây ra lỗi. Negation flags dựa trên từ not/no/never trong clean text, cần kiểm tra ngữ cảnh khi phân tích trường hợp cụ thể.

## Cleaning, sarcasm và manual review

Cleaner P0 bỏ emoji/dấu câu, giữ phủ định và nội dung hashtag. Bảng before/after chứng minh phép biến đổi, chưa chứng minh ảnh hưởng F1 vì chưa chạy preprocessing ablation. Tập review có {len(review)} dòng, sampling seed=42, gồm cặp nhầm, mẫu đúng và margin thấp.

`results/predictions/manual_error_review.csv` có cột hai người review và notes để nhóm ghi các nhãn negation, sarcasm, mixed sentiment, slang/typo, missing context, truncation hoặc uncertain. **Chưa có nhãn sarcasm được con người xác minh, nên chưa có kết luận về sarcasm.** Không tự gán sarcasm từ một từ như great.

Softmax và top-2 margin chưa được calibration. Confidence thấp không đồng nghĩa prediction sai; confidence cao cũng có thể sai.

## Giới hạn

Test đã được mở cho phân tích này; không dùng các lỗi để tiếp tục chọn hyperparameter trên cùng holdout. Các cải tiến từ review được ghi future work hoặc kiểm tra trên validation/holdout mới.
"""
    (root / "reports/error_analysis.md").write_text(analysis, encoding="utf-8")
    return metrics


def write_report(root):
    root = initialize(root)
    profile = read_json(root / "results/metrics/dataset_profile.json")
    selection = read_json(root / "results/metrics/final_selection.json")
    val = pd.read_csv(root / "results/tables/validation_summary.csv")
    test = pd.read_csv(root / "results/tables/test_summary.csv")
    per_class = pd.read_csv(root / "results/tables/final_per_class.csv")
    split = pd.read_csv(root / "results/tables/dataset_splits.csv")
    excluded = pd.read_csv(root / "results/tables/exclusions.csv")
    run = read_json(root / f"models/runs/{selection['winner_config']}/seed-{selection['demo_seed']}/run.json")
    environment = read_json(root / f"models/runs/{selection['winner_config']}/seed-{selection['demo_seed']}/environment.json")
    source = read_json(root / "sources/historical_outputs.json")
    source_manifest = load_benchmark(root, "binary_source")
    crossed_ids = int((source_manifest.groupby("tweet_id").split.nunique() > 1).sum())
    crossed_text = int((source_manifest.groupby("clean_text").split.nunique() > 1).sum())
    source_audit = f"""# Audit nguồn Kaggle

Nguồn: {source['source_url']}, Version {source['version']}, scriptVersionId {source['script_version_id']}. Code và output lịch sử đã được đọc; file nguồn lưu trong `sources/`, không thực thi LSTM trong project này.

- Mô hình nguồn: LSTM196, Embedding4000×128, SpatialDropout0.5; recurrent/input dropout0.3; head Dropout0.2 → Dense100/ReLU → Dropout0.4 → Dense2/softmax.
- Nguồn bỏ neutral và 5.000 negative đầu, còn 6.541 tweet; tokenizer fit toàn bộ subset trước split; train3662/val916/test1963; Adam (LR không tường minh), batch32,20epochs, không ES/checkpoint.
- Output nguồn: test accuracy {source['test_accuracy']:.6%}, loss {source['test_loss']:.6f}; best val accuracy {source['best_val_accuracy']:.2%} ở epoch11. Đây không phải kết quả RNN nhóm.
- A1 sử dụng source cleaner/ordering/padding/preprocessing, nhưng RNN/tanh thay LSTM, seed42 và Adam0.001 tường minh là lựa chọn triển khai mới. Tokenizer tương đương Keras đã được kiểm tra trên toàn6.541tweet; maxlen tính lại=31. Không gọi A1 exact reproduction.
- Trong split A1 tạo lại, có {crossed_ids} tweet IDs và {crossed_text} chuỗi clean text xuất hiện ở hơn một split. Đây là kiểm tra overlap trên file hiện có, không đo mức tăng metric do leakage.
- Phiên bản TensorFlow/Keras/CPU/RAM lịch sử không đủ thông tin. Environment hiện tại lưu riêng, không gán version mới cho output cũ.
"""
    (root / "reports/source_audit.md").write_text(source_audit, encoding="utf-8")
    winner_test = test[test.config_id == selection["winner_config"]].iloc[0]
    b = val[val.benchmark == "three_class"]
    b_text = md_table(readable_summary(b, "val")) if not b.empty else "Bộ minimum chưa có benchmark ba lớp."
    class_rows = []
    for config_id in dict.fromkeys(("C0", selection["winner_config"])):
        for path in (root / "results/metrics").glob(f"{config_id}_seed-*_test.json"):
            metric = read_json(path)
            for name in metric["label_names"]:
                class_rows.append({"config_id": config_id, "seed": metric["seed"], "sentiment": name,
                                   "f1": metric["classification_report"][name]["f1-score"]})
    class_summary = pd.DataFrame(class_rows).groupby(["config_id", "sentiment"], sort=False).agg(
        f1_mean=("f1", "mean"), f1_std=("f1", "std"), seeds=("seed", "count")).reset_index()
    class_summary.to_csv(root / "results/tables/baseline_final_per_class.csv", index=False)
    tradeoff_notes = ""
    if selection["winner_config"] != "C0" and "C0" in class_summary.config_id.values and run["config"]["benchmark"] == "three_class":
        indexed = class_summary.set_index(["config_id", "sentiment"])
        notes = []
        for name in ("negative", "neutral", "positive"):
            baseline = indexed.loc[("C0", name), "f1_mean"]
            final = indexed.loc[(selection["winner_config"], name), "f1_mean"]
            notes.append(f"{name}: {baseline:.4f} → {final:.4f}")
        tradeoff_notes = "F1 trung bình C0 → cấu hình cuối: " + "; ".join(notes) + ". Đây là tradeoff quan sát trên test sau khi selection đã khóa, không dùng để quay lại chọn model."
    experiment_notes = ""
    if selection["tier"] != "minimum":
        lookup = val.set_index("config_id")
        a2_seed = read_json(root / "models/runs/A2/seed-42/run.json")
        a2_test = read_json(root / "results/metrics/A2_seed-42_test.json")
        a1_test = test[test.config_id == "A1"].iloc[0]
        experiment_notes = f"""### Phân tích các thay đổi đã đo

A1 đạt test Accuracy {a1_test.test_accuracy_mean:.2%} với RNN, còn nguồn công bố {source['test_accuracy']:.2%} với LSTM. Kiến trúc và seed/runtime khác nhau nên chênh lệch này không xác nhận hay bác bỏ khả năng tái lập cùng mô hình nguồn. A2 biến thiên lớn giữa seeds: test Macro-F1 SD={test[test.config_id == 'A2'].iloc[0].test_macro_f1_std:.4f}. Seed42 dừng ở epoch {a2_seed['epochs_trained']}, chọn checkpoint epoch {a2_seed['best_epoch']}; F1 positive trên test chỉ {a2_test['classification_report']['positive']['f1-score']:.4f}. Không chọn riêng seed3407 tốt hơn để che biến thiên này; chưa có ablation đủ để quy nguyên nhân cho cleaner, leakage, dropout hoặc ES.

- **Sequence length:** T20 có mean validation Macro-F1 {lookup.loc['B1-20', 'val_macro_f1_mean']:.4f}, so với T40 {lookup.loc['C0', 'val_macro_f1_mean']:.4f}. T40 và T60 cho cùng validation metrics ở cả ba seeds trong lần chạy này; median wall time lần lượt {lookup.loc['C0', 'training_seconds_median']:.1f}s và {lookup.loc['B1-60', 'training_seconds_median']:.1f}s. Khi không có thêm token cần giữ và PAD được mask, T60 chỉ tăng chi phí ở setup này.
- **Hidden units:** H64/H128/H196 có mean validation Macro-F1 lần lượt {lookup.loc['B3-64', 'val_macro_f1_mean']:.4f}/{lookup.loc['B3-128', 'val_macro_f1_mean']:.4f}/{lookup.loc['C0', 'val_macro_f1_mean']:.4f}. Tăng H không tạo cải thiện đơn điệu; cần đọc cùng số parameters và thời gian F21.
- **Class weighting:** balanced tăng mean validation Macro-F1 từ {lookup.loc['C0', 'val_macro_f1_mean']:.4f} lên {lookup.loc['B5-balanced', 'val_macro_f1_mean']:.4f}. Đây là cấu hình được chọn bằng validation. Ba seeds chưa đủ để khẳng định statistical significance.
"""
    config_table = pd.DataFrame({"Hyperparameter": list(run["config"]), "Value": [str(v) for v in run["config"].values()]})
    config_table.to_csv(root / "results/tables/training_configuration.csv", index=False)
    dataset_table = pd.DataFrame([
        {"Attribute": "Source", "Value": profile["source"]},
        {"Attribute": "Raw rows", "Value": profile["raw_rows"]},
        {"Attribute": "Raw columns", "Value": profile["raw_columns"]},
        {"Attribute": "Model input", "Value": "text"},
        {"Attribute": "Target", "Value": "airline_sentiment"},
        {"Attribute": "Main labels", "Value": "negative=0, neutral=1, positive=2"},
        {"Attribute": "Raw SHA-256", "Value": profile["sha256"]},
    ])
    dataset_table.to_csv(root / "results/tables/dataset.csv", index=False)
    comparison_table = val.merge(test.drop(columns=["seeds", "support"]), on=["config_id", "benchmark"], how="left")
    comparison_table["test_policy"] = comparison_table.config_id.map(lambda name: "final evaluation" if name in selection["test_configs"] else "withheld: validation-only ablation")
    comparison_table.to_csv(root / "results/tables/experiment_comparison.csv", index=False)
    total_seconds = float(pd.read_csv(root / "results/tables/validation_runs.csv").training_seconds.sum())
    max_length = pd.read_csv(root / f"data/processed/{selection['winner_config']}/train_representation_stats.csv").token_length.max()
    report = f"""# CHƯƠNG 3. THỰC NGHIỆM VÀ ĐÁNH GIÁ

## 3.1. Phạm vi và môi trường thực tế

Dự án giữ **RNN** làm mô hình chính. Notebook Kaggle dùng LSTM/binary là nguồn tham khảo, không phải cùng kiến trúc với thực nghiệm nhóm. Kết quả dưới đây được sinh từ các run hoàn tất, không có số Accuracy/F1 điền trước.

Môi trường thực thi lần này: {environment['os']}; Python {environment['python']}; TensorFlow {environment['versions']['tensorflow']}; Keras {environment['versions']['keras']}; CPU {environment['machine']}, {environment['cpu_physical_cores']} physical cores; RAM {environment['ram_bytes']/2**30:.1f}GiB; GPU {environment['gpu_devices'] or 'không có GPU TensorFlow'}. Float32, deterministic operations; intra/inter threads {environment['intra_threads']}/{environment['inter_threads']}. Tổng wall time training các run đăng ký (kể cả full-split metrics từng epoch): {total_seconds/60:.2f} phút. Thời gian không đại diện Colab GPU.

Notebook và dependencies hỗ trợ chuyển lên Colab; Google Drive dùng lưu artifacts nếu nhóm chọn. **Các kết quả hiện có được chạy trên máy cá nhân**, không ghi nhầm là Colab/T4.

## 3.2. Dataset, exclusions và split

Dataset raw: {profile['raw_rows']:,}tweet, {profile['raw_columns']}cột; SHA-256 `{profile['sha256']}`. Input chỉ là text, target sentiment; metadata hãng không làm feature. Duplicate ID raw={profile['duplicate_tweet_ids']}, duplicate raw text={profile['duplicate_raw_texts']}; các field confidence/gold/reason không đưa vào model.

{md_table(excluded, decimals=0)}

Nhóm duplicate tạo theo tweet ID, raw text chuẩn hóa và clean representation; conflicting-label groups cách ly; exact duplicate records bỏ; các nhóm còn lại không chéo split sạch. A1 giữ split nguồn riêng.

{md_table(split, decimals=0)}

Tỷ lệ trên dữ liệu sạch có thể lệch nhẹ70/15/15 do giữ group. Manifest lưu row IDs,group IDs và hashes. Source binary/A2/three-class có population khác nhau nên không so accuracy như cùng benchmark.

## 3.3. Kiến trúc và training protocol

Embedding → SpatialDropout → RNN(tanh) → Dropout → Dense100/ReLU → Dropout → softmax. Word vocabulary train-only ở A2/C0/B; PAD0,OOV1 và marker IDs ở P0; padding được mask ở C0/B. Sparse categorical cross-entropy; Adam LR0.001; batch32;max20epochs. Clean runs checkpoint theo valMacro-F1, ES patience5/min_delta0.001;A1 dùng epochcuối.

{md_table(config_table)}

Seeds42,2026,3407 trên cùng split. F1 tính trên toàn validation/train trong inference mode, không trung bình F1 minibatch. Balanced weights chỉ dùng train. Source code/config/split/checkpoint hashes được lưu. Matrix khóa trước test; chọn meanvalMacro-F1. Các config cách max dưới0.005 được ưu tiên ít parameters, sau đó median training time; đây là quy tắc thực dụng, không phải kiểm định tương đương.

Các bảng Accuracy dùng thang phần trăm; SD bên cạnh là điểm phần trăm. Macro-F1 dùng thang0–1. Run đơn hoặc majority baseline không báo SD như một ước lượng độ bất định.

## 3.4. Baseline và controlled experiments

{md_table(readable_summary(val, 'val'))}

A1 là adapted pipeline, A2 là clean binary, C0 là clean three-class. B1 thay T20/40/60, B3 thayH64/128/196, B5 thay classweightNone/balanced; mỗi variant khácC0 một factor. Embedding dimensions/dropout/batch/LR/split/budget giữ cố định.

Train token length lớn nhất ở representation cuối là {int(max_length)}. Trong benchmark P0 hiện có, T40 vàT60 không cắt tweet; do đó không diễn giải T60 là giữ thêm context nếu coverage như nhau. Với PAD được mask, đây chủ yếu là đối chứng padding/chi phí; T20 khảo sát truncation. SD là độ biến thiên initialization/shuffle/dropout qua ba training seeds, không phải CI do split khác nhau.

### Bảng B trên validation

{b_text}

{experiment_notes}

Không tự ghép các mức thắng. Config được chọn thực sự đã chạy: **{selection['winner_config']}**. Seed demo={selection['demo_seed']}, chọn validation Macro-F1 trung vị trước khi xem test. Nếu config khác có meanvalF1 nhỉnh hơn trong tolerance, việc ưu tiên model nhỏ hơn tuân theo quy tắc đặt trước.

## 3.5. Đánh giá cuối trên test

Test chỉ mở sau selection; A1/A2/C0/winner được đánh giá trong benchmark riêng. Các ablation khác không có test metric theo protocol.

{md_table(readable_summary(test, 'test'))}

Mô hình cuối {selection['winner_config']}: testAccuracy mean={winner_test.test_accuracy_mean:.2%}, SD={winner_test.test_accuracy_std:.2%}; testMacro-F1 mean={winner_test.test_macro_f1_mean:.2%}, SD={winner_test.test_macro_f1_std:.2%}. Đây là mean±SD qua training seeds trên cùng holdout.

### Per-class của checkpoint dùng demo (seed {selection['demo_seed']})

{md_table(per_class)}

Per-class/CM/hình ví dụ tương ứng checkpoint seed trung vị, không thay mean±SD tổng thể. Majority baseline lấy prior/lớp phổ biến từ train; không học nhãn test. Cần xem Macro-F1 và per-class recall cùng Accuracy vì negative chiếm đa số.

### Đối chiếu F1 từng lớp giữa C0 và cấu hình cuối

{md_table(class_summary)}

{tradeoff_notes}

## 3.6. Hội tụ và phân tích lỗi

HìnhF11/F12/F13 lấy history thật;bestepoch={run['best_epoch']};epochs chạy={run['epochs_trained']};stopreason={run['stop_reason']}. Trainfitloss có dropout/weighting;train_evalF1/valF1 dùng inference mode. Không coi chúng là cùng chế độ tính.

![Learning curves](../results/figures/F13_macro_f1.png)

![Confusion matrix](../results/figures/F14_confusion_count.png)

Phân tích chi tiết và số nhầm từng lớp ở `error_analysis.md`. Review cases ởCSV có cột cho hai người đánh dấu. Chưa có annotation sarcasm được con người xác minh; chưa có preprocessing ablation chứng minh ảnh hưởng cleaning lên metric. Không tạo kết luận trước hoặc gán nguyên nhân từ một vài ví dụ.

## 3.7. Đối chiếu nguồn và hạn chế

Nguồn báo testAccuracy90.32% của LSTM trên6.541tweet/two-class, vocabulary fitted before split, epoch20. RNN nhóm đổi recurrent mechanism;A2/C0 thay protocol/population. Chênh lệch số không phải một phép so sánh công bằng hoặc bằng chứng riêng về leakage. Source audit ghi rõ các khác biệt, và không công bố “đã tái lập LSTM90%”.

Giới hạn: một split cố định;ba seeds không thay cross-validation;duplicate/conflict policy thay population;class-weighting có thể đổi tradeoff;dropout kế thừa nguồn khá mạnh;max20epoch/ES là budget đã khóa;tiếng Anh/tweet airline có domain hạn chế;softmax chưacalibration;CPUtime không so thẳng vớiGPU. B1T40/T60 không khảo sát thêm ngữ cảnh khi không tweet nào dài vượt40tokens.

Future work: manualreview sâu;preprocessing/learningrate/gradient clipping ablation và logging để kiểm tra giả thuyếtgradient;benchmark airline/time riêng;uncertainty bằngbootstrap nhómduplicate. LSTM/GRU/Transformer chỉ là comparator mở rộng riêng nếu nhóm duyệt.

## Tài liệu và artifacts

- [Notebook nguồn]({source['source_url']}) và [dataset]({profile['source']}).
- `results/tables/`: bảng đầu vào báo cáo; `results/metrics/`: metrics/classification reports; `results/figures/manifest.json`: caption scope/purpose; `models/final/`: inference bundle.
- README hướng dẫn rerun và kiểm tra integrity. Không dùng test để tiếp tục tuning sau chương này.
"""
    (root / "reports/experiment_chapter.md").write_text(report, encoding="utf-8")
    (root / "reports/presentation_outline.md").write_text(
        "# Outline thuyết trình\n\n" + "\n".join(f"{i}. {title}" for i, title in enumerate([
            "Bài toán và mục tiêu RNN", "Audit nguồn: LSTM/binary90.32%", "Dataset/imbalance/duplicates (F01–F04)",
            "Train-only pipeline và split (F08)", "Kiến trúc/model summary (F09–F10)", "Seeds/checkpoint/validation/test policy",
            "A1/A2 và giới hạn reproduction", "Ablations trên validation (F20)",
            f"Mô hình cuối {selection['winner_config']}: test/CM/per-class (F14–F16)",
            "Error cases/confidence/length (F17–F19,F22)", "Demo dùng cùng checkpoint", "Kết luận/hạn chế/future work"], 1)), encoding="utf-8")
    return report
