from __future__ import annotations

import shutil

import numpy as np
import pandas as pd

from .common import digest, initialize, is_registered_training_source, object_digest, read_json, save_json, training_source_digest, utc_now
from .config import Configuration
from .data import encode_frame, load_benchmark
from .text import WordTokenizer
from .training import classification_metrics, make_predictor, tf


def verified_runs(root):
    root = initialize(root)
    frozen = read_json(root / "configs/frozen_matrix.json")
    runs = []
    for entry in frozen["matrix"]:
        for seed in entry["seeds"]:
            directory = root / "models/runs" / entry["config"]["id"] / f"seed-{seed}"
            if not (directory / "run.json").exists():
                raise RuntimeError(f"Matrix incomplete: missing {directory}")
            run = read_json(directory / "run.json")
            if run["status"] != "completed" or run["smoke"]:
                raise RuntimeError(f"Run is not a completed scientific run: {directory}")
            if run["config"] != entry["config"] or not is_registered_training_source(root, run["training_source_sha256"]):
                raise RuntimeError(f"Frozen config/code differs from completed run: {directory}")
            for filename, expected in run["artifact_hashes"].items():
                if digest(directory / filename) != expected:
                    raise RuntimeError(f"Modified run artifact: {directory / filename}")
            runs.append(run)
    return frozen, runs


def freeze_selection(root):
    root = initialize(root)
    frozen, runs = verified_runs(root)
    table = pd.DataFrame([{"config_id": run["config"]["id"], "benchmark": run["config"]["benchmark"],
                           "seed": run["seed"], "val_accuracy": run["validation_accuracy"],
                           "val_macro_f1": run["validation_macro_f1"], "val_weighted_f1": run["validation_weighted_f1"],
                           "parameters": run["parameters"], "training_seconds": run["training_seconds"],
                           "epochs_trained": run["epochs_trained"], "best_epoch": run["best_epoch"]} for run in runs])
    table.to_csv(root / "results/tables/validation_runs.csv", index=False)
    summary = table.groupby(["config_id", "benchmark"], sort=False).agg(
        seeds=("seed", "count"), val_accuracy_mean=("val_accuracy", "mean"),
        val_accuracy_std=("val_accuracy", "std"), val_macro_f1_mean=("val_macro_f1", "mean"),
        val_macro_f1_std=("val_macro_f1", "std"), val_weighted_f1_mean=("val_weighted_f1", "mean"),
        parameters=("parameters", "first"), training_seconds_median=("training_seconds", "median"),
        epochs_mean=("epochs_trained", "mean")).reset_index()
    summary["val_accuracy_std"] = summary["val_accuracy_std"].fillna(0)
    summary["val_macro_f1_std"] = summary["val_macro_f1_std"].fillna(0)
    summary.to_csv(root / "results/tables/validation_summary.csv", index=False)
    if frozen["tier"] == "minimum":
        winner = "A2"
    else:
        candidates = summary[summary.benchmark == "three_class"].copy()
        maximum = candidates.val_macro_f1_mean.max()
        candidates = candidates[(maximum - candidates.val_macro_f1_mean) < frozen["validation_tie_tolerance"]]
        order = {entry["config"]["id"]: i for i, entry in enumerate(frozen["matrix"])}
        candidates["order"] = candidates.config_id.map(order)
        winner = candidates.sort_values(["parameters", "training_seconds_median", "order"]).iloc[0].config_id
    winner_runs = sorted([run for run in runs if run["config"]["id"] == winner],
                         key=lambda run: (run["validation_macro_f1"], run["seed"]))
    demo_run = winner_runs[len(winner_runs) // 2]
    selection = {"winner_config": winner, "demo_seed": demo_run["seed"],
                 "criterion": "mean_validation_macro_f1; tolerance<0.005; fewer_parameters; median_training_time; frozen_order",
                 "demo_seed_policy": "median_validation_macro_f1_before_test",
                 "tier": frozen["tier"], "matrix_sha256": digest(root / "configs/frozen_matrix.json"),
                 "run_fingerprints": {f"{run['config']['id']}/seed-{run['seed']}": run["fingerprint"] for run in runs},
                 "test_configs": list(dict.fromkeys(["A1", "A2"] + ([] if frozen["tier"] == "minimum" else ["C0"]) + [winner]))}
    signature = object_digest(selection)
    path = root / "results/metrics/final_selection.json"
    if path.exists():
        previous = read_json(path)
        if previous["selection_signature"] != signature:
            raise RuntimeError("A different selection is already frozen; never change selection after reading test.")
        return previous
    selection.update({"selection_signature": signature, "frozen_at_utc": utc_now(), "test_status": "locked_until_evaluate_final"})
    save_json(path, selection)
    return selection


def evaluate_final(root):
    root = initialize(root)
    selection_path = root / "results/metrics/final_selection.json"
    if not selection_path.exists():
        raise RuntimeError("Test is locked. Complete the registered matrix and freeze selection first.")
    selection = read_json(selection_path)
    frozen, runs = verified_runs(root)
    evaluation_runs = [run for run in runs if run["config"]["id"] in selection["test_configs"]]
    final_rows = []
    evaluation_times = []
    for run in evaluation_runs:
        config = Configuration(**run["config"])
        directory = root / "models/runs" / config.id / f"seed-{run['seed']}"
        representation = read_json(directory / "representation.json")
        if digest(root / f"data/splits/{config.benchmark}.csv") != representation["split_sha256"]:
            raise RuntimeError("Split changed after training")
        test = load_benchmark(root, config.benchmark)
        test = test[test.split == "test"].copy()
        fingerprint = object_digest({"selection": selection["selection_signature"], "run": run["fingerprint"],
                                     "checkpoint": run["artifact_hashes"][run["selected_checkpoint"]],
                                     "test_rows": [int(i) for i in test.row_id]})
        metric_path = root / f"results/metrics/{config.id}_seed-{run['seed']}_test.json"
        predictions_path = root / f"results/predictions/{config.id}_seed-{run['seed']}_test.csv"
        if metric_path.exists():
            metrics = read_json(metric_path)
            if metrics["evaluation_fingerprint"] != fingerprint or digest(predictions_path) != metrics["predictions_sha256"]:
                raise RuntimeError("Final test artifacts or evaluation provenance changed")
        else:
            tokenizer = WordTokenizer.from_dict(read_json(directory / "tokenizer.json"))
            features, statistics = encode_frame(test, tokenizer, representation["sequence_length"], representation["legacy"])
            labels = test.airline_sentiment.map(config.labels).to_numpy(dtype="int32")
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(directory / run["selected_checkpoint"], compile=False)
            probabilities = make_predictor(model)(features)
            metrics = classification_metrics(labels, probabilities, config.label_names)
            output = test[["row_id", "tweet_id", "airline", "airline_sentiment", "text", "clean_text"]].copy()
            output = output.rename(columns={"airline_sentiment": "true_label", "text": "raw_text"})
            output["predicted_label"] = [config.label_names[i] for i in probabilities.argmax(axis=1)]
            output["correct"] = output.true_label == output.predicted_label
            output["confidence"] = probabilities.max(axis=1)
            sorted_probs = np.sort(probabilities, axis=1)
            output["top2_margin"] = sorted_probs[:, -1] - sorted_probs[:, -2]
            output["config_id"], output["seed"] = config.id, run["seed"]
            for i, name in enumerate(config.label_names):
                output[f"prob_{name}"] = probabilities[:, i]
            output = output.merge(statistics, on="row_id", validate="one_to_one", sort=False)
            output.to_csv(predictions_path, index=False)
            metrics.update({"evaluation_fingerprint": fingerprint, "predictions_sha256": digest(predictions_path),
                            "config_id": config.id, "seed": run["seed"], "benchmark": config.benchmark,
                            "selection_signature": selection["selection_signature"], "evaluated_at_utc": utc_now()})
            save_json(metric_path, metrics)
            (root / f"results/metrics/{config.id}_seed-{run['seed']}_classification_report.txt").write_text(
                metrics["classification_report_text"], encoding="utf-8")
            print(f"Final test: {config.id} seed={run['seed']} accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}", flush=True)
        final_rows.append({"config_id": config.id, "seed": run["seed"], "benchmark": config.benchmark,
                           "test_accuracy": metrics["accuracy"], "test_macro_f1": metrics["macro_f1"],
                           "test_weighted_f1": metrics["weighted_f1"], "support": metrics["support"]})
        evaluation_times.append(metrics["evaluated_at_utc"])
    for benchmark in dict.fromkeys(run["config"]["benchmark"] for run in evaluation_runs):
        config = next(Configuration(**run["config"]) for run in evaluation_runs if run["config"]["benchmark"] == benchmark)
        frame = load_benchmark(root, benchmark)
        train, test = frame[frame.split == "train"], frame[frame.split == "test"]
        priors = train.airline_sentiment.value_counts().reindex(config.label_names, fill_value=0).to_numpy(dtype=float) / len(train)
        probabilities = np.tile(priors, (len(test), 1))
        metric = classification_metrics(test.airline_sentiment.map(config.labels).to_numpy(dtype="int32"), probabilities, config.label_names)
        metric["benchmark"] = benchmark
        save_json(root / f"results/metrics/majority_{benchmark}.json", metric)
        final_rows.append({"config_id": "majority", "seed": -1, "benchmark": benchmark,
                           "test_accuracy": metric["accuracy"], "test_macro_f1": metric["macro_f1"],
                           "test_weighted_f1": metric["weighted_f1"], "support": metric["support"]})
    final_table = pd.DataFrame(final_rows)
    final_table.to_csv(root / "results/tables/test_runs.csv", index=False)
    summary = final_table.groupby(["config_id", "benchmark"], sort=False).agg(
        test_accuracy_mean=("test_accuracy", "mean"), test_accuracy_std=("test_accuracy", "std"),
        test_macro_f1_mean=("test_macro_f1", "mean"), test_macro_f1_std=("test_macro_f1", "std"),
        test_weighted_f1_mean=("test_weighted_f1", "mean"), support=("support", "first"), seeds=("seed", "count")).reset_index().fillna(0)
    summary.to_csv(root / "results/tables/test_summary.csv", index=False)
    demo_directory = root / "models/runs" / selection["winner_config"] / f"seed-{selection['demo_seed']}"
    demo_run = read_json(demo_directory / "run.json")
    for source, destination in ((demo_run["selected_checkpoint"], "model.keras"), ("tokenizer.json", "tokenizer.json"),
                                ("label_map.json", "label_map.json"), ("representation.json", "representation.json"),
                                ("model_summary.txt", "model_summary.txt")):
        shutil.copy2(demo_directory / source, root / "models/final" / destination)
    save_json(root / "models/final/config.json", demo_run["config"])
    save_json(root / "models/final/manifest.json", {"config_id": selection["winner_config"], "seed": selection["demo_seed"],
              "selection_signature": selection["selection_signature"], "trained_model_sha256": digest(root / "models/final/model.keras"),
              "tokenizer_sha256": digest(root / "models/final/tokenizer.json"), "softmax_is_calibrated": False,
              "asset_hashes": {name: digest(root / "models/final" / name) for name in
                               ("model.keras", "tokenizer.json", "label_map.json", "config.json", "representation.json")}})
    release_path = root / "results/metrics/test_release.json"
    release = read_json(release_path) if release_path.exists() else {"released_at_utc": utc_now()}
    release.update({
              "selection_signature": selection["selection_signature"], "evaluated_configs": selection["test_configs"],
              "first_test_prediction_at_utc": min(evaluation_times),
              "registered_test_predictions_completed_at_utc": max(evaluation_times),
              "policy": "No further tuning on this holdout; final errors are descriptive analysis only."})
    save_json(release_path, release)
    return summary
