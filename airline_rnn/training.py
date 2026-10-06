from __future__ import annotations

import importlib.metadata
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from .common import digest, initialize, is_registered_training_source, object_digest, read_json, save_json, training_source_digest, utc_now
from .config import configurations
from .data import prepare_data, prepare_representation
from .model import RNN, build_model, configure_runtime, tf, training_dataset


def environment_snapshot():
    versions = {}
    for package in ("tensorflow", "keras", "numpy", "pandas", "scikit-learn", "matplotlib"):
        versions[package] = importlib.metadata.version(package)
    return {"recorded_at_utc": utc_now(), "python": platform.python_version(), "os": platform.platform(),
            "processor": platform.processor(), "machine": platform.machine(), "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_logical_cores": psutil.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
            "gpu_devices": [{"name": device.name, "details": tf.config.experimental.get_device_details(device)}
                            for device in tf.config.list_physical_devices("GPU")],
            "tensorflow_build": tf.sysconfig.get_build_info(), "versions": versions,
            "precision": "float32", "deterministic_ops": True, "intra_threads": tf.config.threading.get_intra_op_parallelism_threads(),
            "inter_threads": tf.config.threading.get_inter_op_parallelism_threads()}


def classification_metrics(labels, probabilities, names):
    predicted = np.argmax(probabilities, axis=1)
    label_ids = list(range(len(names)))
    report = classification_report(labels, predicted, labels=label_ids, target_names=names,
                                   output_dict=True, zero_division=0)
    report_text = classification_report(labels, predicted, labels=label_ids, target_names=names,
                                        digits=4, zero_division=0)
    selected = np.clip(probabilities[np.arange(len(labels)), labels], 1e-7, 1.0)
    return {"accuracy": float(accuracy_score(labels, predicted)),
            "macro_f1": float(f1_score(labels, predicted, labels=label_ids, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, predicted, labels=label_ids, average="weighted", zero_division=0)),
            "macro_precision": float(report["macro avg"]["precision"]),
            "macro_recall": float(report["macro avg"]["recall"]),
            "loss": float(-np.log(selected).mean()), "support": len(labels), "label_names": names,
            "confusion_matrix": confusion_matrix(labels, predicted, labels=label_ids).tolist(),
            "classification_report": report, "classification_report_text": report_text}


def make_predictor(model):
    @tf.function(reduce_retracing=True)
    def forward(features):
        return model(features, training=False)

    def predict(features, batch_size=256):
        return np.concatenate([forward(features[start:start + batch_size]).numpy()
                               for start in range(0, len(features), batch_size)], axis=0)
    return predict


class ExperimentCallback(tf.keras.callbacks.Callback):
    def __init__(self, config, seed, directory, arrays):
        super().__init__()
        self.config, self.seed, self.directory, self.arrays = config, seed, directory, arrays
        self.records = []
        self.best_score, self.best_loss, self.best_epoch = -1.0, float("inf"), 0
        self.stop_reference, self.wait = -1.0, 0

    def on_train_begin(self, logs=None):
        self.predict = make_predictor(self.model)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_started = time.perf_counter()

    def on_epoch_end(self, epoch, logs=None):
        fit_seconds = time.perf_counter() - self.epoch_started
        metric_started = time.perf_counter()
        val_probs = self.predict(self.arrays["x_validation"])
        train_probs = self.predict(self.arrays["x_train"])
        val = classification_metrics(self.arrays["y_validation"], val_probs, self.config.label_names)
        train = classification_metrics(self.arrays["y_train"], train_probs, self.config.label_names)
        record = {"epoch": epoch + 1, "train_loss": float(logs["loss"]), "train_accuracy": float(logs["accuracy"]),
                  "train_eval_loss": train["loss"], "train_eval_accuracy": train["accuracy"],
                  "train_eval_macro_f1": train["macro_f1"], "val_loss": val["loss"],
                  "val_accuracy": val["accuracy"], "val_macro_f1": val["macro_f1"],
                  "val_weighted_f1": val["weighted_f1"], "learning_rate": float(self.model.optimizer.learning_rate.numpy()),
                  "fit_seconds": fit_seconds, "metric_seconds": time.perf_counter() - metric_started}
        if val["macro_f1"] > self.best_score or (val["macro_f1"] == self.best_score and val["loss"] < self.best_loss):
            self.best_score, self.best_loss, self.best_epoch = val["macro_f1"], val["loss"], epoch + 1
            self.model.save(self.directory / "best.keras")
        if val["macro_f1"] > self.stop_reference + self.config.min_delta:
            self.stop_reference, self.wait = val["macro_f1"], 0
        else:
            self.wait += 1
        self.records.append(record)
        pd.DataFrame(self.records).to_csv(self.directory / "history.csv", index=False)
        save_json(self.directory / "progress.json", {"epoch": epoch + 1, "best_epoch": self.best_epoch,
                  "best_val_macro_f1": self.best_score, "updated_at_utc": utc_now()})
        print(f"{self.config.id} seed={self.seed} epoch={epoch + 1:02d} "
              f"val_acc={val['accuracy']:.4f} val_macro_f1={val['macro_f1']:.4f} "
              f"fit={fit_seconds:.1f}s metrics={record['metric_seconds']:.1f}s", flush=True)
        if self.config.early_stopping and self.wait >= self.config.patience:
            self.model.stop_training = True


def run_experiment(root, config, seed, *, smoke=False, max_epochs=None, restart_incomplete=False):
    root = initialize(root)
    metadata = prepare_representation(root, config)
    directory = root / ("models/smoke" if smoke else "models/runs") / config.id / f"seed-{seed}"
    fingerprint = object_digest({"config": config.to_dict(), "seed": seed, "representation": metadata,
                                 "training_code": training_source_digest(), "smoke": smoke, "max_epochs": max_epochs})
    if (directory / "run.json").exists():
        existing = read_json(directory / "run.json")
        # A naming migration may reuse a completed cache only with its exact original fingerprint.
        if existing.get("status") == "completed" and is_registered_training_source(root, existing["training_source_sha256"]):
            fingerprint = object_digest({"config": config.to_dict(), "seed": seed, "representation": metadata,
                                         "training_code": existing["training_source_sha256"],
                                         "smoke": smoke, "max_epochs": max_epochs})
        if existing["fingerprint"] != fingerprint:
            raise ValueError(f"Config/code/data changed for existing run {directory}; use a new project/output root.")
        if existing.get("status") == "completed":
            for filename, expected_hash in existing["artifact_hashes"].items():
                if digest(directory / filename) != expected_hash:
                    raise ValueError(f"Artifact changed: {directory / filename}")
            print(f"Verified cached run: {config.id} seed={seed}", flush=True)
            return existing
        if not restart_incomplete:
            raise RuntimeError(f"Incomplete run at {directory}. Use --restart-incomplete to archive and restart this seed.")
        archive = directory.with_name(directory.name + "-interrupted-" + str(time.time_ns()))
        directory.rename(archive)
    directory.mkdir(parents=True, exist_ok=True)
    tf.keras.backend.clear_session()
    configure_runtime(seed)
    snapshot = environment_snapshot()
    save_json(directory / "environment.json", snapshot)
    save_json(root / "environment/runtime.json", snapshot)
    arrays = dict(np.load(root / f"data/processed/{config.id}/train_validation.npz"))
    if smoke:
        arrays = {key: value[:256 if "train" in key and "validation" not in key else 96]
                  for key, value in arrays.items()}
    model = build_model(config, metadata["sequence_length"])
    model.summary(print_fn=lambda line: None)
    summary = []
    model.summary(print_fn=lambda line: summary.append(line))
    (directory / "model_summary.txt").write_text("\n".join(summary).replace(RNN.__name__, "RNN"), encoding="utf-8")
    layers = [{"layer": layer.name, "type": "RNN" if isinstance(layer, RNN) else type(layer).__name__,
               "output_shape": str(layer.output.shape), "parameters": layer.count_params()}
              for layer in model.layers]
    save_json(directory / "architecture.json", layers)
    shutil.copy2(root / f"data/processed/{config.id}/tokenizer.json", directory / "tokenizer.json")
    save_json(directory / "label_map.json", config.labels)
    save_json(directory / "representation.json", metadata)
    weights = None
    if config.class_weighting == "balanced":
        counts = np.bincount(arrays["y_train"], minlength=len(config.labels))
        weights = {i: float(len(arrays["y_train"]) / (len(config.labels) * counts[i])) for i in range(len(counts))}
    run = {"config": config.to_dict(), "seed": seed, "fingerprint": fingerprint,
           "training_source_sha256": training_source_digest(), "status": "running", "smoke": smoke,
           "started_at_utc": utc_now(), "class_weights": weights,
           "parameters": model.count_params(), "train_support": len(arrays["y_train"]),
           "validation_support": len(arrays["y_validation"]), "epochs_requested": max_epochs or config.max_epochs}
    save_json(directory / "run.json", run)
    callback = ExperimentCallback(config, seed, directory, arrays)
    started = time.perf_counter()
    try:
        model.fit(training_dataset(arrays["x_train"], arrays["y_train"], config.batch_size, seed),
                  epochs=max_epochs or config.max_epochs, callbacks=[callback], class_weight=weights, verbose=0)
        model.save(directory / "last.keras")
        training_seconds = time.perf_counter() - started
        selected_name = "last.keras" if config.id == "A1" else "best.keras"
        selected = tf.keras.models.load_model(directory / selected_name)
        probabilities = make_predictor(selected)(arrays["x_validation"])
        metrics = classification_metrics(arrays["y_validation"], probabilities, config.label_names)
        save_json(directory / "validation_metrics.json", metrics)
        np.savez_compressed(directory / "validation_predictions.npz", row_ids=arrays["ids_validation"],
                            true=arrays["y_validation"], probabilities=probabilities)
        (directory / "validation_classification_report.txt").write_text(metrics["classification_report_text"], encoding="utf-8")
        run.update({"status": "completed", "completed_at_utc": utc_now(), "training_seconds": training_seconds,
                    "epochs_trained": len(callback.records), "best_epoch": callback.best_epoch,
                    "selected_checkpoint": selected_name, "stop_reason": "early_stopping" if model.stop_training else "max_epochs",
                    "validation_accuracy": metrics["accuracy"], "validation_macro_f1": metrics["macro_f1"],
                    "validation_weighted_f1": metrics["weighted_f1"]})
        run["artifact_hashes"] = {name: digest(directory / name) for name in
                                  (selected_name, "tokenizer.json", "label_map.json", "representation.json",
                                   "history.csv", "validation_metrics.json", "validation_predictions.npz")}
        save_json(directory / "run.json", run)
        if not smoke:
            history_dir = root / "results/histories" / config.id
            history_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(directory / "history.csv", history_dir / f"seed-{seed}.csv")
        return run
    except BaseException as error:
        run.update({"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error": repr(error), "updated_at_utc": utc_now()})
        save_json(directory / "run.json", run)
        raise


def run_matrix(root, tier="recommended", restart_incomplete=False):
    root = initialize(root)
    matrix = [{"config": config.to_dict(), "seeds": list(seeds)} for config, seeds in configurations(tier)]
    matrix_path = root / "configs/frozen_matrix.json"
    if matrix_path.exists() and read_json(matrix_path)["matrix"] != matrix:
        raise ValueError("A different matrix is already frozen. Use a separate output root for another tier.")
    if not matrix_path.exists():
        save_json(matrix_path, {"tier": tier, "matrix": matrix, "training_seeds": [42, 2026, 3407],
                  "test_policy": "A1,A2,C0,winner only; after selection; no test during tuning",
                  "validation_tie_tolerance": 0.005, "frozen_at_utc": utc_now()})
    if (root / "results/metrics/final_selection.json").exists():
        print("Selection already frozen; only matching cached runs are permitted.", flush=True)
    runs = []
    for config, seeds in configurations(tier):
        for seed in seeds:
            runs.append(run_experiment(root, config, seed, restart_incomplete=restart_incomplete))
            pd.DataFrame([{k: r[k] for k in ("seed", "validation_accuracy", "validation_macro_f1",
                                            "validation_weighted_f1", "training_seconds", "parameters", "epochs_trained", "best_epoch")}
                          | {"config_id": r["config"]["id"], "benchmark": r["config"]["benchmark"]}
                          for r in runs]).to_csv(root / "results/tables/validation_runs.csv", index=False)
    return runs
