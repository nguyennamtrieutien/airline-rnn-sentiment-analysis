"""Audit frozen artifacts, recompute held-out metrics and verify inference parity."""
from pathlib import Path
import json
import sys

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import nbformat
import numpy as np
import pandas as pd
from airline_rnn.common import read_json, save_json, utc_now
from airline_rnn.data import load_benchmark
from airline_rnn.evaluation import verified_runs
from airline_rnn.inference import SentimentPredictor
from airline_rnn.model import RNN
from airline_rnn.text import WordTokenizer
from airline_rnn.training import classification_metrics, tf

frozen, runs = verified_runs(root)
selection = read_json(root / "results/metrics/final_selection.json")
assert len(runs) == sum(len(item["seeds"]) for item in frozen["matrix"])
assert read_json(root / "results/metrics/test_release.json")["selection_signature"] == selection["selection_signature"]
for benchmark in ("binary_clean", "three_class"):
    data = load_benchmark(root, benchmark)
    assert data.row_id.is_unique
    assert data.groupby("group_id").split.nunique().max() == 1
    assert data.groupby("tweet_id").split.nunique().max() == 1
    assert data.groupby("clean_text").split.nunique().max() == 1

source = load_benchmark(root, "binary_source").sort_values("row_id")
ours = WordTokenizer.from_dict(read_json(root / "data/processed/A1/tokenizer.json"))
legacy = tf.keras.preprocessing.text.Tokenizer(num_words=4000)
legacy.fit_on_texts(source.clean_text)
assert ours.word_index == legacy.word_index
assert [ours.encode(text)[0] for text in source.clean_text] == legacy.texts_to_sequences(source.clean_text)

recomputed = []
for path in sorted((root / "results/metrics").glob("*_seed-*_test.json")):
    measured = read_json(path)
    assert measured["evaluated_at_utc"] >= selection["frozen_at_utc"]
    predictions = pd.read_csv(root / f"results/predictions/{measured['config_id']}_seed-{measured['seed']}_test.csv", keep_default_na=False)
    assert measured["config_id"] in selection["test_configs"]
    names = measured["label_names"]
    benchmark = load_benchmark(root, measured["benchmark"])
    assert predictions.row_id.tolist() == benchmark.loc[benchmark.split == "test", "row_id"].tolist()
    probabilities = predictions[[f"prob_{name}" for name in names]].to_numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-6)
    labels = predictions.true_label.map({name: i for i, name in enumerate(names)}).to_numpy(dtype="int32")
    check = classification_metrics(labels, probabilities, names)
    for key in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall"):
        np.testing.assert_allclose(check[key], measured[key], atol=1e-12)
    assert check["confusion_matrix"] == measured["confusion_matrix"]
    recomputed.append(path.name)

predictor = SentimentPredictor(root)
assert any(isinstance(layer, tf.keras.layers.RNN) for layer in predictor.model.layers)
assert not any(isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.GRU)) for layer in predictor.model.layers)
predictions = pd.read_csv(root / f"results/predictions/{selection['winner_config']}_seed-{selection['demo_seed']}_test.csv", keep_default_na=False)
for row in predictions.sample(12, random_state=42).itertuples():
    result = predictor.predict(row.raw_text)
    assert result["sentiment"] == row.predicted_label
    assert result["clean_text"] == row.clean_text
    np.testing.assert_allclose(list(result["probabilities"].values()),
        [getattr(row, f"prob_{name}") for name in predictor.names], rtol=2e-5, atol=2e-6)
try:
    predictor.predict("   ")
except ValueError:
    pass
else:
    raise AssertionError("Empty input must be rejected")

notebook = nbformat.read(root / "notebooks/Airline_RNN_Complete.ipynb", as_version=4)
nbformat.validate(notebook)
code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert [cell.execution_count for cell in code_cells] == list(range(1, len(code_cells) + 1))
assert not [output for cell in code_cells for output in cell.outputs if output.output_type == "error"]
figures = read_json(root / "results/figures/manifest.json")
for item in figures.values():
    assert (root / "results/figures" / item["png"]).exists()
    if item.get("pdf"):
        assert (root / "results/figures" / item["pdf"]).exists()
receipt = {"verified_at_utc": utc_now(), "status": "passed", "scientific_runs": len(runs),
           "clean_benchmarks_have_no_duplicate_group_overlap": True,
           "test_predictions_timestamped_after_selection": True,
           "source_tokenizer_parity_rows": len(source), "test_reports_recomputed": recomputed,
           "inference_matches_saved_test_predictions": 12, "only_rnn": True,
           "executed_notebook_code_cells": len(code_cells), "figures": len(figures),
           "human_sarcasm_annotation": "pending; never claimed completed"}
save_json(root / "results/metrics/artifact_verification.json", receipt)
print(json.dumps(receipt, ensure_ascii=False, indent=2))
