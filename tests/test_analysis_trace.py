from pathlib import Path

import numpy as np
import pytest

from airline_rnn.analysis import AnalysisRunner
from airline_rnn.inference import SentimentPredictor


@pytest.fixture(scope="module")
def analysis():
    return AnalysisRunner(SentimentPredictor(Path(__file__).resolve().parents[1]))


@pytest.mark.parametrize("text", [
    "Thank you for the helpful service!",
    "@united I can't recommend this https://example.com 😢",
    "qzxvnonexistentword",
    "flight " * 48,
    "😢",
])
def test_trace_preserves_frozen_inference_and_representation(analysis, text):
    events = list(analysis.events(text))
    assert [e["step"] for e in events if e["type"] == "step_start"] == list(range(1, 9))
    completed = {e["step"]: e for e in events if e["type"] == "step_done"}
    assert len(completed) == 8
    assert len({e["request_id"] for e in events}) == 1
    reference = analysis.predictor.predict(text)
    result = events[-1]["result"]
    assert result["sentiment"] == reference["sentiment"]
    assert result["clean_text"] == reference["clean_text"]
    np.testing.assert_allclose(list(result["probabilities"].values()),
                               list(reference["probabilities"].values()), atol=1e-7, rtol=1e-6)
    padded = completed[5]["data"]
    ids = padded["ids"]
    assert len(ids) == 40
    assert padded["mask"] == [i != 0 for i in ids]
    assert padded["padding_tokens"] + padded["retained_tokens"] == 40
    encoded = completed[4]["data"]["ids_before_padding"]
    assert ids[:min(40, len(encoded))] == encoded[:40]
    assert padded["removed_tokens"] == max(0, len(encoded) - 40)
    actual = completed[6]["data"]
    assert actual["embedding_shape"] == [1, 40, 128]
    assert actual["hidden_shape"] == [1, 196]
    assert actual["dropout_active"] is False
    assert completed[3]["data"]["words"] == [token["word"] for token in completed[4]["data"]["tokens"]]
    assert completed[7]["data"]["probabilities"] == result["probabilities"]
    assert completed[8]["data"]["sentiment"] == result["sentiment"]
    assert result["original_text"] == text
    assert result["sequence_length"] == len(ids)
    first = next(i for i, value in enumerate(ids) if value != 0)
    weights = analysis.predictor.model.get_layer("word_embedding").get_weights()[0]
    np.testing.assert_allclose(actual["first_token_embedding_preview"], weights[ids[first], :8], atol=1e-7)
    assert all(e["elapsed_ms"] >= 0 for e in completed.values())
    assert events[-1]["total_elapsed_ms"] >= sum(e["elapsed_ms"] for e in completed.values())


@pytest.mark.parametrize("text", ["", "   ", None, 123, "a" * 4001])
def test_invalid_input_produces_no_successful_trace(analysis, text):
    with pytest.raises(ValueError):
        list(analysis.events(text))
