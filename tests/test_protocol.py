from airline_rnn.model import RNN
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from airline_rnn.config import Configuration, configurations
from airline_rnn.data import quality_filter, stratified_groups
from airline_rnn.text import WordTokenizer, clean_text, pad_sequences, source_clean


def test_negation_markers_and_reserved_ids():
    cleaned = clean_text("@United I can’t recommend this! https://example.com #LateFlight 😢")
    assert cleaned == "usermarker i can not recommend this urlmarker lateflight"
    tokenizer = WordTokenizer(8).fit([cleaned])
    assert tokenizer.word_index["__pad__"] == 0
    assert tokenizer.word_index["__oov__"] == 1
    assert tokenizer.word_index["usermarker"] == 2
    assert tokenizer.word_index["urlmarker"] == 3
    assert len(tokenizer.word_index) <= 8
    assert tokenizer.encode("unseenword")[0] == [1]


def test_legacy_tokenizer_matches_keras_including_ties_and_cap():
    import tensorflow as tf
    corpus = [source_clean(t) for t in ["@United Great flight! 10/10", "@Delta bad flight", "No no no.", "Thanks flight"]]
    custom = WordTokenizer(5, legacy=True).fit(corpus)
    reference = tf.keras.preprocessing.text.Tokenizer(num_words=5, split=" ")
    reference.fit_on_texts(corpus)
    assert custom.word_index == reference.word_index
    texts = corpus + ["unseen great no flight"]
    assert [custom.encode(t)[0] for t in texts] == reference.texts_to_sequences(texts)
    sequences = [custom.encode(t)[0] for t in texts]
    np.testing.assert_array_equal(pad_sequences(sequences, 3, legacy=True),
                                  tf.keras.utils.pad_sequences(sequences, maxlen=3))


def test_train_only_vocabulary_does_not_learn_validation_word():
    tokenizer = WordTokenizer(100).fit(["trainword shared"])
    assert "validationsecret" not in tokenizer.word_index
    assert tokenizer.encode("validationsecret shared")[0][0] == 1


def test_quality_filter_connects_duplicate_ids_and_text_and_quarantines_conflicts():
    frame = pd.DataFrame({"row_id": range(5), "tweet_id": ["a", "a", "b", "c", "d"],
                          "text": ["Great", "Great!", "great", "Bad", "bad"],
                          "airline_sentiment": ["positive", "positive", "positive", "negative", "neutral"]})
    kept, excluded, conflicts = quality_filter(frame, clean_text)
    assert len(kept) == 3
    assert kept.group_id.nunique() == 1
    assert set(conflicts.row_id) == {3, 4}
    assert set(excluded.reason) == {"conflicting_labels_in_duplicate_group"}


def test_duplicate_groups_never_cross_splits():
    rows = [{"row_id": i * 2 + j, "group_id": i, "airline_sentiment": ["negative", "neutral", "positive"][i % 3]}
            for i in range(90) for j in range(2)]
    split = stratified_groups(pd.DataFrame(rows), (0.7, 0.15, 0.15), 42)
    assert split.groupby("group_id").split.nunique().max() == 1
    assert set(split.split) == {"train", "validation", "test"}
    assert split.groupby("split").airline_sentiment.nunique().min() == 3


def test_matrix_changes_one_factor_and_has_22_runs():
    matrix = configurations("recommended")
    assert sum(len(seeds) for _, seeds in matrix) == 22
    base = next(config for config, _ in matrix if config.id == "C0").to_dict()
    for config, _ in matrix:
        if config.id.startswith("B"):
            differences = [key for key, value in config.to_dict().items() if key != "id" and base[key] != value]
            assert len(differences) == 1


def test_macro_f1_reveals_majority_only_prediction():
    from airline_rnn.training import classification_metrics
    true = np.array([0, 0, 0, 1, 2], dtype=np.int32)
    probs = np.array([[1., 0., 0.]] * 5)
    metrics = classification_metrics(true, probs, ["negative", "neutral", "positive"])
    assert metrics["accuracy"] == pytest.approx(0.6)
    assert metrics["macro_f1"] == pytest.approx(0.25)
    assert metrics["classification_report"]["neutral"]["recall"] == 0


def test_padding_mask_and_model_save_reload(tmp_path):
    from airline_rnn.model import build_model, configure_runtime, tf
    configure_runtime(42)
    config = replace(Configuration("test"), vocabulary_size=20, hidden_units=8,
                     embedding_dimension=4, dense_units=6)
    model = build_model(config, 7)
    post = np.array([[2, 5, 6, 0, 0, 0, 0]], dtype=np.int32)
    pre = np.array([[0, 0, 0, 0, 2, 5, 6]], dtype=np.int32)
    np.testing.assert_allclose(model(post, training=False).numpy(), model(pre, training=False).numpy(), atol=1e-6)
    assert sum(isinstance(layer, RNN) for layer in model.layers) == 1
    assert not any(isinstance(layer, (tf.keras.layers.LSTM, tf.keras.layers.GRU)) for layer in model.layers)
    path = tmp_path / "model.keras"
    model.save(path)
    loaded = tf.keras.models.load_model(path)
    np.testing.assert_allclose(model(post, training=False).numpy(), loaded(post, training=False).numpy(), atol=1e-7)
