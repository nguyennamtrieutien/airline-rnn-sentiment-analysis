"""Stream observable inference stages from the frozen model without fitting it."""
from __future__ import annotations

import time
import uuid

import numpy as np

from .text import clean_text, source_clean, pad_sequences
from .training import tf


class AnalysisRunner:
    def __init__(self, predictor):
        self.predictor = predictor
        # Expose actual activations from one forward pass of the loaded model.
        recurrent_layer = next(layer for layer in predictor.model.layers if isinstance(layer, tf.keras.layers.RNN))
        self.traced_model = tf.keras.Model(
            inputs=predictor.model.inputs,
            outputs=[predictor.model.get_layer("word_embedding").output, recurrent_layer.output,
                     predictor.model.get_layer("dense_features").output,
                     predictor.model.get_layer("sentiment_probabilities").output],
        )
        length = predictor.representation["sequence_length"]

        @tf.function(input_signature=[tf.TensorSpec([None, length], tf.int32)])
        def forward(features):
            return self.traced_model([features], training=False)

        self.forward = forward

    @staticmethod
    def validate(text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Hãy nhập một câu hoặc tweet tiếng Anh.")
        if len(text) > 4000:
            raise ValueError("Câu nhập quá dài; tối đa 4.000 ký tự.")

    def events(self, text):
        self.validate(text)
        p = self.predictor
        began = time.perf_counter()
        request_id = uuid.uuid4().hex
        yield {"type": "start", "request_id": request_id, "steps": 8}

        def start(step):
            return {"type": "step_start", "step": step, "request_id": request_id}

        def done(step, data, started):
            return {"type": "step_done", "step": step, "request_id": request_id,
                    "elapsed_ms": (time.perf_counter() - started) * 1000, "data": data}

        yield start(1)
        started = time.perf_counter()
        yield done(1, {"original_text": text, "characters": len(text), "max_characters": 4000}, started)

        yield start(2)
        started = time.perf_counter()
        cleaned = (source_clean if p.representation["legacy"] else clean_text)(text)
        yield done(2, {"clean_text": cleaned, "changed": cleaned != text,
                      "empty_marker": cleaned == "emptymarker",
                      "preprocessing": "legacy" if p.representation["legacy"] else "P0"}, started)

        yield start(3)
        started = time.perf_counter()
        words = p.tokenizer.words(cleaned)
        yield done(3, {"words": words}, started)

        yield start(4)
        started = time.perf_counter()
        sequence, stats = p.tokenizer.encode(cleaned)
        tokens = []
        for word in words:
            index = p.tokenizer.word_index.get(word)
            unknown = index is None or index >= p.tokenizer.vocabulary_size
            tokens.append({"word": word, "id": (None if p.tokenizer.legacy else 1) if unknown else index,
                           "oov": unknown})
        yield done(4, {"tokens": tokens, "ids_before_padding": sequence, **stats,
                      "vocabulary_size": p.tokenizer.vocabulary_size,
                      "vocabulary_fit_scope": p.representation["fit_scope"]}, started)

        yield start(5)
        started = time.perf_counter()
        length = p.representation["sequence_length"]
        features = pad_sequences([sequence], length, p.representation["legacy"])
        mask = features[0] != 0
        retained = min(len(sequence), length)
        yield done(5, {"ids": features[0].tolist(), "mask": mask.tolist(),
                      "shape": list(features.shape), "sequence_length": length,
                      "retained_tokens": retained, "padding_tokens": length - retained,
                      "removed_tokens": max(0, len(sequence) - length),
                      "mask_zero": p.config["mask_zero"], "padding": p.representation["padding"],
                      "truncating": p.representation["truncating"]}, started)

        yield start(6)
        started = time.perf_counter()
        with p.lock:
            embedding, hidden, dense, probabilities = [x.numpy() for x in self.forward(features)]
        first_real = next((i for i, real in enumerate(mask) if real), None)
        yield done(6, {"embedding_shape": list(embedding.shape), "hidden_shape": list(hidden.shape),
                      "dense_shape": list(dense.shape), "output_shape": list(probabilities.shape),
                      "first_token_embedding_preview": embedding[0, first_real, :8].tolist() if first_real is not None else [],
                      "last_hidden_state_preview": hidden[0, :8].tolist(),
                      "hidden_l2_norm": float(np.linalg.norm(hidden[0])),
                      "parameters": p.model.count_params(), "hidden_units": p.config["hidden_units"],
                      "embedding_dimension": p.config["embedding_dimension"],
                      "activation": "tanh", "dense_activation": "relu", "dropout_active": False,
                      "config_id": p.manifest["config_id"], "seed": p.manifest["seed"],
                      "note": "Các vector là giá trị thật trong forward pass; không phải giải thích mức đóng góp của từng từ."}, started)

        yield start(7)
        started = time.perf_counter()
        probabilities = probabilities[0]
        scores = {name: float(probabilities[i]) for i, name in enumerate(p.names)}
        yield done(7, {"probabilities": scores, "probability_sum": float(sum(scores.values()))}, started)

        yield start(8)
        started = time.perf_counter()
        index = int(np.argmax(probabilities))
        result = {"sentiment": p.names[index],
                  "probabilities": scores, "original_text": text, "sequence_length": length,
                  "confidence": float(probabilities[index]), "clean_text": cleaned,
                  "token_length": len(sequence), "unknown_tokens": stats["unknown_tokens"],
                  "truncated": len(sequence) > length, "config_id": p.manifest["config_id"],
                  "seed": p.manifest["seed"],
                  "note": "Softmax chưa được calibration; không diễn giải score như độ tin cậy đã kiểm chứng."}
        yield done(8, {"sentiment": result["sentiment"], "confidence": result["confidence"],
                      "rule": "argmax"}, started)
        yield {"type": "result", "request_id": request_id,
               "total_elapsed_ms": (time.perf_counter() - began) * 1000, "result": result}
