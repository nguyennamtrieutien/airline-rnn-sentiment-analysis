from __future__ import annotations

import threading

import numpy as np

from .common import digest, project_root, read_json
from .text import WordTokenizer, clean_text, pad_sequences, source_clean
from .training import make_predictor, tf


class SentimentPredictor:
    """Load the frozen, validation-selected bundle once; never fit at inference."""

    def __init__(self, root=None):
        self.root = project_root(root)
        folder = self.root / "models/final"
        self.manifest = read_json(folder / "manifest.json")
        for filename, key in (("model.keras", "trained_model_sha256"), ("tokenizer.json", "tokenizer_sha256")):
            if digest(folder / filename) != self.manifest[key]:
                raise ValueError(f"Inference bundle checksum mismatch: {filename}")
        for filename, expected in self.manifest.get("asset_hashes", {}).items():
            if digest(folder / filename) != expected:
                raise ValueError(f"Inference bundle checksum mismatch: {filename}")
        self.config = read_json(folder / "config.json")
        self.representation = read_json(folder / "representation.json")
        labels = read_json(folder / "label_map.json")
        self.names = [name for name, _ in sorted(labels.items(), key=lambda item: item[1])]
        self.tokenizer = WordTokenizer.from_dict(read_json(folder / "tokenizer.json"))
        self.model = tf.keras.models.load_model(folder / "model.keras", compile=False)
        if self.model.output_shape[-1] != len(self.names):
            raise ValueError("Model output and label map do not match")
        self.forward = make_predictor(self.model)
        self.lock = threading.Lock()

    def predict(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Hãy nhập một câu hoặc tweet tiếng Anh.")
        if len(text) > 4000:
            raise ValueError("Câu nhập quá dài; tối đa 4.000 ký tự.")
        cleaned = (source_clean if self.representation["legacy"] else clean_text)(text)
        sequence, stats = self.tokenizer.encode(cleaned)
        features = pad_sequences([sequence], self.representation["sequence_length"], self.representation["legacy"])
        with self.lock:
            probabilities = self.forward(features)[0]
        index = int(np.argmax(probabilities))
        return {"sentiment": self.names[index],
                "probabilities": {name: float(probabilities[i]) for i, name in enumerate(self.names)},
                "confidence": float(probabilities[index]), "clean_text": cleaned,
                "token_length": len(sequence), "unknown_tokens": stats["unknown_tokens"],
                "truncated": len(sequence) > self.representation["sequence_length"],
                "config_id": self.manifest["config_id"], "seed": self.manifest["seed"],
                "note": "Softmax chưa được calibration; không diễn giải score như độ tin cậy đã kiểm chứng."}
