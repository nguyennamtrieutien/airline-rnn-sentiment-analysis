"""Export frozen inference weights, without fitting or changing the checkpoint."""
from __future__ import annotations

import html
import html.entities
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from airline_rnn.common import digest, save_json
from airline_rnn.inference import SentimentPredictor
from airline_rnn.text import CONTRACTIONS


def main():
    predictor = SentimentPredictor(ROOT)  # Verifies every frozen inference asset.
    config = predictor.config
    representation = predictor.representation
    if representation["legacy"] or not config["mask_zero"]:
        raise ValueError("Browser export supports the selected P0 masked RNN bundle only")
    output = ROOT / "deploy/browser-model"
    output.mkdir(parents=True, exist_ok=True)
    names = ["embedding", "input_kernel", "recurrent_kernel", "rnn_bias",
             "dense_kernel", "dense_bias", "output_kernel", "output_bias"]
    weights = predictor.model.get_weights()
    expected_shapes = [(config["vocabulary_size"], config["embedding_dimension"]),
                       (config["embedding_dimension"], config["hidden_units"]),
                       (config["hidden_units"], config["hidden_units"]), (config["hidden_units"],),
                       (config["hidden_units"], config["dense_units"]), (config["dense_units"],),
                       (config["dense_units"], len(predictor.names)), (len(predictor.names),)]
    if [tuple(weight.shape) for weight in weights] != expected_shapes:
        raise ValueError("Unexpected trained model architecture")
    tensors, offset = [], 0
    with (output / "weights.bin").open("wb") as stream:
        for name, weight in zip(names, weights, strict=True):
            array = np.asarray(weight, dtype="<f4")
            tensors.append({"name": name, "shape": list(array.shape), "offset": offset,
                            "length": int(array.size)})
            stream.write(array.tobytes(order="C"))
            offset += array.size
    shutil.copy2(ROOT / "models/final/tokenizer.json", output / "tokenizer.json")
    # Same HTML5 entity table and invalid numeric references used by Python P0.
    save_json(output / "preprocessing.json", {
        "contractions": CONTRACTIONS, "html_entities": html.entities.html5,
        "invalid_charrefs": html._invalid_charrefs,
        "invalid_codepoints": sorted(html._invalid_codepoints),
        "whitespace_codepoints": [code for code in range(0x110000) if chr(code).isspace()],
    })
    save_json(output / "manifest.json", {
        "format": "airline-rnn-browser-v1", "dtype": "float32-le", "tensors": tensors,
        "parameters": predictor.model.count_params(), "labels": predictor.names,
        "config": config, "representation": representation,
        "seed": predictor.manifest["seed"],
        "source_model_sha256": predictor.manifest["trained_model_sha256"],
        "source_tokenizer_sha256": predictor.manifest["tokenizer_sha256"],
        "assets": {name: digest(output / name) for name in
                   ("weights.bin", "tokenizer.json", "preprocessing.json")},
        "softmax_is_calibrated": False,
    })
    print(json.dumps({"output": str(output), "parameters": offset,
                      "weights_bytes": (output / "weights.bin").stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
