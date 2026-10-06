"""Prepare CI parity fixtures from recorded TF predictions (no ML install)."""
from pathlib import Path
import csv
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "deploy/browser-model/manifest.json").read_text())
    reference = json.loads((ROOT / "deploy/browser-reference-cases.json").read_text())
    if reference["source_model_sha256"] != manifest["source_model_sha256"]:
        raise ValueError("Browser parity references belong to a different checkpoint")
    cases = []
    with (ROOT / "results/predictions/B5-balanced_seed-3407_test.csv").open() as stream:
        for row in csv.DictReader(stream):
            cases.append({"text": row["raw_text"], "clean_text": row["clean_text"],
                          "token_length": int(row["token_length"]), "unknown_tokens": int(row["unknown_tokens"]),
                          "truncated": row["truncated"] == "True", "sentiment": row["predicted_label"],
                          "probabilities": {name: float(row["prob_" + name]) for name in manifest["labels"]}})
    output = ROOT / ".build/browser/verification-input.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"original_test_cases": len(cases), "cases": cases + reference["cases"]}))
    print(output)


if __name__ == "__main__":
    main()
