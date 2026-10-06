"""Compare JS inference with saved held-out predictions and fresh TF edge cases."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from airline_rnn.analysis import AnalysisRunner
from airline_rnn.common import save_json
from airline_rnn.inference import SentimentPredictor


def main():
    predictor = SentimentPredictor(ROOT)
    runner = AnalysisRunner(predictor)
    cases = []
    with (ROOT / "results/predictions/B5-balanced_seed-3407_test.csv").open() as stream:
        for row in csv.DictReader(stream):
            cases.append({"text": row["raw_text"], "clean_text": row["clean_text"],
                          "token_length": int(row["token_length"]), "unknown_tokens": int(row["unknown_tokens"]),
                          "truncated": row["truncated"] == "True", "sentiment": row["predicted_label"],
                          "probabilities": {name: float(row["prob_" + name]) for name in predictor.names}})
    original_test_cases = len(cases)
    extras = ["Thank you for the helpful service!", "My flight was delayed again and nobody helped me.",
              "What time does flight 123 depart?", "I can't recommend this airline.",
              "I cannot complain; it's great.", "We weren't happy, they're not helping.",
              "@united Thanks! See https://example.com/a?q=x and www.example.org.",
              "&amp; &notit; &#x27; &#39; &apos; can't &unknown; &#0; &#128; &#1114111;",
              "Ｉ ＣＡＮ’Ｔ fly! İ can't! Γcan't! ‘won’t’", "😀✈️!!!", "flibbertigibbet xyzunknown __proto__ constructor",
              "good " * 500, "bad " * 1000,
              "I'm sure you'll help; we've waited and I'd like an answer.",
              "shan't ain't won't that's what's there's here's let's", "foo\u001cbar\u2003baz\ufeffqux",
              "https://example.org\u001dcan't wait", "one&#1;two&#xD800;three&#9999999999999999;four"]
    for text in extras:
        events = list(runner.events(text))
        final = events[-1]["result"]
        padding = next(event["data"] for event in events if event["type"] == "step_done" and event["step"] == 5)
        trace = next(event["data"] for event in events if event["type"] == "step_done" and event["step"] == 6)
        cases.append({"text": text, **{key: final[key] for key in
                     ("clean_text", "token_length", "unknown_tokens", "truncated", "sentiment", "probabilities")},
                      "ids": padding["ids"], "hidden_preview": trace["last_hidden_state_preview"]})
    fixtures = ROOT / ".build/browser/verification-input.json"
    save_json(fixtures, {"original_test_cases": original_test_cases, "cases": cases})
    node = os.environ.get("AIRLINE_NODE") or shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required to verify browser inference")
    completed = subprocess.run([node, str(ROOT / "scripts/verify_browser_model.mjs"), str(fixtures)],
                               cwd=ROOT, text=True, capture_output=True, check=True)
    result = json.loads(completed.stdout)
    save_json(ROOT / "results/metrics/browser_inference_verification.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
