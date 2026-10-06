from __future__ import annotations

import argparse
import json
import shutil

from .common import initialize, project_root, read_json


def bootstrap(root):
    root = initialize(root)
    source = project_root()
    if root != source:
        for path in (source / "sources").glob("*"):
            destination = root / "sources" / path.name
            if path.is_file() and not destination.exists():
                shutil.copy2(path, destination)
    return root


def main():
    parser = argparse.ArgumentParser(description="Reproducible RNN airline sentiment experiments")
    parser.add_argument("command", choices=["prepare", "smoke", "train", "evaluate", "report", "all", "infer", "demo", "status"])
    parser.add_argument("--root", help="Output project root; choose a NEW root to train a fresh matrix")
    parser.add_argument("--tier", choices=["minimum", "recommended", "advanced"], default="recommended")
    parser.add_argument("--restart-incomplete", action="store_true", help="Archive interrupted run and restart its seed")
    parser.add_argument("--text", help="An English tweet for inference")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    root = bootstrap(args.root)
    if args.command in ("prepare", "smoke", "all"):
        from .data import prepare_data
        from .reporting import generate_eda
        profile = prepare_data(root)
        generate_eda(root)
        print(json.dumps(profile, ensure_ascii=False, indent=2))
    if args.command == "smoke":
        from .config import Configuration
        from .training import run_experiment
        run_experiment(root, Configuration("C0"), 42, smoke=True, max_epochs=2, restart_incomplete=args.restart_incomplete)
    if args.command in ("train", "all"):
        from .training import run_matrix
        # Once test is released, enforce a complete cache before allowing a re-run.
        if (root / "results/metrics/final_selection.json").exists():
            from .evaluation import verified_runs
            verified_runs(root)
        run_matrix(root, args.tier, restart_incomplete=args.restart_incomplete)
    if args.command in ("evaluate", "all"):
        from .evaluation import evaluate_final, freeze_selection
        print(json.dumps(freeze_selection(root), ensure_ascii=False, indent=2))
        print(evaluate_final(root).to_string(index=False))
    if args.command in ("report", "all"):
        from .reporting import generate_eda, generate_results, write_report
        generate_eda(root)
        generate_results(root)
        write_report(root)
        print(f"Reports and figures: {root / 'reports'}")
    if args.command == "infer":
        from .inference import SentimentPredictor
        if not args.text:
            parser.error("infer requires --text")
        print(json.dumps(SentimentPredictor(root).predict(args.text), ensure_ascii=False, indent=2))
    if args.command == "demo":
        from .demo import serve
        serve(root, args.port)
    if args.command == "status":
        paths = sorted((root / "models/runs").glob("*/seed-*/run.json"))
        for path in paths:
            run = read_json(path)
            print(f"{run['config']['id']:12s} seed={run['seed']:4d} {run['status']:11s} val Macro-F1={run.get('validation_macro_f1', 'pending')}")
        selection = root / "results/metrics/final_selection.json"
        print("Selection:", read_json(selection) if selection.exists() else "not frozen; test locked")


if __name__ == "__main__":
    main()
