from __future__ import annotations

import hashlib
import json
import os
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def project_root(root=None):
    return Path(root or os.environ.get("AIRLINE_RNN_ROOT", Path(__file__).resolve().parents[1])).resolve()


def initialize(root):
    root = project_root(root)
    for folder in ("data/raw", "data/interim", "data/processed", "data/splits", "configs",
                   "models/runs", "models/final", "results/histories", "results/figures",
                   "results/metrics", "results/predictions", "results/tables", "reports",
                   "environment", "notebooks", "sources"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    return root


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def training_source_digest():
    folder = Path(__file__).parent
    return object_digest({name: digest(folder / name) for name in
                          ("text.py", "data.py", "config.py", "model.py", "training.py")})


def is_registered_training_source(root, expected):
    """Accept exact current source or the checked, archived naming migration."""
    current = training_source_digest()
    if expected == current:
        return True
    folder = project_root(root) / "sources"
    manifest_path = folder / "naming_migration.json"
    if not manifest_path.exists():
        return False
    manifest = read_json(manifest_path)
    if expected != manifest["before_training_source_sha256"] or current != manifest["after_training_source_sha256"]:
        return False
    archive = folder / manifest["training_snapshot"]
    if not archive.exists() or digest(archive) != manifest["training_snapshot_sha256"]:
        return False
    with zipfile.ZipFile(archive) as snapshot:
        hashes = {name: hashlib.sha256(snapshot.read(name)).hexdigest() for name in
                  ("text.py", "data.py", "config.py", "model.py", "training.py")}
    return object_digest(hashes) == expected
