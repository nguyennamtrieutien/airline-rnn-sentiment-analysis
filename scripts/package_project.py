"""Create a portable project ZIP, retaining all scientific runs and provenance."""
from pathlib import Path
import json
import zipfile

root = Path(__file__).resolve().parents[1]
target = root.parent / "airline-rnn-project.zip"
skip_parts = {".venv", ".git", ".build", "node_modules", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".DS_Store"}
with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if not path.is_file() or any(part in skip_parts or part.endswith(".egg-info") for part in relative.parts):
            continue
        if relative.parts[:2] == ("environment", "jupyter"):
            continue
        if relative.name in ("demo.log", "preview_server.log"):
            continue  # live server logs are not experiment provenance
        archive.write(path, Path(root.name) / relative)
with zipfile.ZipFile(target) as archive:
    bad = archive.testzip()
    if bad:
        raise RuntimeError(f"ZIP verification failed: {bad}")
    entries = archive.namelist()
    for name in ("README.md", "data/raw/Tweets.csv", "models/final/model.keras", "notebooks/Airline_RNN_Complete.ipynb",
                 "reports/final/BaoCao_Airline_RNN.docx", "reports/final/ThuyetTrinh_Airline_RNN.pptx",
                 "reports/final/KichBan_ThuyetTrinh_Demo.md"):
        assert f"airline-rnn/{name}" in entries
    assert not any(set(Path(name).parts) & skip_parts for name in entries)
print(json.dumps({"archive": str(target), "size_mib": round(target.stat().st_size / 2**20, 1),
                  "files": len(entries), "crc_check": "passed", "venv_and_machine_kernel_excluded": True,
                  "private_build_and_node_modules_excluded": True}, indent=2))
