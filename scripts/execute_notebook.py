"""Execute all cells with the project venv kernel and write notebook + HTML."""
from pathlib import Path
import os
import sys

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter

root = Path(__file__).resolve().parents[1]
path = root / "notebooks/Airline_RNN_Complete.ipynb"
kernel_dir = root / "environment/jupyter/share/jupyter/kernels/airline-rnn"
kernel_dir.mkdir(parents=True, exist_ok=True)
(kernel_dir / "kernel.json").write_text(__import__("json").dumps({
    "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "Airline RNN", "language": "python"}), encoding="utf-8")
os.environ["JUPYTER_PATH"] = str(root / "environment/jupyter/share/jupyter")
notebook = nbformat.read(path, as_version=4)
client = NotebookClient(notebook, kernel_name="airline-rnn", timeout=14400,
                        resources={"metadata": {"path": str(root)}})
client.execute()
nbformat.validate(notebook)
nbformat.write(notebook, path)
exporter = HTMLExporter(template_name="lab")
html, _ = exporter.from_notebook_node(notebook)
path.with_suffix(".html").write_text(html, encoding="utf-8")
print(f"Executed {sum(cell.cell_type == 'code' for cell in notebook.cells)} code cells: {path}")
print(path.with_suffix(".html"))
