"""Execute a notebook in place (the outputs are saved into the notebook).

    python notebooks/execute_notebook.py notebooks/01_human_immune.ipynb [kernel] [--keep-stderr]

Needs nbformat + nbclient in the calling Python; `kernel` (default: python3) names the Jupyter kernel that holds the
analysis environment. Text written to stderr (library warnings, which carry local installation paths) is not stored
unless --keep-stderr is given.
"""
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

args = [a for a in sys.argv[1:] if a != "--keep-stderr"]
keep_stderr = "--keep-stderr" in sys.argv
path = Path(args[0]).resolve()
kernel = args[1] if len(args) > 1 else "python3"
nb = nbformat.read(path, as_version=4)
t0 = time.time()
try:
    NotebookClient(nb, kernel_name=kernel, timeout=-1, resources={"metadata": {"path": str(path.parent)}}).execute()
finally:
    if not keep_stderr:
        for cell in nb.cells:
            if cell.cell_type == "code":
                cell.outputs = [o for o in cell.outputs if not (o.output_type == "stream" and o.name == "stderr")]
    nbformat.write(nb, path)          # keep the outputs up to a failing cell
print(f"executed {path.name} in {time.time() - t0:.0f} s")
