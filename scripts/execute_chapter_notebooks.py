"""Execute chapter Notebooks in fresh kernels and keep evidence outside the source tree.

Install nbformat, nbclient, and ipykernel in a virtual environment first.
python scripts/execute_chapter_notebooks.py --output-dir /tmp/rl-labs [--labs bandit gridworld]
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
from nbclient import NotebookClient

from build_chapter_notebooks import LABS, ROOT
from notebook_execution import code_digest


def main():
    catalog = json.loads((ROOT / "docs/.vitepress/theme/components/lab-catalog.json").read_text())
    all_labs = {lab["id"]: lab for lab in catalog["labs"]}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--labs", nargs="+", choices=all_labs, default=list(LABS))
    parser.add_argument("--locale", choices=["zh", "en"], default="zh")
    parser.add_argument("--engine", choices=["jupyter", "ipython"], default="jupyter")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["HANDS_ON_LAB_OUTPUT"] = str(output / "artifacts")
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["HANDS_ON_LAB_PREINSTALLED"] = "1"
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.machine(),
        "device": "CPU",
        "locale": args.locale,
        "engine": args.engine,
        "packages": {name: importlib.metadata.version(name) for name in
                     ["numpy", "torch", "matplotlib", "gymnasium", "stable-baselines3"]},
        "hosted_platforms": "Colab and ModelScope not executed in this local run",
        "notebooks": {},
    }
    with tempfile.TemporaryDirectory(prefix="rl-lab-kernel-") as directory:
        kernel_dir = Path(directory) / "hands-on-lab"
        kernel_dir.mkdir()
        (kernel_dir / "kernel.json").write_text(json.dumps({
            "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
            "display_name": "Hands-on RL validation", "language": "python",
        }))
        for lab_id in args.labs:
            if lab_id in {"unity", "ai2thor", "minecraft"}:
                os.environ.pop("SDL_VIDEODRIVER", None)
            else:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
            path = ROOT / all_labs[lab_id]["paths"][args.locale]
            os.environ["HANDS_ON_LAB_ID"] = lab_id
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            workspace = output / "workspaces" / lab_id
            workspace.mkdir(parents=True, exist_ok=True)
            if lab_id not in LABS:
                slug = Path(all_labs[lab_id]["path"]).stem
                isolated = workspace / "modelscope-space" / slug
                shutil.copytree(ROOT / "modelscope-space" / slug, isolated, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns("artifacts", "__pycache__", ".cache"))
            manager = KernelManager(kernel_name="hands-on-lab", kernel_spec_manager=
                                    KernelSpecManager(kernel_dirs=[directory]))
            client = NotebookClient(notebook, km=manager, timeout=1200,
                                    resources={"metadata": {"path": str(workspace)}})
            print(f"Running {lab_id} in a fresh {args.engine} process on CPU...", flush=True)
            started = time.monotonic()
            try:
                if args.engine == "ipython":
                    subprocess.run([
                        sys.executable, str(ROOT / "scripts/execute_notebook_ipython.py"),
                        str(path), str(output / f"{lab_id}-executed.ipynb"), str(workspace),
                    ], check=True, timeout=1200 * sum(c.cell_type == "code" for c in notebook.cells))
                    notebook = nbformat.read(output / f"{lab_id}-executed.ipynb", as_version=4)
                else:
                    client.execute()
                metrics_path = output / "artifacts" / lab_id / "metrics.json"
                metrics = json.loads(metrics_path.read_text())
                item = {"status": "passed", "seconds": round(time.monotonic() - started, 2),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
                        "code_sha256": code_digest(notebook),
                        "metrics": metrics}
            except Exception as error:
                item = {"status": "failed", "seconds": round(time.monotonic() - started, 2),
                        "error": str(error)}
                raise
            finally:
                if args.engine == "ipython" and (output / f"{lab_id}-executed.ipynb").exists():
                    notebook = nbformat.read(output / f"{lab_id}-executed.ipynb", as_version=4)
                nbformat.write(notebook, output / f"{lab_id}-executed.ipynb")
                report["notebooks"][lab_id] = item
                (output / "validation.json").write_text(json.dumps(report, indent=2))
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
            print(f"Passed {lab_id}: {item['seconds']}s; {json.dumps(metrics)}", flush=True)
    print(f"Evidence: {output / 'validation.json'}", flush=True)


if __name__ == "__main__":
    main()
