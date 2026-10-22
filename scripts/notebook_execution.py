"""Retain executed outputs while rebuilding teaching text around unchanged code."""

import copy
import hashlib
import json
from pathlib import Path


def cell_text(cell):
    source = cell.get("source", "")
    return source if isinstance(source, str) else "".join(source)


def code_digest(notebook):
    sources = [cell_text(cell).strip() for cell in notebook["cells"]
               if cell["cell_type"] == "code"]
    return hashlib.sha256(json.dumps(sources, ensure_ascii=False).encode()).hexdigest()


def without_execution(notebook):
    result = copy.deepcopy(notebook)
    result["cells"] = [cell for cell in result["cells"]
                       if not cell.get("metadata", {}).get("saved_run_result")]
    result["metadata"].pop("execution", None)
    result["metadata"].pop("widgets", None)
    for cell in result["cells"]:
        if cell["cell_type"] == "code":
            cell.update(outputs=[], execution_count=None)
    return result


def with_saved_outputs(notebook, destination):
    path = Path(destination)
    if not path.exists():
        return notebook
    saved = json.loads(path.read_text(encoding="utf-8"))
    evidence = saved.get("metadata", {}).get("execution", {})
    if evidence.get("code_sha256") != code_digest(notebook):
        return notebook
    if code_digest(saved) != code_digest(notebook):
        return notebook
    source = [cell for cell in saved["cells"] if cell["cell_type"] == "code"]
    target = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    for old, new in zip(source, target):
        new["outputs"] = copy.deepcopy(old.get("outputs", []))
        new["execution_count"] = old.get("execution_count")
    notebook["metadata"]["execution"] = copy.deepcopy(evidence)
    if "widgets" in saved["metadata"]:
        notebook["metadata"]["widgets"] = copy.deepcopy(saved["metadata"]["widgets"])
    notebook["cells"].extend(copy.deepcopy(cell) for cell in saved["cells"]
                             if cell.get("metadata", {}).get("saved_run_result"))
    return notebook
