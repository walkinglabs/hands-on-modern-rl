"""Verify generated chapter labs, local execution evidence, and bilingual catalog links.

Uses only the standard library. Numerical experiments are executed separately with
execute_chapter_notebooks.py, which needs a notebook-capable virtual environment.
"""

import ast
import hashlib
import json
from pathlib import Path

from build_chapter_notebooks import LABS, ROOT, build_notebook


def main():
    component_dir = ROOT / "docs/.vitepress/theme/components"
    catalog = json.loads((component_dir / "lab-catalog.json").read_text())
    english = json.loads((component_dir / "lab-english.json").read_text())
    report = json.loads((ROOT / "code/chapter-notebooks/validation.json").read_text())
    assert set(report["notebooks"]) == set(LABS)
    assert len(catalog["labs"]) == 22
    assert len({lab["id"] for lab in catalog["labs"]}) == 22
    theme = (ROOT / "docs/.vitepress/theme/index.js").read_text()
    assert "component('ChapterNotebookLink', ChapterNotebookLink)" in theme
    for lab in catalog["labs"]:
        lab_id = lab["id"]
        assert lab["path"] and (ROOT / lab["path"]).is_file(), lab_id
        for key in ["name", "desc", "result", "next"]:
            assert english[lab_id].get(key), (lab_id, key)
        for prefix in ["", "en/"]:
            lesson = ROOT / "docs" / prefix / lab["lesson"]
            lesson = lesson / "index.md" if lesson.is_dir() else lesson.with_suffix(".md")
            assert lesson.is_file(), lesson
        if lab_id not in LABS:
            continue
        assert lab["validation"] == "local-cpu-executed"
        path = ROOT / lab["path"]
        notebook = json.loads(path.read_text())
        assert notebook == build_notebook(lab_id), f"Regenerate {lab_id}"
        evidence = report["notebooks"][lab_id]
        assert evidence["status"] == "passed" and evidence["metrics"], lab_id
        assert evidence["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), lab_id
        texts = []
        for cell in notebook["cells"]:
            text = "".join(cell["source"])
            texts.append(text)
            if cell["cell_type"] == "code":
                tree = ast.parse(text)
                assert not any(isinstance(node, ast.Pass) for node in ast.walk(tree)), lab_id
                assert not any("TODO" in line for line in text.splitlines()), lab_id
        combined = "\n".join(texts)
        for marker in ["Checklist", "Exercise", "References", "metrics.json", "plt.show()"]:
            assert marker in combined, (lab_id, marker)
    print("Verified 10 generated and locally executed chapter labs, 22 catalog paths, and bilingual links")


if __name__ == "__main__":
    main()
