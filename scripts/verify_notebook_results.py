"""Check saved outputs, execution fingerprints, result pages and catalog links."""

import hashlib
import json
from pathlib import Path

from build_chapter_notebooks import ROOT, LABS
from notebook_execution import code_digest, cell_text


def main():
    catalog = json.loads((ROOT / "docs/.vitepress/theme/components/lab-catalog.json").read_text())
    checked = 0
    for locale in ["zh", "en"]:
        chapter = ROOT / "code/chapter-notebooks" / ("validation.en.json" if locale == "en" else "validation.json")
        evidence = json.loads(chapter.read_text())["notebooks"]
        evidence.update(json.loads((ROOT / f"code/online-experiments/validation.studio.{locale}.json").read_text())["notebooks"])
        for lab in catalog["labs"]:
            path = ROOT / lab["paths"][locale]
            notebook = json.loads(path.read_text())
            prose = "\n".join(cell_text(cell) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
            for phrase in ["不能据此宣称", "这些检查不能证明", "不能证明通用对话能力", "does not verify arbitrary mathematical reasoning", "These checks do not establish mathematical ability"]:
                assert phrase not in prose, (lab["id"], phrase)
            result_path = lab.get("results", {}).get(locale)
            if not result_path:
                assert not lab["device"].startswith("CPU"), lab["id"]
                continue
            record = evidence[lab["id"]]
            assert record["status"] == "passed"
            assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest(), path
            execution = notebook["metadata"]["execution"]
            assert execution["code_sha256"] == code_digest(notebook) == record["code_sha256"], path
            assert execution["result_page"] == result_path
            cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            assert all(cell["execution_count"] is not None for cell in cells), path
            outputs = [output for cell in cells for output in cell["outputs"]]
            assert outputs and not any(output["output_type"] == "error" for output in outputs), path
            assert any(cell.get("metadata", {}).get("saved_run_result") for cell in notebook["cells"]), path
            html = (ROOT / "docs/public" / result_path).read_text()
            assert "data:image/" in html and "metrics.json" in html, result_path
            checked += 1
    print(f"Verified saved outputs, code fingerprints and result pages for {checked} Notebooks")


if __name__ == "__main__":
    main()
