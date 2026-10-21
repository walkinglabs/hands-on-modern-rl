"""Check all Chinese/English lab destinations and the English CPU run evidence."""

import ast
import hashlib
import json
import re

from build_chapter_notebooks import LABS, ROOT
from build_localized_notebooks import build_all, cell_text


def main():
    build_all(check=True)
    catalog = json.loads((ROOT / "docs/.vitepress/theme/components/lab-catalog.json").read_text(encoding="utf-8"))
    evidence = json.loads((ROOT / "code/chapter-notebooks/validation.en.json").read_text(encoding="utf-8"))
    assert set(evidence["notebooks"]) == set(LABS)
    assert evidence["locale"] == "en"
    cjk = re.compile(r"[\u3400-\u9fff]")
    paths = set()
    for lab in catalog["labs"]:
        assert set(lab["paths"]) == {"zh", "en"}, lab["id"]
        for locale, relative in lab["paths"].items():
            paths.add(relative)
            path = ROOT / relative
            notebook = json.loads(path.read_text(encoding="utf-8"))
            for cell in notebook["cells"]:
                text = cell_text(cell)
                if cell["cell_type"] == "code":
                    ast.parse(text)
                if locale == "en":
                    assert not cjk.search(text), (lab["id"], locale, text[:100])
                elif cell["cell_type"] == "markdown":
                    assert cjk.search(text), (lab["id"], locale)
            if locale == "en" and lab["id"] in LABS:
                record = evidence["notebooks"][lab["id"]]
                assert record["status"] == "passed"
                assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert len(paths) == 44
    print("Verified 22 Chinese and 22 English Notebooks, language coverage and 10 English CPU runs")


if __name__ == "__main__":
    main()
