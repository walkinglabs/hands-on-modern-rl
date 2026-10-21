"""Build the self-contained chapter labs from readable percent-format sources.

Run with Python 3.10+: python scripts/build_chapter_notebooks.py [--check].
The original twelve Studio notebooks keep their existing, separate generator.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "code/chapter-notebooks"
DESTINATION = ROOT / "code/online-experiments"
LABS = {
    "bandit": "Exploration and regret",
    "gridworld": "Value iteration and Q-learning",
    "dqn": "DQN and Double DQN",
    "reinforce": "REINFORCE and a learned baseline",
    "pendulum": "Continuous Actor-Critic control",
    "dpo": "Preference learning with a tiny language model",
    "grpo": "Group-relative advantages and verifiable rewards",
    "financial": "GRPO with mock financial tools",
    "worldmodel": "Learned dynamics and planning",
    "offline": "Offline learning and action coverage",
    "agentic-tools": "Multi-turn tools and credit assignment",
    "agentic-code": "A code-repair agent with actual Python tests",
    "agentic-finqa": "A multi-turn SQL and calculator agent",
    "agentic-research": "Retrieval, evidence selection and citations",
}


def build_notebook(lab_id):
    source = (SOURCES / f"{lab_id}.py").read_text(encoding="utf-8")
    chunks = []
    kind, lines = None, []
    for line in source.splitlines():
        if line.startswith("# %%"):
            if kind:
                chunks.append((kind, lines))
            kind = "markdown" if "[markdown]" in line else "code"
            lines = []
            if "[include:agentic-common]" in line:
                lines = (SOURCES / "_agentic_common.py").read_text(encoding="utf-8").splitlines()
        elif line.startswith("from _agentic_common import "):
            # Inline this module above so the Notebook also runs without the source tree.
            continue
        else:
            lines.append(line)
    if kind:
        chunks.append((kind, lines))
    cells = []
    for index, (kind, lines) in enumerate(chunks):
        if kind == "markdown":
            lines = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in lines]
        text = "\n".join(lines).strip() + "\n"
        if kind == "code":
            ast.parse(text)
            for number, line in enumerate(text.splitlines(), 1):
                if len(line) > 100:
                    raise ValueError(f"{lab_id}: cell {index}, line {number} exceeds 100 characters")
        cell = {
            "cell_type": kind,
            "id": hashlib.sha256(f"{lab_id}-{index}".encode()).hexdigest()[:12],
            "metadata": {},
            "source": text.splitlines(keepends=True),
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
            "colab": {"name": f"hands-on-modern-rl-{lab_id}.ipynb", "provenance": []},
            "title": LABS[lab_id],
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for lab_id in LABS:
        notebook = build_notebook(lab_id)
        path = DESTINATION / f"hands-on-modern-rl-{lab_id}.ipynb"
        rendered = json.dumps(notebook, ensure_ascii=False, indent=2) + "\n"
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != rendered:
                raise SystemExit(f"Outdated Notebook: {path}")
        else:
            path.write_text(rendered, encoding="utf-8")
    print(f"{'Checked' if args.check else 'Built'} {len(LABS)} chapter Notebooks")


if __name__ == "__main__":
    main()
