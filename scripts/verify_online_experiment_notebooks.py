#!/usr/bin/env python3
"""Validate the one-to-one Studio, notebook, README, and hero-link mapping."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "scripts" / "build_online_experiment_notebooks.py"


def load_specs() -> tuple[list[dict], object]:
    module_spec = importlib.util.spec_from_file_location("online_notebook_builder", GENERATOR)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Cannot import {GENERATOR}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.EXPERIMENTS, module


def main() -> None:
    specs, builder = load_specs()
    errors: list[str] = []
    slugs = [item["slug"] for item in specs]
    if len(slugs) != len(set(slugs)):
        errors.append("experiment slugs are not unique")

    actual_spaces = sorted(
        path.name
        for path in (ROOT / "modelscope-space").glob("hands-on-modern-rl-*")
        if path.is_dir()
    )
    if sorted(slugs) != actual_spaces:
        errors.append(f"Studio/spec mismatch: expected={sorted(slugs)} actual={actual_spaces}")

    root_readmes = [ROOT / "README.md", ROOT / "README.zh.md", ROOT / "code" / "online-experiments" / "README.md"]
    for spec in specs:
        slug = spec["slug"]
        notebook_path = ROOT / "code" / "online-experiments" / f"{slug}.ipynb"
        studio_dir = ROOT / "modelscope-space" / slug
        expected_url = builder.notebook_url(slug)

        if not notebook_path.exists():
            errors.append(f"{slug}: notebook is missing")
            continue
        try:
            notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{slug}: invalid notebook JSON: {exc}")
            continue

        if notebook.get("nbformat") != 4:
            errors.append(f"{slug}: expected nbformat 4")
        metadata = notebook.get("metadata", {}).get("homrl", {})
        if metadata.get("studio") != slug or metadata.get("modelscope_notebook_url") != expected_url:
            errors.append(f"{slug}: notebook metadata does not match its Studio")
        if metadata.get("resource") != spec["resource"]:
            errors.append(f"{slug}: resource metadata mismatch")

        for index, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            try:
                ast.parse(cell.get("source", ""))
            except SyntaxError as exc:
                errors.append(f"{slug}: code cell {index} does not compile: {exc}")

        notebook_source = "\n".join(str(cell.get("source", "")) for cell in notebook.get("cells", []))
        for required_text in (
            "STEPS_PER_EPOCH",
            "EPOCHS",
            "Epoch model:",
            "Visualize selected policy",
            "No saved epoch model was found",
        ):
            if required_text not in notebook_source:
                errors.append(f"{slug}: notebook is missing epoch/inference workflow marker {required_text!r}")
        if spec["kind"] == "runtime":
            for required_text in ("checkpoints=EPOCHS", "notebook-models.json", "persist_epoch_models"):
                if required_text not in notebook_source:
                    errors.append(f"{slug}: runtime notebook is missing {required_text!r}")
        elif "playground.load_models(EXPERIMENT)" not in notebook_source:
            errors.append(f"{slug}: Gymnasium notebook does not reload saved epoch models")

        studio_readme = studio_dir / "README.md"
        if expected_url not in studio_readme.read_text(encoding="utf-8"):
            errors.append(f"{slug}: Studio README is missing its companion URL")
        for readme in root_readmes:
            readme_url = expected_url
            if readme.name == "README.zh.md":
                readme_url = expected_url.replace(
                    "/code/online-experiments/", "/code/online-experiments/zh/"
                )
            if readme_url not in readme.read_text(encoding="utf-8"):
                errors.append(f"{slug}: {readme.relative_to(ROOT)} is missing its companion URL")

        if slug.endswith("experiment-gymnasium"):
            hero_source = (studio_dir / "app.py").read_text(encoding="utf-8")
            if expected_url not in hero_source.replace('"\n    "', ""):
                errors.append(f"{slug}: Gymnasium app does not contain its Notebook URL")
        else:
            runtime_source = (studio_dir / "space_runtime.py").read_text(encoding="utf-8")
            hero_source = (studio_dir / "game_ui.py").read_text(encoding="utf-8")
            if expected_url not in runtime_source.replace('"\n    "', ""):
                errors.append(f"{slug}: runtime does not contain its Notebook URL")
            if "space.get(\"notebook_url\")" not in hero_source or 'copy["notebook"]' not in hero_source:
                errors.append(f"{slug}: shared hero is missing the Notebook button")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"Verified {len(specs)} Studios, {len(specs)} notebooks, all README links, and all hero buttons.")


if __name__ == "__main__":
    main()
