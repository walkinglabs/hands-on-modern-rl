# English experiment Notebooks

The [English lab](https://walkinglabs.github.io/hands-on-modern-rl/en/labs/) opens English explanations, exercises and references for all 26 experiments. The fourteen self-contained chapter Notebooks are stored here. The twelve Studio companions already have English originals in the parent directory.

The fourteen chapter Notebooks use the same training code and budgets as their Chinese counterparts. Their English versions were executed in fresh local CPU kernels; [execution evidence](../../chapter-notebooks/validation.en.json) records file hashes, timings and actual results. Colab, ModelScope and the native GPU simulators retain their separately documented verification limits.

Edit the English teaching text in `code/chapter-notebooks/en/`, then regenerate the counterparts with `python scripts/build_localized_notebooks.py`. Existing parent-directory URLs remain available for compatibility.

Each completed run is saved inside its Notebook, including execution counts, figures and printed results. The lab also links to a standalone result page. Rebuilding teaching text preserves outputs when the code fingerprint matches.
