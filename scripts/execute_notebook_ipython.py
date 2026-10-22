"""Execute one Notebook in a dedicated IPython process and retain rich outputs."""

import argparse
import os
import traceback
from pathlib import Path

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("workspace", type=Path)
    args = parser.parse_args()
    notebook = nbformat.read(args.source, as_version=4)
    os.chdir(args.workspace)
    shell = InteractiveShell.instance()
    error = None
    try:
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type != "code":
                continue
            print(f"Executing code cell {index + 1}/{len(notebook.cells)}", flush=True)
            with capture_output() as captured:
                result = shell.run_cell(cell.source, store_history=True)
            cell.execution_count = shell.execution_count - 1
            cell.outputs = []
            for name in ["stdout", "stderr"]:
                text = getattr(captured, name)
                if text:
                    cell.outputs.append(nbformat.v4.new_output("stream", name=name, text=text))
            for output in captured.outputs:
                cell.outputs.append(nbformat.v4.new_output(
                    "display_data", data=output.data, metadata=output.metadata))
            error = result.error_before_exec or result.error_in_exec
            if error is not None:
                cell.outputs.append(nbformat.v4.new_output(
                    "error", ename=type(error).__name__, evalue=str(error),
                    traceback=traceback.format_exception(type(error), error, error.__traceback__)))
                raise error
        try:
            from ipywidgets import Widget
            notebook.metadata["widgets"] = {
                "application/vnd.jupyter.widget-state+json": Widget.get_manager_state()}
        except ImportError:
            pass
    finally:
        nbformat.write(notebook, args.destination)


if __name__ == "__main__":
    main()
