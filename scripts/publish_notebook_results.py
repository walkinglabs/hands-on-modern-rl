"""Publish checked notebook outputs and standalone result pages from an execution run."""

import argparse
import copy
import hashlib
import html
import json
import re
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from build_chapter_notebooks import ROOT, LABS
from notebook_execution import code_digest, without_execution

SITE = "https://walkinglabs.github.io/hands-on-modern-rl"


def scalar_rows(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from scalar_rows(item, f"{prefix}.{key}".strip("."))
    elif not isinstance(value, list):
        yield prefix, value


def render_outputs(notebook):
    sections = []
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            for mime in ["image/gif", "image/png", "image/jpeg", "image/svg+xml"]:
                if mime in data:
                    payload = data[mime]
                    payload = payload if isinstance(payload, str) else "".join(payload)
                    if mime == "image/svg+xml":
                        sections.append(payload)
                    else:
                        sections.append(f'<img alt="Experiment output" src="data:{mime};base64,{payload}">')
                    break
            else:
                if "text/html" in data:
                    payload = data["text/html"]
                    sections.append(payload if isinstance(payload, str) else "".join(payload))
                else:
                    text = output.get("text", data.get("text/plain", ""))
                    text = text if isinstance(text, str) else "".join(text)
                    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
                    if text.strip():
                        sections.append("<pre>" + html.escape(text) + "</pre>")
    return "\n".join(sections)


def publish(run_dir, locale):
    report_path = run_dir / "validation.json"
    report = json.loads(report_path.read_text())
    catalog = json.loads((ROOT / "docs/.vitepress/theme/components/lab-catalog.json").read_text())
    english = json.loads((ROOT / "docs/.vitepress/theme/components/lab-english.json").read_text())
    by_id = {lab["id"]: lab for lab in catalog["labs"]}
    destination = ROOT / "docs/public/lab-results"
    destination.mkdir(parents=True, exist_ok=True)
    local_time = datetime.fromisoformat(report["started_at"]).astimezone(ZoneInfo("Asia/Shanghai"))
    display_time = local_time.strftime("%Y-%m-%d %H:%M:%S UTC+8")
    for lab_id, record in report["notebooks"].items():
        if record["status"] != "passed":
            continue
        record["environment"] = {key: value for key, value in report.items() if key != "notebooks"}
        lab = by_id[lab_id]
        path = ROOT / lab["paths"][locale]
        current = without_execution(json.loads(path.read_text()))
        executed = json.loads((run_dir / f"{lab_id}-executed.ipynb").read_text())
        assert code_digest(current) == code_digest(executed), f"Changed execution code: {lab_id}"
        codes = [cell for cell in current["cells"] if cell["cell_type"] == "code"]
        completed = [cell for cell in executed["cells"] if cell["cell_type"] == "code"]
        assert all(cell.get("execution_count") is not None for cell in completed), lab_id
        assert not any(output.get("output_type") == "error" for cell in completed
                       for output in cell.get("outputs", [])), lab_id
        assert any(cell.get("outputs") for cell in completed), lab_id
        for source, target in zip(completed, codes):
            target.update(outputs=copy.deepcopy(source.get("outputs", [])),
                          execution_count=source["execution_count"])
        execution = {"status": "passed", "code_sha256": code_digest(current),
                     "executed_at": report["started_at"], "seconds": record["seconds"],
                     "device": report["device"], "python": report["python"],
                     "result_page": f"lab-results/{lab_id}-{locale}.html"}
        current["metadata"]["execution"] = execution
        if "widgets" in executed["metadata"]:
            current["metadata"]["widgets"] = executed["metadata"]["widgets"]
        label = "本次运行结果" if locale == "zh" else "Saved run results"
        link_label = "查看完整结果页" if locale == "zh" else "Open the full result page"
        text = (f"## {label}\n\n{report['device']} · Python {report['python']} · "
                f"{record['seconds']} s\n\n[{link_label}]({SITE}/{execution['result_page']})\n")
        current["cells"].append({"cell_type": "markdown", "id": f"run-{lab_id}-{locale}",
            "metadata": {"saved_run_result": True}, "source": text})
        path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n")
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record["code_sha256"] = execution["code_sha256"]
        title = lab["name"] if locale == "zh" else english[lab_id]["name"]
        rows = list(scalar_rows(record["metrics"]))[:30]
        table = "".join(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(value))}</td></tr>"
                        for key, value in rows)
        content = f'''<!doctype html><html lang="{locale}"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{margin:0;background:#f7f8fb;color:#24324a;font:16px/1.7 system-ui,sans-serif}}
main{{max-width:960px;margin:40px auto;padding:24px;background:white;border-radius:16px}}
h1{{font-size:28px;line-height:1.35}}a{{color:#4c4bc4}}img,svg{{display:block;max-width:100%;height:auto;margin:24px auto}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere;padding:16px;background:#f4f5f8;border-radius:8px;font-size:13px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px;text-align:left;border-bottom:1px solid #eee;overflow-wrap:anywhere}}
small{{color:#68778b}}@media(max-width:600px){{main{{margin:0;padding:20px;border-radius:0}}}}</style>
<main><a href="{SITE}/{'en/' if locale == 'en' else ''}labs/">← {'实验室' if locale == 'zh' else 'Lab'}</a>
<h1>{html.escape(title)}</h1><small>{report['device']} · Python {report['python']} · {record['seconds']} s · {display_time}</small>
<h2>{label}</h2><table>{table}</table>{render_outputs(executed)}
<details><summary>metrics.json</summary><pre>{html.escape(json.dumps(record['metrics'], ensure_ascii=False, indent=2))}</pre></details>
</main></html>'''
        (destination / f"{lab_id}-{locale}.html").write_text(
            "\n".join(line.rstrip() for line in content.splitlines()) + "\n")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    target = ROOT / "code/chapter-notebooks" / ("validation.en.json" if locale == "en" else "validation.json")
    evidence = json.loads(target.read_text())
    evidence["notebooks"].update({key: value for key, value in report["notebooks"].items() if key in LABS})
    evidence["additional_runs"] = [{key: value for key, value in report.items() if key != "notebooks"}]
    target.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n")
    studio = {key: value for key, value in report["notebooks"].items() if key not in LABS}
    if studio:
        studio_path = ROOT / f"code/online-experiments/validation.studio.{locale}.json"
        saved = json.loads(studio_path.read_text()) if studio_path.exists() else {"notebooks": {}}
        saved["notebooks"].update(studio)
        run_metadata = {key: value for key, value in report.items() if key != "notebooks"}
        runs = saved.get("additional_runs", [])
        saved["additional_runs"] = [item for item in runs
            if item.get("started_at") != report["started_at"]] + [run_metadata]
        studio_path.write_text(json.dumps(saved, ensure_ascii=False, indent=2) + "\n")
    passed = sum(item["status"] == "passed" for item in report["notebooks"].values())
    print(f"Published outputs and result pages for {passed} {locale} Notebooks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--locale", choices=["zh", "en"], required=True)
    args = parser.parse_args()
    publish(args.run_dir.resolve(), args.locale)
