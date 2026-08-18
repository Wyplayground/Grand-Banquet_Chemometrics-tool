from __future__ import annotations

import html
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd

from .contracts import TaskType, leaderboard_front_columns


DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def structured_summary(results: pd.DataFrame, task: TaskType, top_n: int = 10) -> Dict[str, object]:
    eligible = results[results["Eligible_For_Ranking"].astype(bool)].sort_values("Selection_Rank").head(top_n)
    columns = [
        "Selection_Rank",
        "Pipeline_ID",
        "Preprocessing",
        "Feature_Selection",
        "Model",
        "Selection_Score",
        "Stability_Percentile",
        "Val_Evaluated",
        "Status",
        "Actual_Backend",
    ]
    return {
        "task": TaskType(task).value,
        "selection_role": "all candidate rankings use calibration-set two-level cross-validation evidence only (shown as CV_*)",
        "validation_role": "independent Val_* is added after ranks are frozen and cannot change Selection_Rank",
        "calibration_role": "Cal_* metrics are full-calibration apparent scores and are not used for ranking",
        "successful_pipelines": int(results["Status"].eq("SUCCESS").sum()),
        "failed_pipelines": int((~results["Status"].eq("SUCCESS")).sum()),
        "top_pipelines": json.loads(
            eligible[[column for column in columns if column in eligible]].to_json(orient="records")
        ),
    }


def build_qwen_prompt(summary: Dict[str, object]) -> str:
    task = summary["task"]
    metric_guidance = (
        "Interpret Kappa, MCC, macro-F1, balanced accuracy, sensitivity, and specificity."
        if task == "classification"
        else "Interpret R2, RMSE, MAE, RPD, RPIQ, RER, bias, slope, and intercept."
    )
    return (
        "You are preparing a technical Grand Banquet chemometrics report. "
        f"The task is {task}. {metric_guidance} "
        "Clearly distinguish pipeline selection by calibration-set two-level cross-validation (shown as CV metrics) "
        "from independent validation metrics (Val_*) added after ranks are frozen. "
        "Cal metrics are apparent full-calibration scores only. "
        "Do not use validation metrics to re-rank candidate pipelines; Selection_Rank uses CV only. "
        "Selection Rank 1 is additionally exported as the fitted bundle. "
        "Discuss CV performance, fold stability, Cal-vs-CV overfit diagnostics, failed pipelines, "
        "actual model backends, the Reproducible Analysis Record, and limitations. "
        "Return a concise academic report with sections: Methods, Ranking Results, Stability and Overfitting, "
        "Independent Validation, Limitations, and Recommended Next Steps.\n\n"
        + json.dumps(summary, indent=2, ensure_ascii=False)
    )


def call_qwen(
    summary: Dict[str, object],
    api_key: Optional[str] = None,
    model: str = "qwen-plus",
    client_factory: Optional[Callable[..., object]] = None,
) -> str:
    key = (api_key or os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not key:
        raise ValueError("A DashScope API key is required through the GUI or DASHSCOPE_API_KEY.")
    if client_factory is None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("The openai package is required for Qwen-compatible report generation.") from exc
        client_factory = OpenAI
    client = client_factory(api_key=key, base_url=DASHSCOPE_BASE_URL)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You write evidence-grounded chemometrics reports without overstating validation."},
            {"role": "user", "content": build_qwen_prompt(summary)},
        ],
        temperature=0.2,
        timeout=120,
    )
    text = completion.choices[0].message.content
    if not text or not str(text).strip():
        raise RuntimeError("Qwen returned an empty report.")
    return str(text).strip()


def write_report_files(text: str, output_dir: str | Path, stem: str = "gb_ai_report") -> Tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    text_path = output_dir / f"{stem}_{timestamp}.md"
    docx_path = output_dir / f"{stem}_{timestamp}.docx"
    pdf_path = output_dir / f"{stem}_{timestamp}.pdf"
    text_path.write_text(text, encoding="utf-8")

    try:
        from docx import Document
    except ImportError as exc:
        raise ImportError("python-docx is required for Word report export.") from exc
    document = Document()
    document.add_heading("Grand Banquet Chemometrics Report", level=0)
    for block in text.split("\n\n"):
        stripped = block.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            document.add_heading(stripped.lstrip("# "), level=min(3, len(stripped) - len(stripped.lstrip("#"))))
        else:
            document.add_paragraph(stripped)
    document.save(docx_path)

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError as exc:
        raise ImportError("reportlab is required for PDF report export.") from exc
    styles = getSampleStyleSheet()
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    for style_name in ("Title", "BodyText"):
        styles[style_name].fontName = "STSong-Light"
    story = [Paragraph("Grand Banquet Chemometrics Report", styles["Title"]), Spacer(1, 6 * mm)]
    for block in text.split("\n\n"):
        if block.strip():
            safe = block.strip().replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
            story.extend([Paragraph(safe, styles["BodyText"]), Spacer(1, 3 * mm)])
    SimpleDocTemplate(str(pdf_path), pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm, topMargin=18 * mm, bottomMargin=18 * mm).build(story)
    return text_path, docx_path, pdf_path


def write_leaderboard_html(results: pd.DataFrame, destination: str | Path, task: TaskType) -> Path:
    destination = Path(destination)
    task = TaskType(task)
    front = [column for column in leaderboard_front_columns(task) if column in results.columns]
    remainder = [column for column in results.columns if column not in front]
    ordered = results.loc[:, front + remainder].copy()
    # Links to CV prediction files when Pipeline_ID is present.
    if "Pipeline_ID" in ordered.columns:
        ordered.insert(
            0,
            "CV_Predictions",
            ordered["Pipeline_ID"].map(
                lambda pipeline_id: f"<a href='predictions/{html.escape(str(pipeline_id))}_cv.csv'>cv</a>"
            ),
        )
    table = ordered.to_html(index=False, border=0, escape=False, na_rep="")
    preprocessings = sorted({str(value) for value in results.get("Preprocessing", pd.Series(dtype=object)).dropna()})
    selectors = sorted({str(value) for value in results.get("Feature_Selection", pd.Series(dtype=object)).dropna()})
    models = sorted({str(value) for value in results.get("Model", pd.Series(dtype=object)).dropna()})
    statuses = sorted({str(value) for value in results.get("Status", pd.Series(dtype=object)).dropna()})

    def options(values):
        return "".join(f"<option value='{html.escape(value)}'>{html.escape(value)}</option>" for value in values)

    content = f"""<!doctype html>
<html>
<head>
<meta charset='utf-8'>
<title>Grand Banquet Leaderboard</title>
<style>
body{{font-family:Segoe UI,sans-serif;margin:24px;background:#E0F7FA;color:#263238}}
h1{{color:#00796B}}
.filters{{display:flex;flex-wrap:wrap;gap:12px;margin:12px 0 18px;padding:12px;background:#FFFFFF;border:1px solid #B0BEC5}}
.filters label{{font-size:12px;display:flex;flex-direction:column;gap:4px}}
.table-wrap{{overflow:auto;max-height:78vh;background:white;border:1px solid #B0BEC5}}
table{{border-collapse:collapse;min-width:100%;background:white}}
th{{background:#00796B;color:white;position:sticky;top:0;z-index:1}}
th,td{{border:1px solid #B0BEC5;padding:6px;font-size:12px;white-space:nowrap}}
tr.rank1{{background:#FFF8E1}}
tr.hidden{{display:none}}
.note{{font-size:13px;max-width:960px}}
</style>
</head>
<body>
<h1>Grand Banquet Selection Leaderboard</h1>
<p class='note'>Candidates are ranked using calibration-set two-level cross-validation evidence only (CV_*). Cal_* scores are full-calibration apparent metrics. Independent validation (Val_*) is added after ranks are frozen and cannot change Selection_Rank. Overfit_* columns are diagnostics only.</p>
<div class='filters'>
<label>Preprocessing<select id='filter-pre'><option value=''>All</option>{options(preprocessings)}</select></label>
<label>Feature selection<select id='filter-sel'><option value=''>All</option>{options(selectors)}</select></label>
<label>Model<select id='filter-model'><option value=''>All</option>{options(models)}</select></label>
<label>Status<select id='filter-status'><option value=''>All</option>{options(statuses)}</select></label>
</div>
<div class='table-wrap'>{table}</div>
<script>
const col = (name) => {{
  const headers = Array.from(document.querySelectorAll('table thead th')).map(th => th.textContent.trim());
  return headers.indexOf(name);
}};
const preCol = col('Preprocessing');
const selCol = col('Feature_Selection');
const modelCol = col('Model');
const statusCol = col('Status');
const rankCol = col('Selection_Rank');
document.querySelectorAll('table tbody tr').forEach(row => {{
  if (rankCol >= 0 && row.children[rankCol] && row.children[rankCol].textContent.trim() === '1') {{
    row.classList.add('rank1');
  }}
}});
function applyFilters() {{
  const pre = document.getElementById('filter-pre').value;
  const sel = document.getElementById('filter-sel').value;
  const model = document.getElementById('filter-model').value;
  const status = document.getElementById('filter-status').value;
  document.querySelectorAll('table tbody tr').forEach(row => {{
    const cells = row.children;
    const ok =
      (!pre || (preCol >= 0 && cells[preCol].textContent.trim() === pre)) &&
      (!sel || (selCol >= 0 && cells[selCol].textContent.trim() === sel)) &&
      (!model || (modelCol >= 0 && cells[modelCol].textContent.trim() === model)) &&
      (!status || (statusCol >= 0 && cells[statusCol].textContent.trim() === status));
    row.classList.toggle('hidden', !ok);
  }});
}}
['filter-pre','filter-sel','filter-model','filter-status'].forEach(id => {{
  document.getElementById(id).addEventListener('change', applyFilters);
}});
</script>
</body>
</html>
"""
    destination.write_text(content, encoding="utf-8")
    return destination
