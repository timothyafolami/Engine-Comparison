#!/usr/bin/env python3
"""
Consolidated report generator that uses only Python stdlib to avoid env issues.
Reads existing outputs in ./output and writes ./output/final_modeling_report.md
"""
from __future__ import annotations

import csv
import json
from pathlib import Path


def read_csv_as_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return [], []
    header = rows[0]
    data = rows[1:]
    # Normalize header: first column may be empty (model name)
    if header and (header[0] is None or header[0].strip() == ""):
        header[0] = "model"
    return header, data


def markdown_table(header: list[str], rows: list[list[str]], max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    # Align columns
    out = []
    out.append("| " + " | ".join(header) + " |")
    out.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        # pad row to header length
        r = list(r) + [""] * max(0, len(header) - len(r))
        out.append("| " + " | ".join(r[: len(header)]) + " |")
    return "\n".join(out)


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "output"
    ev_rank = out_dir / "enhanced_ev_model_rankings.csv"
    ice_rank = out_dir / "enhanced_ice_model_rankings.csv"
    res_json = out_dir / "enhanced_analysis_results.json"
    ev_params_json = out_dir / "ev_model_parameters.json"
    ice_params_json = out_dir / "ice_model_parameters.json"

    # Load core data
    results = json.loads(res_json.read_text(encoding="utf-8"))
    ev_header, ev_rows = read_csv_as_rows(ev_rank)
    ice_header, ice_rows = read_csv_as_rows(ice_rank)
    ev_params = json.loads(ev_params_json.read_text(encoding="utf-8"))
    ice_params = json.loads(ice_params_json.read_text(encoding="utf-8"))

    best_ev = results.get("best_ev_model", "")
    best_ice = results.get("best_ice_model", "")
    best_ev_r2 = results.get("performance_summary", {}).get("best_ev_r2", None)
    best_ice_r2 = results.get("performance_summary", {}).get("best_ice_r2", None)

    lines: list[str] = []
    lines.append("# Final Modeling Report\n")

    lines.append("## Best Models\n")
    if best_ev_r2 is not None:
        lines.append(f"- EV: {best_ev} (Test R²: {best_ev_r2:.4f})\n")
    else:
        lines.append(f"- EV: {best_ev}\n")
    if best_ice_r2 is not None:
        lines.append(f"- ICE: {best_ice} (Test R²: {best_ice_r2:.4f})\n")
    else:
        lines.append(f"- ICE: {best_ice}\n")

    lines.append("## EV Model Rankings (Top 5)\n")
    lines.append(markdown_table(ev_header, ev_rows, max_rows=5))
    lines.append("")

    lines.append("## ICE Model Rankings (Top 5)\n")
    lines.append(markdown_table(ice_header, ice_rows, max_rows=5))
    lines.append("")

    # Tuned params (best models only for brevity)
    lines.append("## Tuned Hyperparameters (Selected)\n")
    if best_ev and best_ev in ev_params:
        lines.append(f"### EV – {best_ev}\n")
        lines.append("```json")
        lines.append(json.dumps(ev_params[best_ev], indent=2))
        lines.append("```")
    if best_ice and best_ice in ice_params:
        lines.append(f"### ICE – {best_ice}\n")
        lines.append("```json")
        lines.append(json.dumps(ice_params[best_ice], indent=2))
        lines.append("```")

    # Feature importances / coefficients if present
    ev_imp = out_dir / "best_ev_model_importances.csv"
    ev_coef = out_dir / "best_ev_model_coefficients.csv"
    ice_imp = out_dir / "best_ice_model_importances.csv"
    ice_coef = out_dir / "best_ice_model_coefficients.csv"

    def section_csv(title: str, path: Path, max_rows: int = 15) -> None:
        if path.exists():
            h, rows = read_csv_as_rows(path)
            lines.append(f"## {title}\n")
            lines.append(markdown_table(h, rows, max_rows=max_rows))
            lines.append("")

    section_csv("Best EV Model – Feature Importances", ev_imp)
    section_csv("Best EV Model – Coefficients", ev_coef)
    section_csv("Best ICE Model – Feature Importances", ice_imp)
    section_csv("Best ICE Model – Coefficients", ice_coef)

    # System specs
    specs = results.get("system_specs", {})
    if specs:
        lines.append("## System Specs\n")
        for k, v in specs.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    out_path = out_dir / "final_modeling_report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()


