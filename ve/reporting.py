from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger


@dataclass
class RunContext:
    original_features: List[str]
    engineered_features: List[str]
    ev_selected_features: List[str]
    ice_selected_features: List[str]
    ev_results: Dict[str, dict]
    ice_results: Dict[str, dict]
    best_ev_model: str
    best_ice_model: str
    system_specs: Dict[str, str]


def save_core_artifacts(
    ctx: RunContext,
    ev_rankings: pd.DataFrame,
    ice_rankings: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving rankings and results artifacts to {}", output_dir)
    ev_rankings.to_csv(output_dir / "enhanced_ev_model_rankings.csv")
    ice_rankings.to_csv(output_dir / "enhanced_ice_model_rankings.csv")

    # Save selected features
    (output_dir / "ev_selected_features.json").write_text(
        json.dumps(ctx.ev_selected_features, indent=2)
    )
    (output_dir / "ice_selected_features.json").write_text(
        json.dumps(ctx.ice_selected_features, indent=2)
    )

    # Save summary JSON (similar shape to previous)
    perf = {
        "best_ev_r2": float(ev_rankings.iloc[0]["test_r2"]) if not ev_rankings.empty else None,
        "best_ice_r2": float(ice_rankings.iloc[0]["test_r2"]) if not ice_rankings.empty else None,
        "ev_features_count": len(ctx.ev_selected_features),
        "ice_features_count": len(ctx.ice_selected_features),
    }
    enhanced = {
        "original_features": ctx.original_features,
        "engineered_features": ctx.engineered_features,
        "ev_selected_features": ctx.ev_selected_features,
        "ice_selected_features": ctx.ice_selected_features,
        "ev_results": ctx.ev_results,
        "ice_results": ctx.ice_results,
        "best_ev_model": ctx.best_ev_model,
        "best_ice_model": ctx.best_ice_model,
        "system_specs": ctx.system_specs,
        "performance_summary": perf,
    }
    (output_dir / "enhanced_analysis_results.json").write_text(
        json.dumps(enhanced, indent=2)
    )


def write_final_report_markdown(output_dir: Path) -> None:
    """Reuse the existing scripts/generate_report.py by re-implementing its logic inline (stdlib only)."""
    ev_rank = output_dir / "enhanced_ev_model_rankings.csv"
    ice_rank = output_dir / "enhanced_ice_model_rankings.csv"
    res_json = output_dir / "enhanced_analysis_results.json"
    ev_params_json = output_dir / "ev_model_parameters.json"
    ice_params_json = output_dir / "ice_model_parameters.json"

    def read_csv(path: Path):
        return pd.read_csv(path) if path.exists() else pd.DataFrame()

    results = json.loads(res_json.read_text(encoding="utf-8")) if res_json.exists() else {}
    ev_df = read_csv(ev_rank)
    ice_df = read_csv(ice_rank)
    ev_params = json.loads(ev_params_json.read_text(encoding="utf-8")) if ev_params_json.exists() else {}
    ice_params = json.loads(ice_params_json.read_text(encoding="utf-8")) if ice_params_json.exists() else {}

    best_ev = results.get("best_ev_model", "")
    best_ice = results.get("best_ice_model", "")
    best_ev_r2 = results.get("performance_summary", {}).get("best_ev_r2", None)
    best_ice_r2 = results.get("performance_summary", {}).get("best_ice_r2", None)
    ev_results = results.get("ev_results", {})
    ice_results = results.get("ice_results", {})

    lines: list[str] = []
    lines.append("# Final Modeling Report\n")
    lines.append("## Best Models\n")
    if best_ev:
        if best_ev_r2 is not None:
            lines.append(f"- EV: {best_ev} (Test R²: {best_ev_r2:.4f})\n")
        else:
            lines.append(f"- EV: {best_ev}\n")
    if best_ice:
        if best_ice_r2 is not None:
            lines.append(f"- ICE: {best_ice} (Test R²: {best_ice_r2:.4f})\n")
        else:
            lines.append(f"- ICE: {best_ice}\n")

    if not ev_df.empty:
        lines.append("## EV Model Rankings (Top 5)\n")
        lines.append(ev_df.head(5).to_markdown(index=False))
        lines.append("")
    if not ice_df.empty:
        lines.append("## ICE Model Rankings (Top 5)\n")
        lines.append(ice_df.head(5).to_markdown(index=False))
        lines.append("")

    # Tuned vs Base snippet (if tuned model is best)
    def tuned_vs_base_snippet(label: str, best_name: str, res_dict: dict) -> list[str]:
        out: list[str] = []
        if isinstance(best_name, str) and best_name.startswith("Tuned "):
            base_name = best_name.replace("Tuned ", "", 1)
            tuned = res_dict.get(best_name, {}) or {}
            base = res_dict.get(base_name, {}) or {}
            try:
                t_mae = float(tuned.get("test_mae"))
                t_r2 = float(tuned.get("test_r2"))
                b_mae = float(base.get("test_mae"))
                b_r2 = float(base.get("test_r2"))
                d_mae = b_mae - t_mae  # positive is improvement
                d_r2 = t_r2 - b_r2     # positive is improvement
                out.append(f"## Tuned vs Base – {label}\n")
                out.append(f"- Base: {base_name} — MAE {b_mae:.2f}, R² {b_r2:.4f}")
                out.append(f"- Tuned: {best_name} — MAE {t_mae:.2f}, R² {t_r2:.4f}")
                out.append(f"- Improvement: ΔMAE {d_mae:+.2f} (lower is better), ΔR² {d_r2:+.4f}")
                out.append("")
            except Exception:
                pass
        return out

    lines += tuned_vs_base_snippet("EV", best_ev, ev_results)
    lines += tuned_vs_base_snippet("ICE", best_ice, ice_results)

    # Tuned params (best-only)
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

    specs = results.get("system_specs", {})
    if specs:
        lines.append("## System Specs\n")
        for k, v in specs.items():
            lines.append(f"- {k}: {v}")
        lines.append("")

    report_path = output_dir / "final_modeling_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written to {}", report_path)
