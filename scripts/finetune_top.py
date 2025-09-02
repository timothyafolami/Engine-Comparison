#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from ve.data import load_and_prepare_data
from ve.features import engineer_features, select_features, ORIGINAL_FEATURES
from ve.tuning import finetune_top_two


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fine-tune top two models per cohort with Optuna/Sklearn")
    p.add_argument("--data-path", default="data/vehicle_comparison_dataset_030417.csv", help="Path to input CSV")
    p.add_argument("--output-dir", default="output", help="Directory to write tuned models")
    p.add_argument("--vehicle", choices=["ev", "ice", "both"], default="both", help="Which cohort(s) to tune")
    p.add_argument("--rank-by", choices=["r2", "mae"], default="r2", help="Metric to pick top-2")
    p.add_argument("--tuning-metric", choices=["r2", "mae"], default="mae", help="Metric to optimize during tuning")
    p.add_argument("--search", choices=["random", "grid"], default="random", help="Sklearn search method for non-boosting models")
    p.add_argument("--random-iter", type=int, default=50, help="RandomizedSearchCV iterations")
    p.add_argument("--optuna-trials", type=int, default=50, help="Optuna trials for boosting models")
    return p


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_and_prepare_data(args.data_path)

    def run_one(tag: str, df: pd.DataFrame):
        eng, _ = engineer_features(df, "ELECTRIC VEHICLES" if tag == "ev" else "ICE VEHICLES")
        feats = select_features(eng, "ELECTRIC VEHICLES" if tag == "ev" else "ICE VEHICLES")
        tuned = finetune_top_two(
            eng,
            feats,
            rank_by=args.rank_by,
            tuning_metric=args.tuning_metric,
            search_method=args.search,
            random_iter=args.random_iter,
            optuna_trials=args.optuna_trials,
        )
        # Save models and metrics
        for tm in tuned:
            tag_name = f"{tag}_{tm.name.replace(' ', '_').lower()}"
            joblib.dump(tm.estimator, out_dir / f"tuned_{tag_name}.joblib")
            (out_dir / f"tuned_{tag_name}_metrics.json").write_text(json.dumps(tm.metrics, indent=2))
            (out_dir / f"tuned_{tag_name}_params.json").write_text(json.dumps(tm.best_params, indent=2))
        # Summary
        summary = [
            {
                "model": tm.name,
                "test_mae": tm.metrics["test_mae"],
                "test_rmse": tm.metrics["test_rmse"],
                "test_r2": tm.metrics["test_r2"],
            }
            for tm in tuned
        ]
        (out_dir / f"tuned_{tag}_top2_summary.json").write_text(json.dumps(summary, indent=2))

    if args.vehicle in ("ev", "both"):
        run_one("ev", ds.ev)
    if args.vehicle in ("ice", "both"):
        run_one("ice", ds.ice)

    print("Fine-tuning complete. See:", out_dir)


if __name__ == "__main__":
    main()

