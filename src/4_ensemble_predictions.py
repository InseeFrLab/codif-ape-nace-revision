"""
Evaluate and combine LLM predictions for ambiguous NAF codes.

This script:
  1. Fetches the MLflow runs listed in --run_ids and reads their
     `output_path` param to locate each model's predictions parquet.
  2. Loads each parquet and aligns predictions across models on
     `liasse_numero`.
  3. Combines them with the majority-voting ensemble strategy.
  4. In eval mode: compares individual and ensemble predictions against
     the manual ground truth and writes a Markdown report.
  5. In prod mode (--export): writes the voting predictions to S3.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Tuple

import mlflow
import pandas as pd
import pyarrow.parquet as pq

import config
from constants.paths import URL_GROUND_TRUTH, URL_SIRENE4_AMBIGUOUS_FINAL
from utils.data import fetch_mapping, get_file_system, merge_dataframes
from utils.ensemble_report import build_ensemble_report
from utils.strategies import (
    compute_accuracies,
    generate_model_names,
    get_model_agreement_stats,
    select_labels_voting,
)

config.setup()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


VAR_TO_KEEP = ["liasse_numero", "nace2025", "codable"]
LEVELS = [5, 4, 3, 2, 1]
ENSEMBLE_METHODS = ["voting_label"]


# ============================================================================
# Pipeline steps
# ============================================================================

def fetch_models_from_mlflow(run_ids: List[str]) -> Dict[str, Dict]:
    """Resolve each run_id to its LLM name and predictions parquet path.

    Two runs of the same `LLM_MODEL` (e.g. qwen3 with and without thinking)
    are disambiguated by appending a `-thinking` suffix to the key.
    """
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

    models: Dict[str, Dict] = {}
    for run_id in run_ids:
        run = mlflow.get_run(run_id)
        params = run.data.params
        thinking = str(params.get("THINKING", "False")).lower() == "true"
        llm_name = params.get("LLM_MODEL", run_id)
        key = f"{llm_name}-thinking" if thinking else llm_name
        if key in models:
            raise ValueError(
                f"Duplicate model key '{key}' across runs "
                f"({models[key]['run_id']} and {run_id}). "
                "Two runs share the same LLM_MODEL/THINKING combination."
            )
        models[key] = {
            "run_id": run_id,
            "path": params["output_path"],
            "llm_model": llm_name,
            "thinking": thinking,
        }
        logger.info("Resolved run %s -> %s (%s)", run_id, key, params["output_path"])
    return models


def load_predictions(models: Dict[str, Dict], fs) -> Dict[str, pd.DataFrame]:
    """Load each model's parquet and keep only the `liasse_numero` shared by all."""
    dfs = {
        name: pq.ParquetDataset(cfg["path"].replace("s3://", ""), filesystem=fs)
        .read()
        .to_pandas()
        for name, cfg in models.items()
    }
    shared_ids = set.intersection(*(set(df["liasse_numero"]) for df in dfs.values()))
    return {
        name: df.loc[df["liasse_numero"].isin(shared_ids)]
        .drop_duplicates(subset=["liasse_numero"])
        for name, df in dfs.items()
    }


def apply_ensemble_strategies(
    merged_df: pd.DataFrame, models: Dict[str, Dict]
) -> Tuple[pd.DataFrame, List[str]]:
    """Add the majority-voting column to `merged_df`."""
    model_columns = [f"nace2025_{name}" for name in models]
    merged_df["nace2025_voting_label"] = select_labels_voting(merged_df, model_columns)
    return merged_df, model_columns


def load_ground_truth(fs) -> pd.DataFrame:
    """Load manual annotations and flag rows where NAF2008->NAF2025 mapping is valid."""
    gt = (
        pq.ParquetDataset(URL_GROUND_TRUTH.replace("s3://", ""), filesystem=fs)
        .read()
        .to_pandas()
        .drop_duplicates(subset="liasse_numero")
    )

    mapping = fetch_mapping()
    naf08_to_naf2025 = {
        m.code.replace(".", ""): {c.code.replace(".", "") for c in m.naf2025}
        for m in mapping
    }
    gt["mapping_ok"] = [
        naf25 in naf08_to_naf2025.get(naf08, set())
        for naf08, naf25 in zip(gt["NAF2008_code"], gt["apet_manual"])
    ]
    return gt[["liasse_numero", "apet_manual", "mapping_ok"]]


def compute_all_accuracies(
    eval_df: pd.DataFrame, base_models: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute the three accuracy views: raw, codable-only, mapping_ok-only."""
    full_models = generate_model_names(base_models, ENSEMBLE_METHODS, include_ensemble=True)

    raw = compute_accuracies(eval_df=eval_df, models=full_models, levels=LEVELS)

    codable: Dict[str, float] = {}
    for model in base_models:
        codable.update(
            compute_accuracies(
                eval_df=eval_df,
                models=[model],
                levels=LEVELS,
                filter_condition=eval_df[f"codable_{model}"] == True,  # noqa: E712
                model_prefix="nace2025_",
            )
        )

    mapping_ok = compute_accuracies(
        eval_df=eval_df,
        models=full_models,
        levels=LEVELS,
        filter_condition=eval_df["mapping_ok"],
    )

    return {"raw": raw, "codable": codable, "mapping_ok": mapping_ok}


def export_final_predictions(merged_df: pd.DataFrame, fs) -> str:
    """Write the majority-voting predictions to S3 and return the output path."""
    final_df = merged_df[["liasse_numero", "nace2025_voting_label"]].rename(
        columns={"nace2025_voting_label": "nace2025"}
    )
    timestamp = datetime.now().strftime("%Y%m%d")
    output_path = f"{URL_SIRENE4_AMBIGUOUS_FINAL}{timestamp}_sirene4_ambiguous.parquet"
    final_df.to_parquet(output_path, filesystem=fs)
    logger.info("Final results exported to %s", output_path)
    return output_path


# ============================================================================
# Main
# ============================================================================

def write_report(report_md: str) -> str:
    """Write the Markdown report next to the script and return its local path."""
    reports_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(reports_dir, f"ensemble_report_{timestamp}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_md)
    logger.info("Ensemble report written to %s", path)
    return path


def main(run_ids: List[str], mode: str = "eval", export: bool = False) -> None:
    fs = get_file_system()

    logger.info("===== STEP 4: ensemble predictions (mode=%s) =====", mode)
    logger.info("Fetching %d run(s) from MLflow", len(run_ids))
    models = fetch_models_from_mlflow(run_ids)

    # Each model's predictions directory is logged here as an explicit input.
    for name, cfg in models.items():
        logger.info("INPUT  : %s (%s)", cfg["path"], name)
    if mode == "eval":
        logger.info("INPUT  : %s (ground truth)", URL_GROUND_TRUTH)
    logger.info(
        "OUTPUT : %s",
        f"{URL_SIRENE4_AMBIGUOUS_FINAL} (final predictions)" if export else "report only (no export)",
    )

    logger.info("Loading predictions from %d model(s)", len(models))
    dfs = load_predictions(models, fs)

    merged_df = merge_dataframes(
        dfs,
        merge_on="liasse_numero",
        var_to_keep=VAR_TO_KEEP,
        columns_to_rename={"nace2025": "nace2025_{key}", "codable": "codable_{key}"},
    )
    merged_df, model_columns = apply_ensemble_strategies(merged_df, models)

    if export:
        export_final_predictions(merged_df, fs)

    if mode == "eval":
        logger.info("Loading ground truth")
        eval_df = merged_df.merge(load_ground_truth(fs), on="liasse_numero", how="inner")

        logger.info("Computing accuracies")
        accuracies = compute_all_accuracies(eval_df, list(models))
        agreement = get_model_agreement_stats(eval_df, model_columns)

        logger.info("Raw accuracies: %s", accuracies["raw"])
        logger.info("Codable accuracies: %s", accuracies["codable"])
        logger.info("Mapping-ok accuracies: %s", accuracies["mapping_ok"])
        logger.info("Model agreement statistics: %s", agreement)

        report_md = build_ensemble_report(
            models=models,
            accuracies=accuracies,
            agreement=agreement,
            levels=LEVELS,
            ensemble_methods=ENSEMBLE_METHODS,
            eval_size=len(eval_df),
            final_output_path=None,
            export_final=export,
        )
        write_report(report_md)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_ids",
        type=str,
        required=True,
        help="Comma-separated MLflow run IDs to combine.",
    )
    parser.add_argument(
        "--mode",
        choices=["prod", "eval"],
        default="eval",
        help="eval: compute accuracy metrics and write a report. prod: export predictions only.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Write the majority-voting predictions to S3.",
    )
    args = parser.parse_args()

    run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
    if not run_ids:
        parser.error("--run_ids must contain at least one run ID")

    main(run_ids=run_ids, mode=args.mode, export=args.export)
