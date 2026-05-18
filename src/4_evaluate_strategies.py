"""
Evaluate and combine LLM predictions for ambiguous NAF codes.

For each LLM listed in `MODELS`, this script:
  1. Loads the parquet of predictions produced by `3_encode_ambiguous.py`.
  2. Aligns predictions across models on `liasse_numero`.
  3. Combines them with three ensemble strategies
     (cascade, voting, weighted voting).
  4. Compares individual and ensemble predictions against the manual
     ground truth, both raw and filtered (codable, mapping_ok).
  5. Picks the strategy with the highest level-5 accuracy and exports
     its predictions as the final NAF2025 file.

------------------------------------------------------------------
Open questions before running (search for "TODO" in this file):
  - S3 path of the new gemma4-26b-moe (normal + thinking) parquets
  - S3 path of the new qwen3-6-35b-moe parquet
  - Cascade order to assign to the new models (model priority)
  - Vote weights to assign to the new models
  - Whether to keep the 3 legacy models or replace them
  - Whether to actually write the final parquet (toggle EXPORT_FINAL)
------------------------------------------------------------------
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import pyarrow.parquet as pq

from constants.paths import URL_GROUND_TRUTH, URL_SIRENE4_AMBIGUOUS_FINAL
from utils.data import fetch_mapping, get_file_system, merge_dataframes
from utils.strategies import (
    compute_accuracies,
    generate_model_names,
    get_model_agreement_stats,
    select_labels_cascade,
    select_labels_voting,
    select_labels_weighted_voting,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

VAR_TO_KEEP = ["liasse_numero", "nace2025", "codable"]
LEVELS = [5, 4, 3, 2, 1]
ENSEMBLE_METHODS = ["cascade_label", "voting_label", "weighted_voting_label"]

# Toggle to True once accuracies have been reviewed and the final
# parquet should be written to S3.
EXPORT_FINAL = False

# Each entry describes one LLM run produced by `3_encode_ambiguous.py`.
#   - path          : S3 path of the parquet of predictions.
#   - weight        : vote weight for the weighted-voting ensemble.
#   - cascade_order : model priority for the cascade strategy (1 = first).
#
# Use distinct keys when the same base model runs in different modes
# (e.g. with vs without thinking) so that prediction columns do not collide.
MODELS: Dict[str, Dict] = {
    # ---- Legacy models -----------------------------------------------------
    # TODO: confirm whether these three models should be kept alongside the
    # new ones, or replaced. Drop the entries that are no longer needed.
    "Mistral-Small-3.2-24B-Instruct-2506": {
        "weight": 1,
        "cascade_order": 1,
        "path": "s3://projet-ape/NAF-revision/relabeled-data-cag/mistralai/Mistral-Small-3.2-24B-Instruct-2506/part-0---2025-11-28--21:30.parquet",
    },
    "DeepSeek-R1-Distill-Qwen-32B": {
        "weight": 1,
        "cascade_order": 2,
        "path": "s3://projet-ape/NAF-revision/relabeled-data-cag/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/part-0---2025-11-28--19:08.parquet",
    },
    "Qwen3-32B": {
        "weight": 1,
        "cascade_order": 3,
        "path": "s3://projet-ape/NAF-revision/relabeled-data-cag/Qwen/Qwen3-32B/part-0---2025-11-28--14:59.parquet",
    },
    # ---- New models --------------------------------------------------------
    # TODO: fill in the parquet paths once `3_encode_ambiguous.py` has been
    # rerun with these new LLMs, and set `weight` / `cascade_order` after
    # reviewing individual-model accuracies.
    "gemma4-26b-moe": {
        "weight": 1,                # TODO: weight to confirm
        "cascade_order": 4,         # TODO: cascade order to confirm
        "path": "TODO_S3_PATH",     # TODO: fill with the actual parquet path
    },
    "gemma4-26b-moe-thinking": {
        "weight": 1,                # TODO: weight to confirm
        "cascade_order": 5,         # TODO: cascade order to confirm
        "path": "TODO_S3_PATH",     # TODO: fill with the actual parquet path
    },
    "qwen3-6-35b-moe": {
        "weight": 1,                # TODO: weight to confirm
        "cascade_order": 6,         # TODO: cascade order to confirm
        "path": "TODO_S3_PATH",     # TODO: fill with the actual parquet path
    },
}


# ============================================================================
# Pipeline steps
# ============================================================================

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
    """Add cascade / voting / weighted-voting columns to `merged_df`."""
    model_columns = [
        f"nace2025_{name}"
        for name in sorted(models, key=lambda n: models[n]["cascade_order"])
    ]
    weights = {f"nace2025_{name}": cfg["weight"] for name, cfg in models.items()}

    merged_df["nace2025_cascade_label"] = select_labels_cascade(merged_df, model_columns)
    merged_df["nace2025_voting_label"] = select_labels_voting(merged_df, model_columns)
    merged_df["nace2025_weighted_voting_label"] = select_labels_weighted_voting(
        merged_df, model_columns, weights
    )
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


def pick_best_strategy(raw_accuracies: Dict[str, float]) -> str:
    """Return the model/strategy name with the highest level-5 accuracy."""
    lvl5 = {k: v for k, v in raw_accuracies.items() if k.endswith("lvl_5")}
    best = max(lvl5, key=lvl5.get)
    return best.replace("accuracy_", "").replace("_lvl_5", "")


def export_final_predictions(
    merged_df: pd.DataFrame, best_strategy: str, fs
) -> str:
    """Write the predictions of the best strategy to S3 and return the output path."""
    final_df = merged_df[["liasse_numero", f"nace2025_{best_strategy}"]].rename(
        columns={f"nace2025_{best_strategy}": "nace2025"}
    )
    timestamp = datetime.now().strftime("%Y%m%d")
    output_path = f"{URL_SIRENE4_AMBIGUOUS_FINAL}{timestamp}_sirene4_ambiguous.parquet"

    if EXPORT_FINAL:
        final_df.to_parquet(output_path, filesystem=fs)
        logger.info("Final results exported to %s", output_path)
    else:
        logger.info(
            "Export skipped (EXPORT_FINAL=False). Target path would be: %s",
            output_path,
        )
    return output_path


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    fs = get_file_system()

    logger.info("Loading predictions from %d model(s)", len(MODELS))
    dfs = load_predictions(MODELS, fs)

    merged_df = merge_dataframes(
        dfs,
        merge_on="liasse_numero",
        var_to_keep=VAR_TO_KEEP,
        columns_to_rename={"nace2025": "nace2025_{key}", "codable": "codable_{key}"},
    )
    merged_df, model_columns = apply_ensemble_strategies(merged_df, MODELS)

    logger.info("Loading ground truth")
    eval_df = merged_df.merge(load_ground_truth(fs), on="liasse_numero", how="inner")

    logger.info("Computing accuracies")
    accuracies = compute_all_accuracies(eval_df, list(MODELS))
    agreement = get_model_agreement_stats(eval_df, model_columns)

    logger.info("Raw accuracies: %s", accuracies["raw"])
    logger.info("Codable accuracies: %s", accuracies["codable"])
    logger.info("Mapping-ok accuracies: %s", accuracies["mapping_ok"])
    logger.info("Model agreement statistics: %s", agreement)

    best_strategy = pick_best_strategy(accuracies["raw"])
    logger.info("Best strategy at level 5: %s", best_strategy)

    export_final_predictions(merged_df, best_strategy, fs)


if __name__ == "__main__":
    main()
