# Not UP-TO-DATE
# Adapt URL_SIRENE4_AMBIGUOUS_RAG/URL_SIRENE4_AMBIGUOUS_CAG

import pandas as pd
from datetime import datetime
import pyarrow.parquet as pq
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#os.chdir("codif-ape-nace-revision/src")

from constants.paths import (
    URL_EXPLANATORY_NOTES,
    URL_GROUND_TRUTH,
    URL_MAPPING_TABLE,
    URL_SIRENE4_AMBIGUOUS_CAG,
    URL_SIRENE4_AMBIGUOUS_FINAL,
)
from constants.data import VAR_TO_KEEP
from mappings.mappings import get_mapping

from utils.data import (
    get_file_system,
    merge_dataframes,
    fetch_mapping
)
from utils.strategies import (
    compute_accuracies,
    generate_model_names,
    compute_accuracies,
    get_model_agreement_stats,
    select_labels_cascade,
    select_labels_voting,
    select_labels_weighted_voting,
)

def check_mapping(naf08, naf25):
    return naf25 in naf08_to_naf2025.get(naf08, set())

fs = get_file_system()

VAR_TO_KEEP = ["liasse_numero", "nace2025", "codable"]

MODEL_TO_USE = {
    "Mistral-Small-3.2-24B-Instruct-2506": {
        "weight": 1,
        "cascade_order": 1,
        "path": "s3://projet-ape/NAF-revision/relabeled-data-cag/mistralai/Mistral-Small-3.2-24B-Instruct-2506/part-0---2025-11-28--21:30.parquet",
    },
    "Qwen3-32B": {
        "weight": 1,
        "cascade_order": 3,
        "path": "s3://projet-ape/NAF-revision/relabeled-data-cag/Qwen/Qwen3-32B/part-0---2025-11-28--14:59.parquet",
    },
    "DeepSeek-R1-Distill-Qwen-32B": {
        "weight": 1,
        "cascade_order": 2,
        "path": "s3://projet-ape/NAF-revision/relabeled-data-cag/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B/part-0---2025-11-28--19:08.parquet",
    },
}

df_dict = {}
for llm_name, llm_values in MODEL_TO_USE.items():
    df_dict[llm_name] = pq.ParquetDataset(
        llm_values["path"].replace('s3://', ''),
        filesystem=fs,
    ).read().to_pandas()


# Annotations present in all models predictions + dedup
list_id = set.intersection(*map(set, [df_dict[llm_name]["liasse_numero"].tolist() for llm_name in MODEL_TO_USE.keys()]))
df_dict = {
    llm_name: df_dict[llm_name]
    .loc[df_dict[llm_name]["liasse_numero"].isin(list_id)]
    .drop_duplicates(subset=["liasse_numero"])
    for llm_name in MODEL_TO_USE.keys()
}

# row_counts = {llm_name: len(df_dict[llm_name]) for llm_name in df_dict.keys()}

merged_df = merge_dataframes(
    df_dict,
    merge_on="liasse_numero",
    var_to_keep=VAR_TO_KEEP,
    columns_to_rename={"nace2025": "nace2025_{key}", "codable": "codable_{key}"},
)

# Order models for cascade method
model_columns = [f"nace2025_{model}" for model in sorted(MODEL_TO_USE.keys(), key=lambda x: MODEL_TO_USE[x]["cascade_order"])]
weights = {f"nace2025_{model}": MODEL_TO_USE[model]["weight"] for model in df_dict.keys()}

merged_df["nace2025_cascade_label"] = select_labels_cascade(merged_df, model_columns)
merged_df["nace2025_voting_label"] = select_labels_voting(merged_df, model_columns)
merged_df["nace2025_weighted_voting_label"] = select_labels_weighted_voting(merged_df, model_columns, weights)



## Fetch annotated values on model predictions -------------------------------

ground_truth = pq.ParquetDataset(URL_GROUND_TRUTH.replace("s3://", ""), filesystem=fs).read().to_pandas()
# TODO: TEMP REMOVE DUPLICATED
ground_truth = ground_truth.drop_duplicates(subset="liasse_numero")

mapping = fetch_mapping()
naf08_to_naf2025 = {m.code.replace('.', ''): [c.code.replace('.', '') for c in m.naf2025] for m in mapping}
ground_truth["mapping_ok"] = [
    check_mapping(naf08, naf25) for naf08, naf25 in zip(ground_truth["NAF2008_code"], ground_truth["apet_manual"])
]
ground_truth = ground_truth.loc[:, ["liasse_numero", "apet_manual", "mapping_ok"]]

eval_df = merged_df.merge(ground_truth, on="liasse_numero", how="inner")


## Get accuracies -------------------------------

LEVELS = [5, 4, 3, 2, 1]
ENSEMBLE_METHODS = ["cascade_label", "voting_label", "weighted_voting_label"]

base_models = list(df_dict.keys())

# Raw accuracies
accuracies_raw = compute_accuracies(
    eval_df=eval_df,
    models=generate_model_names(
        base_models, 
        ENSEMBLE_METHODS, 
        include_ensemble=True),
    levels=LEVELS
)

# Accuracies only on codables
accuracies_codable = {}
for model in base_models:
    codable_mask = eval_df[f"codable_{model}"] == True
    model_accuracies = compute_accuracies(
        eval_df=eval_df,
        models=[model],
        levels=LEVELS,
        filter_condition=codable_mask,
        model_prefix="nace2025_"
    )
    accuracies_codable.update(model_accuracies)


# Accuracies LLM (mapping_ok only)
accuracies_raw_llm = compute_accuracies(
    eval_df=eval_df,
    models=generate_model_names(
        base_models, 
        ENSEMBLE_METHODS, 
        include_ensemble=True
    ),
    levels=LEVELS,
    filter_condition=eval_df["mapping_ok"]
)

stats = get_model_agreement_stats(eval_df, model_columns)

logger.info("Raw accuracies: %s", accuracies_raw)
logger.info("Codable accuracies: %s", accuracies_codable)
logger.info("Raw LLM accuracies: %s", accuracies_raw_llm)
logger.info("Model agreement statistics: %s", stats)

## Choice of best strategy and export final results ------------------------

lvl5_scores = {k: v for k, v in accuracies_raw.items() if k.endswith("lvl_5")}
max_score = max(lvl5_scores.values())
best_strategy = [k for k, v in lvl5_scores.items() if v == max_score][0]
best_strategy = best_strategy.replace('accuracy_', '').replace('_lvl_5', '')

final_df = merged_df.loc[
    :,
    [
        "liasse_numero",
        f"nace2025_{best_strategy}",
    ],
].rename(columns={f"nace2025_{best_strategy}": "nace2025"})

timestamp = datetime.now().strftime("%Y%m%d")
output_path = f"{URL_SIRENE4_AMBIGUOUS_FINAL}{timestamp}_sirene4_ambiguous.parquet"
# final_df.to_parquet(output_path, filesystem=fs)
logger.info(f"Final results exported successfully here: {output_path}")
