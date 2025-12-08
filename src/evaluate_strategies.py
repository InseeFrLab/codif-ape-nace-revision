# Not UP-TO-DATE
# Adapt URL_SIRENE4_AMBIGUOUS_RAG/URL_SIRENE4_AMBIGUOUS_CAG

import pandas as pd
from datetime import datetime
import pyarrow.parquet as pq
import os
os.chdir("codif-ape-nace-revision/src")

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
naf08_to_naf2025 = {m.code: [c.code for c in m.naf2025] for m in mapping}
ground_truth["mapping_ok"] = [
    check_mapping(naf08, naf25) for naf08, naf25 in zip(ground_truth["NAF2008_code"], ground_truth["apet_manual"])
]
ground_truth = ground_truth.loc[:, ["liasse_numero", "apet_manual", "mapping_ok"]]

eval_df = merged_df.merge(ground_truth, on="liasse_numero", how="inner")

accuracies_raw_old = {
    f"accuracy_{model.replace('nace2025_', '')}_lvl_{i}": round(
        (eval_df["apet_manual"].str[:i] == eval_df[f"{model}"].str[:i]).mean() * 100,
        2,
    )
    for i in [5, 4, 3, 2, 1]
    for model in [
        f"nace2025_{x}" for x in list(df_dict.keys()) + ["cascade_label", "voting_label", "weighted_voting_label"]
    ]
}

accuracies_codable_old = {
    f"accuracy_{model}_lvl_{i}": round(
        (
            eval_df[eval_df[f"codable_{model}"] == True]["apet_manual"].str[:i]
            == eval_df[eval_df[f"codable_{model}"] == True][f"nace2025_{model}"].str[:i]
        ).mean()
        * 100,
        2,
    )
    for i in [5, 4, 3, 2, 1]
    for model in df_dict.keys()
}

accuracies_raw_llm_old = {
    f"accuracy_{model.replace('nace2025_', '')}_lvl_{i}": round(
        (
            eval_df[eval_df["mapping_ok"]]["apet_manual"].str[:i] == eval_df[eval_df["mapping_ok"]][f"{model}"].str[:i]
        ).mean()
        * 100,
        2,
    )
    for i in [5, 4, 3, 2, 1]
    for model in [
        f"nace2025_{x}" for x in list(df_dict.keys()) + ["cascade_label", "voting_label", "weighted_voting_label"]
    ]
}

# ------------------------------------------------------------------------------------------------------------------------


LEVELS = [5, 4, 3, 2, 1]
ENSEMBLE_METHODS = ["cascade_label", "voting_label", "weighted_voting_label"]

from typing import Dict, List
def calculate_accuracy(
    predictions: pd.Series, 
    ground_truth: pd.Series, 
    level: int
) -> float:
    """Calcule l'accuracy à un niveau donné de précision."""
    return round(
        (ground_truth.str[:level] == predictions.str[:level]).mean() * 100, 
        2
    )


def generate_model_names(base_models: List[str], include_ensemble: bool = True) -> List[str]:
    """Génère les noms de modèles avec le préfixe nace2025."""
    models = [f"nace2025_{model}" for model in base_models]
    if include_ensemble:
        models.extend([f"nace2025_{method}" for method in ENSEMBLE_METHODS])
    return models


def compute_accuracies(
    eval_df: pd.DataFrame,
    models: List[str],
    levels: List[int],
    filter_condition: pd.Series = None,
    model_prefix: str = ""
) -> Dict[str, float]:
    """
    Calcule les accuracies pour plusieurs modèles et niveaux.
    
    Args:
        eval_df: DataFrame d'évaluation
        models: Liste des noms de modèles
        levels: Liste des niveaux à évaluer
        filter_condition: Condition de filtrage optionnelle (masque booléen)
        model_prefix: Préfixe à ajouter au nom du modèle dans la colonne
    """
    df_filtered = eval_df[filter_condition] if filter_condition is not None else eval_df
    
    accuracies = {}
    for model in models:
        model_col = f"{model_prefix}{model}" if model_prefix else model
        clean_name = model.replace('nace2025_', '')
        
        for level in levels:
            key = f"accuracy_{clean_name}_lvl_{level}"
            accuracies[key] = calculate_accuracy(
                predictions=df_filtered[model_col],
                ground_truth=df_filtered["apet_manual"],
                level=level
            )
    
    return accuracies


# Utilisation
base_models = list(df_dict.keys())

# Accuracies brutes (tous les cas)
accuracies_raw = compute_accuracies(
    eval_df=eval_df,
    models=generate_model_names(base_models, include_ensemble=True),
    levels=LEVELS
)


# Accuracies sur les cas codables
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

accuracies_codable == accuracies_codable_old

# Accuracies LLM (mapping_ok uniquement)
accuracies_raw_llm = compute_accuracies(
    eval_df=eval_df,
    models=generate_model_names(base_models, include_ensemble=True),
    levels=LEVELS,
    filter_condition=eval_df["mapping_ok"]
)

accuracies_raw_llm == accuracies_raw_llm_old






# ------------------------------------------------------------------------------------------------------------------------
stats = get_model_agreement_stats(eval_df, model_columns)

print(f"Raw accuracies : {accuracies_raw}\n\n")
print(f"Codable accuracies : {accuracies_codable}\n\n")
print(f"Raw LLM accuracies : {accuracies_raw_llm}\n\n")
print(f"---------------------------------\nSTATISTIQUES\n {stats}\n\n")

best_strategy = [
    key
    for key in accuracies_raw
    if key.endswith("lvl_5")
    and accuracies_raw[key] == max(accuracies_raw[k] for k in accuracies_raw if k.endswith("lvl_5"))
]

final_df = merged_df.loc[
    :,
    [
        "liasse_numero",
        f"nace2025_{best_strategy[0].replace('accuracy_', '').replace('_lvl_5', '')}",
    ],
].rename(columns={f"nace2025_{best_strategy[0].replace('accuracy_', '').replace('_lvl_5', '')}": "nace2025"})

timestamp = datetime.now().strftime("%Y%m%d")
final_df.to_parquet(f"{URL_SIRENE4_AMBIGUOUS_FINAL}/{timestamp}_sirene4_ambiguous.parquet", filesystem=fs)
