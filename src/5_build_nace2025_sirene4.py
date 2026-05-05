# Not UP-TO-DATE

import pandas as pd

from src.constants.paths import (
    URL_GROUND_TRUTH,
    URL_SIRENE4_AMBIGUOUS_FINAL,
    URL_SIRENE4_EXTRACTION,
    URL_SIRENE4_NACE2025,
    URL_SIRENE4_UNIVOCAL,
)
from src.utils.cache_models import get_file_system

fs = get_file_system()

VAR_TO_KEEP = [
    "liasse_numero",
    "libelle",
    "evenement_type",
    "cj",
    "activ_nat_et",
    "liasse_type",
    "activ_surf_et",
    "activ_sec_agri_et",
    "activ_nat_lib_et",
    "activ_perm_et",
]

# Pour les univoques, on enleve les doublons de liasse seulement
data_univocal = pd.read_parquet(URL_SIRENE4_UNIVOCAL, filesystem=fs)
data_univocal = data_univocal.drop_duplicates(subset="liasse_numero")

# Pour les multivocaux issue de l'annotation humaine on enleve les doublons de liasse et on renomme la colonne apet_manual en nace2025
data_ambiguous_ground_truth = (
    pd.read_parquet(URL_GROUND_TRUTH, filesystem=fs)
    .rename(columns={"apet_manual": "nace2025"})
    .loc[:, ["liasse_numero", "nace2025"]]
)
data_ambiguous_ground_truth = data_ambiguous_ground_truth.drop_duplicates(subset="liasse_numero")

# Pour les multivocaux du LLM on enlève les données qui sont dans l'annotation humaine
data_ambiguous = pd.read_parquet(URL_SIRENE4_AMBIGUOUS_FINAL, filesystem=fs)
data_ambiguous = data_ambiguous.loc[
    ~data_ambiguous["liasse_numero"].isin(data_ambiguous_ground_truth["liasse_numero"].tolist())
]

# Pour les données de sirene 4, on enlève les doublons de liasse
data_sirene4 = pd.read_parquet(URL_SIRENE4_EXTRACTION, filesystem=fs).loc[:, VAR_TO_KEEP]
data_sirene4 = data_sirene4.drop_duplicates(subset="liasse_numero")

# On rajoute les variables annexes aux multivoques, univoques et ground truth contenant l'annotation nace2025
data_univocal = data_univocal.merge(data_sirene4, on="liasse_numero", how="left")
data_ambiguous = data_ambiguous.merge(data_sirene4, on="liasse_numero", how="left")
data_ambiguous_ground_truth = data_ambiguous_ground_truth.merge(data_sirene4, on="liasse_numero", how="left")

# few lines are still duplicated, remove them before merge. Old Label Studio pipeline was not 100% perfect
data_ambiguous_ground_truth = data_ambiguous_ground_truth.drop_duplicates(
    subset=[v for v in VAR_TO_KEEP if v != "liasse_numero"]
)

# On reconstruit les données multivoques en réinjectant les code nace 2025 pour les doublons
data_sirene4_multivoque = data_sirene4.loc[
    ~data_sirene4["liasse_numero"].isin(data_univocal["liasse_numero"].tolist()), VAR_TO_KEEP
]
data_ambiguous_resampled = (
    data_ambiguous.merge(data_sirene4_multivoque, on=[v for v in VAR_TO_KEEP if v != "liasse_numero"], how="left")
    .rename(columns={"liasse_numero_y": "liasse_numero"})
    .drop(columns=["liasse_numero_x"])
)
data_ambiguous_ground_truth_resampled = (
    data_ambiguous_ground_truth.merge(
        data_sirene4_multivoque, on=[v for v in VAR_TO_KEEP if v != "liasse_numero"], how="left"
    )
    .rename(columns={"liasse_numero_y": "liasse_numero"})
    .drop(columns=["liasse_numero_x"])
)
# Certains univoques sont les dans les multivoques, on les enlève
data_ambiguous_ground_truth_resampled.dropna(subset=["liasse_numero"], inplace=True)

data_sirene4_nace2025 = pd.concat(
    [data_univocal, data_ambiguous_resampled, data_ambiguous_ground_truth_resampled], axis=0
)

assert data_sirene4_nace2025.duplicated(subset="liasse_numero").sum() == 0

data_sirene4_nace2025.to_parquet(URL_SIRENE4_NACE2025, filesystem=fs)
