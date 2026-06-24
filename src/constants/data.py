# =============================================================================
# Input schema — source SIRENE 4 column names.
# Centralised here so business logic never hardcodes column names.
# =============================================================================

# Unique row identifier.
ID_VAR = "liasse_numero"

# NAF 2008 (APE) code to recode.
NACE08_VAR = "apet2008"

# Free-text activity description (primary signal for the LLM).
ACTIVITY_LABEL_VAR = "libelle"

# Extra columns appended to the activity description, each prefixed by a French
# label in the prompt. Insertion order is preserved. Every key here is included
# in VAR_TO_KEEP below, so these columns are always loaded from the source file.
ACTIVITY_PRECISION_VARS = {
    "activ_sec_agri_et": "Précisions sur l'activité agricole",
    "activ_nat_lib_et": "Autre nature d'activité",
    "cj_libelle": "Catégorie juridique de l'établissement",
    # "activ_surf_et_libelle" : "Surface commerciale"
}

# All columns selected from the source file by the ambiguous-data loader.
# Splatting ACTIVITY_PRECISION_VARS guarantees every precision column is loaded
# (so none stays silently inert); the remaining names are other context columns.
VAR_TO_KEEP = [
    ID_VAR,
    NACE08_VAR,
    ACTIVITY_LABEL_VAR,
    *ACTIVITY_PRECISION_VARS,
    #"evenement_type",
    #"cj",
    "activ_nat_et",
    "liasse_type",
    "activ_surf_et",
    "activ_nat_lib_et",
    "activ_perm_et",
]
