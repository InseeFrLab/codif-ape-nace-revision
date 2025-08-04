import logging

import pandas as pd

from constants.paths import URL_EXPLANATORY_NOTES, URL_MAPPING_TABLE
from mappings.mappings import get_mapping, get_nace2025_from_mapping
from utils.data import get_file_system, load_excel_from_fs

logger = logging.getLogger(__name__)


def create_content_vdb(df: pd.DataFrame) -> pd.DataFrame:
    """Generate content for each row in the DataFrame."""

    def generate_content(row):
        sections = [
            f"# {row.code} : {row.label}",
            f"## Explications des activités incluses dans la sous-classe\n{row.notes}" if row.get("notes") else None,
            f"## Liste d'exemples d'activités incluses dans la sous-classe\n{row.include}"
            if row.get("include")
            else None,
            f"## Liste d'exemples d'activités non incluses dans la sous-classe\n{row.not_include}"
            if row.get("not_include")
            else None,
        ]
        return "\n\n".join(filter(None, sections))

    df["content"] = df.apply(generate_content, axis=1)
    return df.fillna("")


def fetch_nace2025_labels(excluded_fields: list[str] | None = None) -> pd.DataFrame:
    """
    Fetch NACE 2025 metadata and return a list of NAF2025 objects.
    """
    fs = get_file_system()

    # Load data
    table_corres = load_excel_from_fs(fs, URL_MAPPING_TABLE)
    notes_ex = load_excel_from_fs(fs, URL_EXPLANATORY_NOTES)

    # Generate mapping and codes
    mapping = get_mapping(notes_ex, table_corres)

    nace2025_codes = get_nace2025_from_mapping(mapping)

    df_nace2025 = pd.DataFrame([code.model_dump(exclude=excluded_fields) for code in nace2025_codes])
    return create_content_vdb(df_nace2025)
