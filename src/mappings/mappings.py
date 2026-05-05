import unicodedata
from typing import Dict, List

import pandas as pd

from .models import NAF2008, NAF2025


def format_mapping_table(mapping_table: pd.DataFrame) -> pd.DataFrame:
    """
    Select and rename columns from the mapping table
    """
    columns_mapping = {
        "NAFold-code\n(code niveau sous-classe de la nomenclature actuelle)": "naf08_niv5",
        "NACEold-code\n(niveau classe)": "naf08_niv4",
        "NAFold-intitulé\n(niveau sous-classe)": "lib_naf08_niv5",
        "NACEnew-code\n(niveau classe)": "naf25_niv4",
        "NAFnew-code\n(code niveau sous-classe de la nomenclature 2025, correspondance logique avec les NAFold-codes)": "naf25_niv5",
        "NAFnew-intitulé\n(niveau sous-classe)": "lib_naf25_niv5",
    }

    return mapping_table.iloc[:, [1, 3, 2, 10, 5, 11]].rename(columns=columns_mapping).copy()


def format_explanatory_notes(explanatory_notes: str) -> str:
    return (
        explanatory_notes.iloc[:, [6, 9, 10, 11, 12, 13, 7, 8]]
        .rename(
            columns={
                "Code NAF 2025": "naf08_niv4",
                "Titre": "lib_naf08_niv5",
                "Note.générale": "notes",
            }
        )
        .assign(naf08_niv4=explanatory_notes["Code NAF 2025"])
    )


def find_explanatory_notes(
    explanatory_notes: pd.DataFrame, code25: str, naf_code_column: str = "naf08_niv4"
) -> pd.Series:
    """
    Finds the explanatory notes for a given NAF 2025 code by looking it up in the explanatory_notes DataFrame.
    The function first searches for the full code and progressively attempts to find related codes if no match is found.

    Parameters:
    ----------
    explanatory_notes : pd.DataFrame
        The DataFrame containing the explanatory notes for NAF codes, with columns for NAF codes, general notes,
        what the code comprises, and what it does not comprise.

    code25 : str
        The NAF 2025 code to find explanatory notes for.

    naf_code_column : str, optional
        The name of the column in explanatory_notes where the NAF code is stored. Default is 'naf08_niv4'.

    Returns:
    -------
    row : pd.Series
        A row from the explanatory_notes DataFrame containing the notes for the NAF code.

    """

    explanatory_notes = format_explanatory_notes(explanatory_notes)

    # Attempt to find level 5 naf code in explanatory_notes
    row = explanatory_notes.loc[explanatory_notes[naf_code_column] == code25]

    # If doesn't exist, try to find level 4 naf code in explanatory_notes (remove the last character)
    if row.empty:
        raise ValueError(f"Could not find notes for {code25}")

    return row.iloc[0]


def get_explanatory_notes(explanatory_notes: pd.DataFrame, code25: str, note_type: str) -> str:
    """
    Retrieves explanatory notes of a specific type ('include', 'not_include', or 'notes') for a given NAF 2025 code.

    Parameters:
    ----------
    explanatory_notes : pd.DataFrame
        DataFrame containing the explanatory notes for NAF codes.

    code25 : str
        NAF 2025 code to look up the explanatory notes.

    note_type : str
        Type of explanatory note to retrieve. Can be 'include', 'not_include', or 'notes'.

    Returns:
    -------
    str
        The corresponding explanatory note for the given NAF code and note type, or None if no note exists.
    """
    # Lookup the row for the given NAF 2025 code
    row = find_explanatory_notes(explanatory_notes, code25)

    # Map note_type to corresponding columns
    note_mapping = {
        "include": ["Comprend", "Comprend.aussi"],
        "not_include": ["Ne.comprend.pas"],
        "notes": ["notes"],
    }

    if note_type not in note_mapping:
        raise ValueError(f"Invalid note_type '{note_type}'. Must be 'include', 'not_include', or 'notes'.")

    # Extract the relevant fields for the note_type
    columns = note_mapping[note_type]
    extracted_values = [
        unicodedata.normalize("NFKC", row[col]) for col in columns if col in row and pd.notnull(row[col])
    ]

    # Combine relevant fields for 'include' or return the value for 'not_include' and 'notes'
    if extracted_values:
        return "\n".join(extracted_values) if note_type == "include" else extracted_values[0]


def get_mapping(explanatory_notes: pd.DataFrame, mapping_table: pd.DataFrame) -> list[NAF2008]:
    """
    Generates a mapping of NAF 2008 codes to NAF 2025 codes with explanatory notes.
    """
    mapping_table = format_mapping_table(mapping_table)
    return [
        NAF2008(
            code=code08,
            label=subset["lib_naf08_niv5"].iloc[0],
            naf2025=[
                NAF2025(
                    code=row.naf25_niv5,
                    label=row.lib_naf25_niv5,
                    include=get_explanatory_notes(explanatory_notes, row.naf25_niv5, "include"),
                    not_include=get_explanatory_notes(explanatory_notes, row.naf25_niv5, "not_include"),
                    notes=get_explanatory_notes(explanatory_notes, row.naf25_niv5, "notes"),
                )
                for row in subset.itertuples()
            ],
        )
        for code08, subset in mapping_table.groupby("naf08_niv5")
    ]


def get_nace2025_from_mapping(mapping: List[NAF2008]) -> List[NAF2025]:
    """Generate unique NAF2025 codes from the mapping."""
    unique_codes = {}
    for code08 in mapping:
        for code25 in code08.naf2025:
            if code25.code not in unique_codes:
                unique_codes[code25.code] = code25
    return list(unique_codes.values())


def check_mapping(naf08: str, naf25: str, naf08_to_naf2025: Dict[str, List[str]]) -> bool:
    """Check if mapping is correct."""
    return naf25 in naf08_to_naf2025.get(naf08, [])
