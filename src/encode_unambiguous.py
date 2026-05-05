import os

import duckdb
import pandas as pd

from constants.paths import (
    URL_EXPLANATORY_NOTES,
    URL_MAPPING_TABLE,
    URL_SIRENE4_EXTRACTION,
    URL_SIRENE4_UNIVOCAL,
)
from mappings.mappings import get_mapping
from utils.data import get_file_system


def encore_univoque():
    """
    Processes the NAF code mappings and relabels the source dataset with new NAF codes (2025 version)
    for rows with unambiguous mappings (univoque codes), then outputs the result as a Parquet file.

    Parameters:
    -----------
    url_source : str
        The S3 URL of the source dataset in Parquet format to be relabeled.

    url_out : str
        The S3 URL where the relabeled output dataset will be saved as a Parquet file.

    Returns:
    --------
    None
        The function writes the relabeled dataset to the specified output location.
    """

    fs = get_file_system()

    # Load excel files containing informations about mapping
    with fs.open(URL_MAPPING_TABLE) as f:
        table_corres = pd.read_excel(f, dtype=str)

    with fs.open(URL_EXPLANATORY_NOTES) as f:
        notes_ex = pd.read_excel(f, dtype=str)

    mapping = get_mapping(notes_ex, table_corres)

    # Select all univoque codes
    univoques = {code.code: code.naf2025[0].code for code in mapping if len(code.naf2025) == 1}

    con = duckdb.connect(database=":memory:")

    # Construct the CASE statement from the dictionary mapping
    case_statement = "CASE "
    for nace08, nace2025 in univoques.items():
        case_statement += f"WHEN apet_finale = '{nace08}' THEN '{nace2025}' "
    case_statement += "ELSE NULL END AS nace2025"

    # SQL query with renamed column and new column using CASE for mapping
    query = f"""
        SELECT
            liasse_numero,
            {case_statement}
        FROM
            read_parquet('{URL_SIRENE4_EXTRACTION}')
        WHERE
            apet_finale IN ('{"', '".join(univoques.keys())}')
    """

    con.execute(
        f"""
        SET s3_endpoint='{os.getenv("AWS_S3_ENDPOINT")}';
        SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
        SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
        SET s3_session_token='';

        COPY
        ({query})
        TO '{URL_SIRENE4_UNIVOCAL}'
        (FORMAT 'parquet')
    ;
    """
    )


if __name__ == "__main__":
    encore_univoque()
