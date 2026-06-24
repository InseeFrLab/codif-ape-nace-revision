import argparse
import logging
import os

import duckdb
import pandas as pd

import config
from constants.data import ID_VAR, NACE08_VAR
from constants.paths import (
    URL_EXPLANATORY_NOTES,
    URL_MAPPING_TABLE,
    URL_SIRENE4_EXTRACTION,
    URL_WORKFLOW_UNIVOCAL,
)
from mappings.mappings import get_mapping
from utils.data import get_file_system

config.setup()
logger = logging.getLogger(__name__)


def encode_unambiguous(input_url: str, output_url: str):
    """Relabel the unambiguous (univocal) NAF 2008 → NAF 2025 codes.

    For every NAF 2008 code that maps to exactly one NAF 2025 code, rewrite it
    directly (no LLM needed) and write the result to `output_url`.

    Args:
        input_url: S3 path of the source SIRENE 4 extraction.
        output_url: S3 path of the univocal predictions Parquet to write.
    """
    logger.info("===== STEP 2: encode unambiguous =====")
    logger.info("INPUT  : %s", input_url)
    logger.info("INPUT  : %s", URL_MAPPING_TABLE)
    logger.info("INPUT  : %s", URL_EXPLANATORY_NOTES)
    logger.info("OUTPUT : %s", output_url)

    fs = get_file_system()

    # Load excel files containing informations about mapping
    with fs.open(URL_MAPPING_TABLE) as f:
        table_corres = pd.read_excel(f, dtype=str)

    with fs.open(URL_EXPLANATORY_NOTES) as f:
        notes_ex = pd.read_excel(f, dtype=str)

    mapping = get_mapping(notes_ex, table_corres)

    # Select all univoque codes. Mapping codes are dotted ("01.11Z") while the
    # input NACE08_VAR column is validated dotless ("0111Z"), so strip dots on
    # both the NAF08 key (to match the input) and the NAF2025 value (kept dotless,
    # consistent with the ambiguous path's postprocessing).
    univoques = {
        code.code.replace(".", ""): code.naf2025[0].code.replace(".", "")
        for code in mapping
        if len(code.naf2025) == 1
    }
    logger.info("Found %d univocal NAF 2008 codes to rewrite.", len(univoques))

    con = duckdb.connect(database=":memory:")

    # Construct the CASE statement from the dictionary mapping
    case_statement = "CASE "
    for nace08, nace2025 in univoques.items():
        case_statement += f"WHEN {NACE08_VAR} = '{nace08}' THEN '{nace2025}' "
    case_statement += "ELSE NULL END AS nace2025"

    # SQL query with renamed column and new column using CASE for mapping
    query = f"""
        SELECT
            {ID_VAR},
            {case_statement}
        FROM
            read_parquet('{input_url}')
        WHERE
            {NACE08_VAR} IN ('{"', '".join(univoques.keys())}')
    """

    con.execute(
        f"""
        SET s3_endpoint='{os.getenv("AWS_S3_ENDPOINT")}';
        SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
        SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
        SET s3_session_token='';

        COPY
        ({query})
        TO '{output_url}'
        (FORMAT 'parquet')
    ;
    """
    )
    logger.info("✅ Univocal predictions written to %s", output_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Relabel unambiguous (univocal) NAF codes.")
    parser.add_argument(
        "--input_url",
        type=str,
        default=None,
        help="S3 path of the source extraction. Defaults to URL_SIRENE4_EXTRACTION.",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help="Run id scoping all workflow outputs. Used to derive --output_url when omitted.",
    )
    parser.add_argument(
        "--output_url",
        type=str,
        default=None,
        help="S3 path of the univocal predictions Parquet. Defaults to the job-scoped path.",
    )
    args = parser.parse_args()

    if args.output_url is None and args.job_id is None:
        parser.error("provide --job_id (to derive the output path) or an explicit --output_url")

    encode_unambiguous(
        input_url=args.input_url or URL_SIRENE4_EXTRACTION,
        output_url=args.output_url or URL_WORKFLOW_UNIVOCAL.format(job_id=args.job_id),
    )
