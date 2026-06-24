"""Step 0 — validate the input data before the relabeling pipeline runs.

Checks, using DuckDB push-down queries (cheap even on large Parquet files):
  - every column required downstream (constants.data.VAR_TO_KEEP) is present;
  - `liasse_numero` and `libelle` are non-null/non-empty;
  - `apet_finale` is a well-formed NAF 2008 code (4 digits + 1 uppercase letter).

Duplicate `liasse_numero` are tolerated (downstream uses DISTINCT ON) and only
warned about. The step writes no data: it is a gate that raises on failure.
"""

import argparse
import logging

import config
from constants.data import ACTIVITY_LABEL_VAR, ID_VAR, NACE08_VAR, VAR_TO_KEEP
from constants.paths import URL_SIRENE4_EXTRACTION
from utils.data import load_data_from_s3

config.setup()
logger = logging.getLogger(__name__)

# NAF 2008 / APE code: 4 digits followed by one uppercase letter, e.g. "0111Z".
NAF08_PATTERN = "^[0-9]{4}[A-Z]$"


def _column_names(url: str) -> list[str]:
    desc = load_data_from_s3(f"DESCRIBE SELECT * FROM read_parquet('{url}')")
    return list(desc["column_name"])


def validate_input(input_url: str) -> None:
    logger.info("===== STEP 0: input validation =====")
    logger.info("INPUT  : %s", input_url)
    logger.info("OUTPUT : none (validation gate — raises on failure)")

    # 1) Required columns present
    columns = _column_names(input_url)
    missing = [c for c in VAR_TO_KEEP if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_url}: {missing}")
    logger.info("All %d required columns present.", len(VAR_TO_KEEP))

    # 2) Value-level quality (single aggregate pass)
    query = f"""
        SELECT
            count(*) AS n_rows,
            count(DISTINCT {ID_VAR}) AS liasse_distinct,
            count(*) FILTER (WHERE {ID_VAR} IS NULL) AS liasse_null,
            count(*) FILTER (WHERE {NACE08_VAR} IS NULL) AS apet_null,
            count(*) FILTER (
                WHERE {NACE08_VAR} IS NOT NULL
                AND NOT regexp_matches({NACE08_VAR}, '{NAF08_PATTERN}')
            ) AS apet_bad_format,
            count(*) FILTER (WHERE {ACTIVITY_LABEL_VAR} IS NULL OR trim({ACTIVITY_LABEL_VAR}) = '') AS libelle_empty
        FROM read_parquet('{input_url}')
    """
    stats = load_data_from_s3(query).iloc[0].to_dict()
    logger.info("Rows: %s | distinct liasse: %s", stats["n_rows"], stats["liasse_distinct"])
    logger.info(
        "Nulls — liasse_numero: %s, apet_finale: %s | apet bad format: %s | libelle empty: %s",
        stats["liasse_null"], stats["apet_null"], stats["apet_bad_format"], stats["libelle_empty"],
    )

    errors = []
    if stats["n_rows"] == 0:
        errors.append("input file is empty")
    if stats["liasse_null"]:
        errors.append(f"{stats['liasse_null']} rows with null liasse_numero")
    if stats["apet_null"]:
        errors.append(f"{stats['apet_null']} rows with null apet_finale")
    if stats["apet_bad_format"]:
        errors.append(f"{stats['apet_bad_format']} rows with apet_finale not matching {NAF08_PATTERN}")
    if stats["libelle_empty"]:
        errors.append(f"{stats['libelle_empty']} rows with empty libelle")

    duplicates = stats["n_rows"] - stats["liasse_distinct"]
    if duplicates:
        logger.warning("%s duplicate liasse_numero (tolerated downstream via DISTINCT ON).", duplicates)

    if errors:
        raise ValueError("Input validation FAILED:\n - " + "\n - ".join(errors))
    logger.info("✅ Input validation passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate input data before the relabeling pipeline.")
    parser.add_argument("--mode", choices=["prod", "eval"], required=True)
    parser.add_argument(
        "--input_url",
        type=str,
        default=None,
        help="S3 path to validate. Required in prod; defaults to URL_SIRENE4_EXTRACTION in eval.",
    )
    args = parser.parse_args()

    if args.mode == "prod" and args.input_url is None:
        parser.error("--input_url is required in prod mode")

    validate_input(args.input_url or URL_SIRENE4_EXTRACTION)
