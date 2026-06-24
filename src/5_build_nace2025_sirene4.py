"""Step 5 — assemble the final NACE 2025 SIRENE 4 dataset.

Combines the three sources of NAF 2025 codes into a single deduplicated table:
  - univocal rewrites           (step 2 output, URL_WORKFLOW_UNIVOCAL)
  - LLM predictions (ambiguous) (step 4 output, URL_WORKFLOW_ENSEMBLE)
  - human annotations           (ground truth, URL_GROUND_TRUTH)

All step inputs/outputs are scoped by --job_id under the run directory.
Auxiliary descriptive variables are re-attached from the source extraction.
"""

import argparse
import logging

import pandas as pd

import config
from constants.paths import (
    URL_GROUND_TRUTH,
    URL_SIRENE4_EXTRACTION,
    URL_WORKFLOW_ENSEMBLE,
    URL_WORKFLOW_FINAL,
    URL_WORKFLOW_UNIVOCAL,
)
from utils.data import get_file_system

config.setup()
logger = logging.getLogger(__name__)

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


def build_nace2025_sirene4(input_url: str, job_id: str, output_url: str = None):
    univocal_url = URL_WORKFLOW_UNIVOCAL.format(job_id=job_id)
    ambiguous_url = URL_WORKFLOW_ENSEMBLE.format(job_id=job_id)
    output_url = output_url or URL_WORKFLOW_FINAL.format(job_id=job_id)

    logger.info("===== STEP 5: build final NACE 2025 dataset =====")
    logger.info("INPUT  : %s (univocal)", univocal_url)
    logger.info("INPUT  : %s (ambiguous, LLM)", ambiguous_url)
    logger.info("INPUT  : %s (ground truth)", URL_GROUND_TRUTH)
    logger.info("INPUT  : %s (auxiliary variables)", input_url)
    logger.info("OUTPUT : %s", output_url)

    fs = get_file_system()

    # Univocal: drop duplicate liasse only
    data_univocal = pd.read_parquet(univocal_url, filesystem=fs)
    data_univocal = data_univocal.drop_duplicates(subset="liasse_numero")

    # Ambiguous (human annotation): drop duplicate liasse, rename apet_manual → nace2025
    data_ambiguous_ground_truth = (
        pd.read_parquet(URL_GROUND_TRUTH, filesystem=fs)
        .rename(columns={"apet_manual": "nace2025"})
        .loc[:, ["liasse_numero", "nace2025"]]
    )
    data_ambiguous_ground_truth = data_ambiguous_ground_truth.drop_duplicates(subset="liasse_numero")

    # Ambiguous (LLM): drop rows already covered by human annotation
    data_ambiguous = pd.read_parquet(ambiguous_url, filesystem=fs)
    data_ambiguous = data_ambiguous.loc[
        ~data_ambiguous["liasse_numero"].isin(data_ambiguous_ground_truth["liasse_numero"].tolist())
    ]

    # Source SIRENE 4 auxiliary variables: drop duplicate liasse
    data_sirene4 = pd.read_parquet(input_url, filesystem=fs).loc[:, VAR_TO_KEEP]
    data_sirene4 = data_sirene4.drop_duplicates(subset="liasse_numero")

    # Attach auxiliary variables to each source
    data_univocal = data_univocal.merge(data_sirene4, on="liasse_numero", how="left")
    data_ambiguous = data_ambiguous.merge(data_sirene4, on="liasse_numero", how="left")
    data_ambiguous_ground_truth = data_ambiguous_ground_truth.merge(data_sirene4, on="liasse_numero", how="left")

    # few lines are still duplicated, remove them before merge. Old Label Studio pipeline was not 100% perfect
    data_ambiguous_ground_truth = data_ambiguous_ground_truth.drop_duplicates(
        subset=[v for v in VAR_TO_KEEP if v != "liasse_numero"]
    )

    # Rebuild multivocal rows by re-injecting NAF 2025 codes onto duplicates
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
    # Some univocal rows leak into multivocal, drop them
    data_ambiguous_ground_truth_resampled.dropna(subset=["liasse_numero"], inplace=True)

    data_sirene4_nace2025 = pd.concat(
        [data_univocal, data_ambiguous_resampled, data_ambiguous_ground_truth_resampled], axis=0
    )

    assert data_sirene4_nace2025.duplicated(subset="liasse_numero").sum() == 0

    data_sirene4_nace2025.to_parquet(output_url, filesystem=fs)
    logger.info("✅ Final dataset (%d rows) written to %s", len(data_sirene4_nace2025), output_url)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the final NACE 2025 SIRENE 4 dataset.")
    parser.add_argument(
        "--input_url",
        type=str,
        default=None,
        help="S3 path of the source extraction (auxiliary variables). Defaults to URL_SIRENE4_EXTRACTION.",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        required=True,
        help="Run id scoping all workflow outputs (locates univocal + ensemble inputs and the final output).",
    )
    parser.add_argument(
        "--output_url",
        type=str,
        default=None,
        help="S3 path of the final NACE 2025 dataset. Defaults to the job-scoped path.",
    )
    args = parser.parse_args()

    build_nace2025_sirene4(
        input_url=args.input_url or URL_SIRENE4_EXTRACTION,
        job_id=args.job_id,
        output_url=args.output_url,
    )
