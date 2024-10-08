import argparse
import os

import duckdb
import mlflow
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from langchain_core.output_parsers import PydanticOutputParser
from vllm import LLM
from vllm.sampling_params import SamplingParams

from src.constants.llm import LLM_MODEL, MAX_NEW_TOKEN, TEMPERATURE, TOP_P, REP_PENALTY, MODEL_TO_ARGS
from src.constants.paths import (
    URL_EXPLANATORY_NOTES,
    URL_MAPPING_TABLE,
    URL_SIRENE4_EXTRACTION,
    URL_SIRENE4_MULTIVOCAL,
    URL_GROUND_TRUTH,
)
from src.constants.prompting import MODEL_TO_PROMPT_FORMAT
from src.llm.model import cache_model_from_hf_hub
from src.llm.prompting import generate_prompt, apply_template
from src.llm.response import LLMResponse, process_response
from src.mappings.mappings import get_mapping
from src.utils.data import get_file_system


def encore_multivoque(
    experiment_name: str,
    run_name: str,
):
    parser = PydanticOutputParser(pydantic_object=LLMResponse)
    fs = get_file_system()

    # Load excel files containing informations about mapping
    with fs.open(URL_MAPPING_TABLE) as f:
        table_corres = pd.read_excel(f, dtype=str)

    with fs.open(URL_EXPLANATORY_NOTES) as f:
        notes_ex = pd.read_excel(f, dtype=str)

    mapping = get_mapping(notes_ex, table_corres)
    mapping_multivocal = [code for code in mapping if len(code.naf2025) > 1]

    con = duckdb.connect(database=":memory:")
    data = con.query(
        f"""
        SET s3_endpoint='{os.getenv("AWS_S3_ENDPOINT")}';
        SET s3_access_key_id='{os.getenv("AWS_ACCESS_KEY_ID")}';
        SET s3_secret_access_key='{os.getenv("AWS_SECRET_ACCESS_KEY")}';
        SET s3_session_token='';

        SELECT
            *
        FROM
            read_parquet('{URL_SIRENE4_EXTRACTION}')
        WHERE
            apet_finale IN ('{"', '".join([m.code for m in mapping_multivocal])}')
    ;
    """
    ).to_df()

    # We keep only unique ids
    data = data[~data.duplicated(subset="liasse_numero")]

    # We keep only non duplicated description and complementary variables
    data = data[
        ~data.duplicated(
            subset=[
                "apet_finale",
                "libelle_activite",
                "evenement_type",
                "cj",
                "activ_nat_et",
                "liasse_type",
                "activ_surf_et",
            ]
        )
    ]
    data.reset_index(drop=True, inplace=True)
    con.close()

    ground_truth = (
        pq.ParquetDataset(URL_GROUND_TRUTH.replace("s3://", ""), filesystem=fs).read().to_pandas()
    )

    data = data.loc[data["liasse_numero"].isin(ground_truth["liasse_numero"].sample(10).tolist())]

    cache_model_from_hf_hub(
        LLM_MODEL,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_NEW_TOKEN,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REP_PENALTY,
    )

    llm = LLM(
        model=LLM_MODEL,
        **MODEL_TO_ARGS.get(LLM_MODEL, {})
    )

    prompts = [generate_prompt(row, mapping_multivocal, parser) for row in data.itertuples()]

    batch_prompts = apply_template([p.prompt for p in prompts], MODEL_TO_PROMPT_FORMAT[LLM_MODEL])

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
        responses = [outputs[i].outputs[0].text for i in range(len(outputs))]

        results = [
            process_response(response=response, prompt=prompt, parser=parser)
            for response, prompt in zip(responses, prompts)
        ]

        df = data.merge(pd.DataFrame(results), on="liasse_numero").loc[
                :,
                [
                    "liasse_numero",
                    "apet_finale",
                    "nace2025",
                    "libelle_activite",
                    "activ_sec_agri_et",
                    "activ_nat_lib_et",
                    "evenement_type",
                    "cj",
                    "activ_nat_et",
                    "liasse_type",
                    "activ_surf_et",
                    "nace08_valid",
                    "codable",
                ],
            ]

        pq.write_to_dataset(
            pa.Table.from_pandas(df),
            root_path=f"{URL_SIRENE4_MULTIVOCAL}/{"--".join(LLM_MODEL.split("/"))}",
            partition_cols=["nace08_valid", "codable"],
            basename_template="part-{i}.parquet",
            existing_data_behavior="overwrite_or_ignore",
            filesystem=fs,
        )

        ground_truth = (
            pq.ParquetDataset(URL_GROUND_TRUTH.replace("s3://", ""), filesystem=fs)
            .read()
            .to_pandas()
        )

        mlflow.log_param("num_not_coded", len(df) - df["codable"].sum())
        mlflow.log_param("pct_not_coded", round((len(df) - df["codable"].sum())/len(df) * 100, 2))

        # Keep only rows coded by the model
        df = df[df["codable"]]

        df = ground_truth.merge(
            df, on="liasse_numero", suffixes=("_gt", "_llm")
        )  # .loc[: , ["liasse_numero", "nace2025", "apet_manual"]]

        accuracies = {
            f"accuracy_lvl_{i}": round(
                (df["apet_manual"].str[:i] == df["nace2025"].str[:i]).mean() * 100, 2
            )
            for i in [5, 4, 3, 2, 1]
        }
        for metric, value in accuracies.items():
            mlflow.log_metric(metric, value)

        mlflow.log_param("LLM_MODEL", LLM_MODEL)
        mlflow.log_param("TEMPERATURE", TEMPERATURE)
        mlflow.log_param("TOP_P", TOP_P)
        mlflow.log_param("REP_PENALTY", REP_PENALTY)
        mlflow.log_param("input_path", URL_SIRENE4_EXTRACTION)
        mlflow.log_param(
            "output_path", f"{URL_SIRENE4_MULTIVOCAL}/{"--".join(LLM_MODEL.split("/"))}"
        )

        failed_to_log = df[df["apet_manual"].str[:1] != df["nace2025"].str[:1]].loc[:20, ["liasse_numero", "libelle", "apet_manual", "nace2025", "activ_nat_lib_et", "codable"]]
        mlflow.log_table(
            data=failed_to_log,
            artifact_file="sample_misclassified.json",
        )

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recode into NACE2025 nomenclature")

    assert (
        "MLFLOW_TRACKING_URI" in os.environ
    ), "Please set the MLFLOW_TRACKING_URI environment variable."

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="Test",
        help="Experiment name in MLflow",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Run name in MLflow",
    )

    args = parser.parse_args()

    encore_multivoque(**vars(args))
