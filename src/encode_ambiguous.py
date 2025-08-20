import asyncio
import logging
import os
import time
import tempfile
import mlflow
import config
from constants.paths import URL_SIRENE4_EXTRACTION
from evaluation.evaluator import Evaluator
from strategies.base import EncodeStrategy
from strategies.cag import CAGStrategy
from strategies.rag import RAGStrategy
from utils.data import get_ambiguous_data

config.setup()


async def run_encode(
    strategy_cls: EncodeStrategy,
    experiment_name: str,
    run_name: str,
    collection_name: str,
    llm_name: str,
    third: int,
    prompts_from_file: bool,
    prompt_name: str,
    prompt_label: str,
    sample_size: int = None,
):
    """Main workflow to run encoding strategy, generate prompts, call LLM, evaluate, and log with MLflow."""

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        strategy = _initialize_strategy(strategy_cls, llm_name, prompt_name, prompt_label, collection_name)
        data = _load_data(strategy, third, sample_size)
        prompts, retrieval_time_mn = await _retrieve_prompts(strategy, data, prompts_from_file)
        generation_outputs, generation_time_mn = _generate_outputs(strategy, prompts)
        results = _process_and_merge(strategy, data, generation_outputs)
        metrics, df_eval = _evaluate_and_enrich(results, prompts, retrieval_time_mn, generation_time_mn, strategy)
        _log_mlflow(strategy, llm_name, collection_name, results, metrics, df_eval)


def _initialize_strategy(strategy_cls, llm_name, prompt_name, prompt_label, collection_name):
    logging.info("Initializing strategy ==========================")
    return strategy_cls(
        generation_model=llm_name,
        prompt_name=prompt_name,
        prompt_label=prompt_label,
        collection_name=collection_name,
    )


def _load_data(strategy, third, sample_size=None):
    logging.info("Loading ambiguous data ==========================")
    data = get_ambiguous_data(strategy.mapping, third, only_annotated=True)
    if sample_size is not None:
        data = data.head(n=sample_size).reset_index(drop=True)
    return data


async def _retrieve_prompts(strategy, data, load_from_file=False):
    logging.info("Retrieving prompts ==========================")
    start_time = time.time()
    prompts = await strategy.get_prompts(data, load_prompts_from_file=load_from_file)
    retrieval_time_mn = (time.time() - start_time) / 60
    logging.info("Prompts retrieved")
    return prompts, retrieval_time_mn


def _generate_outputs(strategy, prompts):
    logging.info("Starting generation ==========================")
    start_time = time.time()
    outputs = strategy.call_llm(prompts, strategy.sampling_params)
    generation_time_mn = (time.time() - start_time) / 60
    return outputs, generation_time_mn


def _process_and_merge(strategy, data, outputs):
    processed_outputs = strategy.process_outputs(outputs)
    return data.merge(processed_outputs, left_index=True, right_index=True)


def _evaluate_and_enrich(results, prompts, retrieval_time_mn, generation_time_mn, strategy):
    metrics, df_eval = Evaluator().evaluate(results, prompts)
    metrics.update(
        {
            "num_coded": results["codable"].sum(),
            "num_not_coded": len(results) - results["codable"].sum(),
            "pct_not_coded": round((len(results) - results["codable"].sum()) / len(results) * 100, 2),
            "retrieval_time_mn": round(retrieval_time_mn, 1),
            "generation_time_mn": round(generation_time_mn, 1),
        }
    )
    return metrics, df_eval


def _log_mlflow(strategy, llm_name, collection_name, results, metrics, df_eval):
    output_path = strategy.save_results(results, third=None)
    mlflow.log_params(
        {
            "LLM_MODEL": llm_name,
            "TEMPERATURE": strategy.sampling_params.temperature,
            "input_path": URL_SIRENE4_EXTRACTION,
            "output_path": output_path,
            "strategy": "cag" if isinstance(strategy, CAGStrategy) else "rag",
            "COLLECTION_NAME": collection_name,
            "EMBEDDING_MODEL": strategy.db.vector_name,
        }
    )

    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "df_eval.csv")
        df_eval.to_csv(file_path, index=False)
        mlflow.log_artifact(file_path, artifact_path="dataframes")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["rag", "cag"], required=True)
    parser.add_argument("--experiment_name", type=str, default="Test")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--collection_name", type=str, default="embeddings_qwen")
    parser.add_argument("--llm_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--third", type=int, default=None)
    parser.add_argument("--prompts_from_file", action="store_true")
    parser.add_argument("--prompt_name", type=str, default="rag-classifier")
    parser.add_argument("--prompt_label", type=str, default="production")
    parser.add_argument("--sample_size", type=int, default=None)

    args = parser.parse_args()

    assert "MLFLOW_TRACKING_URI" in os.environ, "Set MLFLOW_TRACKING_URI"

    # Logging of parameters
    logging.info("===== Run parameters =====")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("==========================")

    STRATEGY_MAP = {
        "rag": RAGStrategy,
        "cag": CAGStrategy,
    }

    asyncio.run(
        run_encode(
            strategy_cls=STRATEGY_MAP[args.strategy],
            experiment_name=args.experiment_name,
            run_name=args.run_name,
            collection_name=args.collection_name,
            llm_name=args.llm_name,
            third=args.third,
            prompts_from_file=args.prompts_from_file,
            prompt_name=args.prompt_name,
            prompt_label=args.prompt_label,
            sample_size=args.sample_size,
        )
    )
