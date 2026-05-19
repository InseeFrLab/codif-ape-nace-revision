import asyncio
import logging
import os
import tempfile
import time

import mlflow

import config
from constants.paths import URL_SIRENE4_EXTRACTION
from evaluation.evaluator import Evaluator
from strategies.base import EncodeStrategy
from strategies.cag import CAGStrategy
from strategies.rag import RAGStrategy
from constants.data import VAR_TO_KEEP
from utils.data import get_ambiguous_data
from utils.error_report import (
    build_llm_errors_report,
    build_not_codable_report,
    build_retriever_errors_report,
)
from utils.report import build_report

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
    top_k: int,
    only_annotated: bool,
    sample_size: int = None,
    save_prompts: bool = False,
    thinking: bool = False,
    max_new_tokens: int | None = None,
):
    """Main workflow to run encoding strategy, generate prompts, call LLM, evaluate, and log with MLflow."""

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        strategy = _initialize_strategy(
            strategy_cls, llm_name, prompt_name, prompt_label, collection_name,
            thinking=thinking, max_new_tokens=max_new_tokens,
        )
        data = _load_data(strategy, third, only_annotated, sample_size)
        # prompts, retrieval_time_mn = asyncio.run(_retrieve_prompts(strategy, data, top_k, prompts_from_file, save_prompts))
        prompts, retrieval_time_mn = await _retrieve_prompts(strategy, data, top_k, prompts_from_file, save_prompts)

        # generation_outputs, generation_time_mn = asyncio.run(_generate_outputs(strategy, prompts))
        generation_outputs, generation_time_mn = await _generate_outputs(strategy, prompts)
        results = _process_and_merge(strategy, data, generation_outputs)
        metrics, df_eval = _evaluate_and_enrich(results, prompts, retrieval_time_mn, generation_time_mn, strategy)
        _log_mlflow(
            strategy, llm_name, collection_name, results, metrics, df_eval, top_k,
            prompts=prompts,
            run_name=run_name, sample_size=sample_size, only_annotated=only_annotated,
        )


def _initialize_strategy(
    strategy_cls, llm_name, prompt_name, prompt_label, collection_name,
    *, thinking: bool = False, max_new_tokens: int | None = None,
):
    logging.info("Initializing strategy ==========================")

    kwargs = {
        "generation_model": llm_name,
        "prompt_name": prompt_name,
        "prompt_label": prompt_label,
        "thinking": thinking,
        "max_new_tokens": max_new_tokens,
    }

    if strategy_cls in [RAGStrategy]:
        kwargs["collection_name"] = collection_name

    return strategy_cls(**kwargs)


def _load_data(strategy, third, only_annotated, sample_size=None):
    logging.info("Loading ambiguous data ==========================")
    data = get_ambiguous_data(strategy.mapping, third, only_annotated, VAR_TO_KEEP)
    if sample_size is not None:
        data = data.sample(n=sample_size).reset_index(drop=True)
    return data


async def _retrieve_prompts(strategy, data, top_k, load_from_file=False, save_prompts=False):
    logging.info("Retrieving prompts ==========================")
    start_time = time.time()
    prompts = await strategy.get_prompts(
        data,
        load_prompts_from_file=load_from_file,
        top_k=top_k,
        save=save_prompts
    )
    retrieval_time_mn = (time.time() - start_time) / 60
    logging.info("Prompts retrieved")
    return prompts, retrieval_time_mn


async def _generate_outputs(strategy, prompts):
    logging.info("Starting generation ======")
    start_time = time.time()
    outputs = await strategy.call_llm(prompts)
    generation_time_mn = (time.time() - start_time) / 60
    logging.info(f"✅ Génération terminée pour {len(prompts)} prompts en {generation_time_mn:.2f} min.")
    return outputs, generation_time_mn


def _process_and_merge(strategy, data, outputs):
    processed_outputs = strategy.process_outputs(outputs)
    return data.merge(processed_outputs, left_index=True, right_index=True)


def _evaluate_and_enrich(results, prompts, retrieval_time_mn, generation_time_mn, strategy):
    metrics, df_eval = Evaluator().evaluate(results, prompts)
    generation_time_sec = generation_time_mn * 60
    iter_per_sec = len(results) / generation_time_sec if generation_time_sec > 0 else 0.0
    metrics.update(
        {
            "num_coded": results["codable"].sum(),
            "num_not_coded": len(results) - results["codable"].sum(),
            "pct_not_coded": round((len(results) - results["codable"].sum()) / len(results) * 100, 2),
            "retrieval_time_mn": round(retrieval_time_mn, 1),
            "generation_time_mn": round(generation_time_mn, 1),
            "generation_iter_per_sec": round(iter_per_sec, 2),
        }
    )
    metrics.update(strategy.token_stats)
    return metrics, df_eval


def _log_mlflow(
    strategy, llm_name, collection_name, results, metrics, df_eval, top_k,
    *, prompts=None, run_name=None, sample_size=None, only_annotated=None,
):
    output_path = strategy.save_results(results, third=None)
    params = {
        "LLM_MODEL": llm_name,
        "TEMPERATURE": strategy.sampling_params["temperature"],
        "MAX_NEW_TOKENS": strategy.sampling_params["max_tokens"],
        "THINKING": strategy.thinking,
        "input_path": URL_SIRENE4_EXTRACTION,
        "output_path": output_path,
        "strategy": "cag" if isinstance(strategy, CAGStrategy) else "rag",
        "top_k": top_k,
        "URL_SIRENE4_EXTRACTION": URL_SIRENE4_EXTRACTION,
    }

    # If RAG
    if hasattr(strategy, "db"):
        params["COLLECTION_NAME"] = collection_name
        params["EMBEDDING_MODEL"] = getattr(strategy.db, "model_name", None)

    mlflow.log_params(params)
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

    report_md = build_report(
        strategy, llm_name, collection_name, top_k, sample_size, only_annotated, metrics, run_name,
        df_eval=df_eval,
    )
    error_reports = {
        "llm_errors.md": build_llm_errors_report(
            strategy, prompts, results, df_eval, max_examples=5, run_name=run_name,
        ),
        "not_codable.md": build_not_codable_report(
            strategy, prompts, results, df_eval, max_examples=5, run_name=run_name,
        ),
    }
    # Retriever errors only make sense for RAG (CAG has no retrieval step).
    if hasattr(strategy, "db"):
        error_reports["retriever_errors.md"] = build_retriever_errors_report(
            strategy, prompts, results, df_eval, max_examples=5, run_name=run_name,
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        df_path = os.path.join(tmpdir, "df_eval.csv")
        df_eval.to_csv(df_path, index=False)
        mlflow.log_artifact(df_path, artifact_path="dataframes")

        report_path = os.path.join(tmpdir, "report.md")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        mlflow.log_artifact(report_path, artifact_path="reports")

        for filename, content in error_reports.items():
            path = os.path.join(tmpdir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            mlflow.log_artifact(path, artifact_path="reports")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", choices=["rag", "cag"], required=True)
    parser.add_argument("--experiment_name", type=str, default="Test")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--collection_name", type=str, default=None)
    parser.add_argument("--llm_name", type=str, choices=["qwen3-6-35b-moe", "gemma4-26b-moe"])
    parser.add_argument("--third", type=int, default=None)
    parser.add_argument("--prompts_from_file", action="store_true")
    parser.add_argument("--save_prompts", action="store_true")
    parser.add_argument("--prompt_name", type=str, default=None)
    parser.add_argument("--prompt_label", type=str, default="production")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=None)
    parser.add_argument(
        "--only_annotated",
        type=str,
        choices=["true", "false"],
        default="false",
    )
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="Enable LLM thinking mode (longer reasoning, more tokens).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override completion token budget. Defaults: 100 (fast) / 2048 (thinking).",
    )
    args = parser.parse_args()
    
    assert "MLFLOW_TRACKING_URI" in os.environ, "Set MLFLOW_TRACKING_URI"

    if args.only_annotated == "true":
        args.only_annotated = True
    else:
        args.only_annotated = False

    if args.strategy == "cag":
        args.prompt_name = "cag-classifier"
        args.top_k = None
    else:
        args.prompt_name = "rag-classifier"

    print("Arguments used :")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

    # Logging of parameters
    logging.info("===== Run parameters =====")
    for key, value in vars(args).items():
        logging.info(f"{key}: {value}")
    logging.info("==========================")

    STRATEGY_MAP = {
        "cag": CAGStrategy,
        "rag": RAGStrategy,
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
            top_k=args.top_k,
            save_prompts=args.save_prompts,
            only_annotated=args.only_annotated,
            thinking=args.thinking,
            max_new_tokens=args.max_new_tokens,
        )
    )

