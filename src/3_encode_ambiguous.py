import asyncio
import logging
import os
import tempfile
import time
from datetime import datetime

import mlflow

import config
from constants.paths import URL_SIRENE4_EXTRACTION, URL_WORKFLOW_AMBIGUOUS
from evaluation.evaluator import Evaluator
from strategies.base import EncodeStrategy
from strategies.cag import CAGStrategy
from strategies.rag import RAGStrategy
from constants.data import VAR_TO_KEEP
from utils.batch import (
    BatchPaths,
    completed_batch_ids,
    iter_batches,
    merge_token_stats,
    read_all_prompts,
    read_all_results,
    write_batch,
)
from utils.data import get_ambiguous_data, write_run_id_to_s3
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
    prompt_name: str,
    prompt_label: str,
    top_k: int,
    mode: str,
    job_id: str,
    batch_size: int,
    input_url: str = None,
    sample_size: int = None,
    thinking: bool = False,
    max_new_tokens: int | None = None,
):
    """Main workflow: encode ambiguous data in resumable batches, then evaluate
    (eval mode) and log everything to MLflow."""

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        strategy = _initialize_strategy(
            strategy_cls, llm_name, prompt_name, prompt_label, collection_name,
            thinking=thinking, max_new_tokens=max_new_tokens,
        )
        data = _load_data(strategy, third, mode, input_url, sample_size)
        if data.empty:
            raise ValueError("No data to encode after loading/filtering.")

        ambiguous_dir = f"{URL_WORKFLOW_AMBIGUOUS.format(job_id=job_id)}/{strategy.model_subdir}"
        paths = BatchPaths(ambiguous_dir)
        logging.info("===== STEP 3: encode ambiguous (%s, mode=%s) =====", strategy.__class__.__name__, mode)
        logging.info("INPUT  : %s", input_url or URL_SIRENE4_EXTRACTION)
        logging.info("OUTPUT : %s (job_id=%s)", paths.results_dir, job_id)

        run_stats = await _run_batches(strategy, data, top_k, batch_size, paths)

        # Reconsolidate from S3 so metrics/eval cover ALL batches, including
        # those completed in earlier (resumed) executions.
        results = read_all_results(strategy.fs, paths)
        prompts = read_all_prompts(strategy.fs, paths) if mode == "eval" else None

        metrics, df_eval = _evaluate_and_enrich(results, prompts, run_stats, strategy, mode)
        _log_mlflow(
            strategy, llm_name, collection_name, results, metrics, df_eval, top_k,
            mode=mode, output_path=paths.results_dir, input_url=input_url,
            prompts=prompts, run_name=run_name, sample_size=sample_size,
        )
        write_run_id_to_s3(job_id, llm_name, mlflow.active_run().info.run_id)


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


def _load_data(strategy, third, mode, input_url=None, sample_size=None):
    logging.info("Loading ambiguous data ==========================")
    # eval mode: input_url=None → filters to annotated rows via ground-truth join
    # prod mode: input_url=<path> → uses the provided file, no ground-truth filter
    data = get_ambiguous_data(strategy.mapping, third, input_url if mode == "prod" else None, VAR_TO_KEEP)
    if sample_size is not None:
        # Seeded so batch boundaries stay reproducible across resumes (same
        # job_id + sample_size => same sampled rows => resumable).
        data = data.sample(n=sample_size, random_state=2025).reset_index(drop=True)
    return data


async def _run_batches(strategy, data, top_k, batch_size, paths):
    """Encode `data` in batches of `batch_size`, persisting prompts+results per
    batch to S3. Batches whose results part already exists are skipped (resume).

    LLM call mechanics are unchanged: each batch is a single call_llm over its
    prompts, so concurrency/retries behave exactly as in a one-shot run.

    Returns timing and token stats for the batches run in THIS execution
    (skipped batches contribute nothing; consolidated row-level metrics are
    computed separately from the full S3 output).
    """
    done = completed_batch_ids(strategy.fs, paths)
    n_batches = (len(data) + batch_size - 1) // batch_size
    if done:
        logging.info("Resuming: %d/%d batches already completed", len(done), n_batches)

    retrieval_time_mn = 0.0
    generation_time_mn = 0.0
    token_stats_per_batch = []

    for batch_id, batch_df in iter_batches(data, batch_size):
        if batch_id in done:
            logging.info("Batch %d already done — skipping", batch_id)
            continue

        logging.info("Batch %d/%d — %d rows ==========", batch_id, n_batches - 1, len(batch_df))

        t0 = time.time()
        prompts = await strategy.get_prompts(batch_df, top_k=top_k)
        retrieval_time_mn += (time.time() - t0) / 60

        t1 = time.time()
        outputs = await strategy.call_llm(prompts)
        generation_time_mn += (time.time() - t1) / 60

        processed = strategy.process_outputs(outputs)
        results = batch_df.merge(processed, left_index=True, right_index=True)

        # Persist before moving on so a later crash never loses this batch.
        write_batch(strategy.fs, paths, batch_id, prompts, results)
        token_stats_per_batch.append(strategy.token_stats)

    return {
        "retrieval_time_mn": retrieval_time_mn,
        "generation_time_mn": generation_time_mn,
        "token_stats": merge_token_stats(token_stats_per_batch),
    }


def _evaluate_and_enrich(results, prompts, run_stats, strategy, mode):
    generation_time_sec = run_stats["generation_time_mn"] * 60
    iter_per_sec = len(results) / generation_time_sec if generation_time_sec > 0 else 0.0
    metrics = {
        "num_coded": results["codable"].sum(),
        "num_not_coded": len(results) - results["codable"].sum(),
        "pct_not_coded": round((len(results) - results["codable"].sum()) / len(results) * 100, 2),
        "retrieval_time_mn": round(run_stats["retrieval_time_mn"], 1),
        "generation_time_mn": round(run_stats["generation_time_mn"], 1),
        "generation_iter_per_sec": round(iter_per_sec, 2),
    }
    metrics.update(run_stats["token_stats"])

    if mode == "prod":
        return metrics, None

    eval_metrics, df_eval = Evaluator().evaluate(results, prompts)
    metrics.update(eval_metrics)
    return metrics, df_eval


def _log_mlflow(
    strategy, llm_name, collection_name, results, metrics, df_eval, top_k,
    *, mode, output_path, input_url=None, prompts=None, run_name=None, sample_size=None,
):
    params = {
        "LLM_MODEL": llm_name,
        "TEMPERATURE": strategy.sampling_params["temperature"],
        "MAX_NEW_TOKENS": strategy.sampling_params["max_tokens"],
        "THINKING": strategy.thinking,
        "input_path": input_url or URL_SIRENE4_EXTRACTION,
        "output_path": output_path,
        "strategy": "cag" if isinstance(strategy, CAGStrategy) else "rag",
        "top_k": top_k,
        "mode": mode,
    }

    if hasattr(strategy, "db"):
        params["COLLECTION_NAME"] = collection_name
        params["EMBEDDING_MODEL"] = getattr(strategy.db, "model_name", None)

    mlflow.log_params(params)
    for metric, value in metrics.items():
        mlflow.log_metric(metric, value)

    if mode == "eval":
        report_md = build_report(
            strategy, llm_name, collection_name, top_k, sample_size, True, metrics, run_name,
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
    parser.add_argument(
        "--mode",
        choices=["prod", "eval"],
        required=True,
        help=(
            "prod: run on provided input_url, skip evaluation metrics. "
            "eval: run on URL_SIRENE4_EXTRACTION filtered to annotated rows, compute metrics."
        ),
    )
    parser.add_argument(
        "--input_url",
        type=str,
        default=None,
        help="S3 path to the input Parquet file (prod mode only).",
    )
    parser.add_argument(
        "--job_id",
        type=str,
        default=None,
        help=(
            "Stable identifier for the output directory. Pass the SAME job_id to "
            "resume a crashed run (completed batches are skipped). Omit to start "
            "fresh (a timestamped job_id is generated; no resume)."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Number of rows per batch (prompts built, LLM-called, and committed together).",
    )
    parser.add_argument("--experiment_name", type=str, default="Test")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--collection_name", type=str, default=None)
    parser.add_argument("--llm_name", type=str, choices=["qwen3-6-35b-moe", "gemma4-26b-moe"])
    parser.add_argument("--third", type=int, default=None)
    parser.add_argument("--prompt_name", type=str, default=None)
    parser.add_argument("--prompt_label", type=str, default="production")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--sample_size", type=int, default=None)
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

    if args.mode == "prod" and args.input_url is None:
        parser.error("--input_url is required in prod mode")

    # Sampling is seeded (see _load_data), so a sampled run is still resumable
    # against the same job_id — no mutual exclusion needed.
    if args.job_id is None:
        args.job_id = datetime.now().strftime("run-%Y%m%d-%H%M%S")
        logging.info("No --job_id provided — fresh run (no resume). job_id=%s", args.job_id)

    # Harmonise identifiers: the MLflow run name is derived from the job_id (one
    # run per model) so Argo run / job_id / S3 paths / MLflow all share one id.
    if args.run_name is None:
        suffix = "-thinking" if args.thinking else ""
        args.run_name = f"{args.job_id}--{args.llm_name}{suffix}"

    if args.strategy == "cag":
        args.prompt_name = "cag-classifier"
        args.top_k = None
    else:
        args.prompt_name = "rag-classifier"

    print("Arguments used :")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")

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
            prompt_name=args.prompt_name,
            prompt_label=args.prompt_label,
            sample_size=args.sample_size,
            top_k=args.top_k,
            mode=args.mode,
            job_id=args.job_id,
            batch_size=args.batch_size,
            input_url=args.input_url,
            thinking=args.thinking,
            max_new_tokens=args.max_new_tokens,
        )
    )
