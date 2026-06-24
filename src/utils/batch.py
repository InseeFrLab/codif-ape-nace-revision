"""Resumable batched encoding: stable S3 layout, per-batch IO, and resume helpers.

Each run writes one Parquet part per batch under a stable, job-scoped directory:

    <base_dir>/<job_id>/results/part-XXXXX.parquet   (presence = batch committed)
    <base_dir>/<job_id>/prompts/part-XXXXX.parquet

Re-running with the same `job_id` skips batches whose results part already
exists, so a crashed multi-hour run resumes where it stopped. Batch boundaries
are deterministic given identical, deterministically-ordered input data.
"""

import logging
from typing import Dict, List

import pandas as pd

from utils.data import df_to_prompts, prompts_to_df

logger = logging.getLogger(__name__)


def _strip_scheme(path: str) -> str:
    """s3fs filesystem methods expect bucket-relative paths, not s3:// URLs."""
    return path.replace("s3://", "")


class BatchPaths:
    """Stable, resumable S3 layout for one (model, job_id) encoding run."""

    def __init__(self, base_dir: str, job_id: str):
        self.root = f"{base_dir}/{job_id}"
        self.results_dir = f"{self.root}/results"
        self.prompts_dir = f"{self.root}/prompts"

    def result_file(self, batch_id: int) -> str:
        return f"{self.results_dir}/part-{batch_id:05d}.parquet"

    def prompt_file(self, batch_id: int) -> str:
        return f"{self.prompts_dir}/part-{batch_id:05d}.parquet"


def iter_batches(data: pd.DataFrame, batch_size: int):
    """Yield (batch_id, batch_df) in order. Each batch_df has a reset index so
    it aligns positionally with the LLM outputs on merge."""
    for batch_id, start in enumerate(range(0, len(data), batch_size)):
        yield batch_id, data.iloc[start : start + batch_size].reset_index(drop=True)


def completed_batch_ids(fs, paths: BatchPaths) -> set:
    """Return the set of batch ids whose results part already exists on S3."""
    directory = _strip_scheme(paths.results_dir)
    if not fs.exists(directory):
        return set()
    ids = set()
    for f in fs.ls(directory):
        name = f.split("/")[-1]
        if name.startswith("part-") and name.endswith(".parquet"):
            ids.add(int(name[len("part-") : -len(".parquet")]))
    return ids


def write_batch(fs, paths: BatchPaths, batch_id: int, prompts: List, results_df: pd.DataFrame) -> None:
    """Persist one batch. Prompts are written first; the results part is written
    last and serves as the atomic commit marker used by `completed_batch_ids`."""
    prompts_to_df(prompts).to_parquet(paths.prompt_file(batch_id), filesystem=fs)
    results_df.to_parquet(paths.result_file(batch_id), filesystem=fs)
    logger.info("Committed batch %d (%d rows)", batch_id, len(results_df))


def _sorted_parts(fs, directory: str) -> List[str]:
    # Zero-padded names sort lexicographically in numeric batch order.
    return sorted(f for f in fs.ls(_strip_scheme(directory)) if f.endswith(".parquet"))


def read_all_results(fs, paths: BatchPaths) -> pd.DataFrame:
    """Reconcatenate every committed results part, in batch order."""
    parts = _sorted_parts(fs, paths.results_dir)
    if not parts:
        raise RuntimeError(f"No results found under {paths.results_dir}")
    return pd.concat((pd.read_parquet(f, filesystem=fs) for f in parts), ignore_index=True)


def read_all_prompts(fs, paths: BatchPaths) -> List:
    """Reconcatenate every prompts part, in the same batch order as the results,
    and rebuild the list-of-conversations the Evaluator expects."""
    parts = _sorted_parts(fs, paths.prompts_dir)
    df = pd.concat((pd.read_parquet(f, filesystem=fs) for f in parts), ignore_index=True)
    return df_to_prompts(df)


def merge_token_stats(stats_list: List[Dict]) -> Dict:
    """Aggregate per-batch token stats (see EncodeStrategy._compute_token_stats)
    into a single summary across all batches run in this execution."""
    stats_list = [s for s in stats_list if s]
    if not stats_list:
        return {}
    n = sum(s["n_calls"] for s in stats_list)
    completion_sum = sum(s["completion_tokens_sum"] for s in stats_list)
    prompt_sum = sum(s["prompt_tokens_sum"] for s in stats_list)
    return {
        "completion_tokens_mean": completion_sum / n,
        "completion_tokens_max": max(s["completion_tokens_max"] for s in stats_list),
        "completion_tokens_min": min(s["completion_tokens_min"] for s in stats_list),
        "completion_tokens_sum": completion_sum,
        "prompt_tokens_mean": prompt_sum / n,
        "prompt_tokens_max": max(s["prompt_tokens_max"] for s in stats_list),
        "prompt_tokens_min": min(s["prompt_tokens_min"] for s in stats_list),
        "prompt_tokens_sum": prompt_sum,
        "n_calls": n,
    }
