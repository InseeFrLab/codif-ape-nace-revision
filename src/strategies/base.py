import asyncio
import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import httpx
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, RateLimitError
from openai.types.chat import ParsedChatCompletion
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tqdm.asyncio import tqdm

from constants.data import ACTIVITY_LABEL_VAR, ACTIVITY_PRECISION_VARS
from constants.vector_db import MAX_CONCURRENCY
from utils.data import fetch_mapping, get_file_system

logger = logging.getLogger(__name__)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exc, APIStatusError):
        return exc.status_code in {408, 429, 500, 502, 503, 504}
    return False


class EncodeStrategy(ABC):
    """
    Abstract base class for encoding strategies (RAG or CAG).
    Generation is delegated to an OpenAI-compatible API (llm.lab).
    """

    def __init__(
        self,
        generation_model: str = "gemma4-31b",
    ):
        self.fs = get_file_system()
        self.mapping = fetch_mapping()
        self.generation_model = generation_model

        self.client = AsyncOpenAI(
            base_url=os.environ["LLMLAB_URL"],
            api_key=os.environ["LLMLAB_API_KEY"],
            timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        )

        self.response_format: Optional[BaseModel] = None
        self.sampling_params: Dict[str, Any] = {}
        self.token_stats: Dict[str, float] = {}

    @abstractmethod
    def get_prompts(self, data: pd.DataFrame, load_prompts_from_file: bool = False) -> List[List[Dict]]:
        """Each strategy defines how it builds prompts."""
        pass

    @property
    @abstractmethod
    def output_path(self) -> str:
        """Each strategy defines its output path."""
        pass

    def postprocess_results(self, df):
        """Default postprocess: remove dots from 'nace2025'."""
        df["nace2025"] = df["nace2025"].str.replace(".", "", regex=False)
        return df

    def save_results(self, df: pd.DataFrame, third: int) -> str:
        """Save the results to the specified output path."""
        output_path = self.output_path.format(third=f"{third}" if third else "", i="{i}")

        pq.write_to_dataset(
            pa.Table.from_pandas(df),
            root_path="/".join(output_path.split("/")[:-1]),
            basename_template=output_path.split("/")[-1],
            existing_data_behavior="overwrite_or_ignore",
            filesystem=self.fs,
        )
        return output_path.format(i=0)

    def _format_activity_description(self, row: Any) -> str:
        """
        Format the activity description from the row data, appending the
        precision fields declared in ACTIVITY_PRECISION_VARS (with their label).
        Column names come from constants.data — none are hardcoded here.
        """
        label = row.get(ACTIVITY_LABEL_VAR)
        activity = label.lower() if label.isupper() else label

        for var, prefix in ACTIVITY_PRECISION_VARS.items():
            value = row.get(var)
            if value:
                activity += f"\n{prefix} : {value.lower()}"

        return activity

    @retry(
        stop=stop_after_attempt(6),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception(_is_retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    async def _call_once(self, messages: List[Dict]) -> ParsedChatCompletion:
        # `chat.completions.parse` enforces the Pydantic schema server-side via strict
        # JSON-Schema and parses the response into a typed model under `.parsed`.
        return await self.client.chat.completions.parse(
            model=self.generation_model,
            messages=messages,
            logprobs=True,
            response_format=self.response_format,
            **self.sampling_params,
        )

    async def call_llm(
        self,
        messages_list: List[List[Dict]],
        *,
        max_concurrency: Optional[int] = None,
        error_policy: str = "store_none",
    ) -> List[Optional[ParsedChatCompletion]]:
        """
        Run chat completions concurrently with bounded concurrency, retries, and a progress bar.

        Args:
            messages_list: list of conversations.
            max_concurrency: override the default concurrency for this call.
            error_policy: "raise" → re-raise on failure ; "store_none" → return None for failed items
                          (default) ; "store_exception" → return the Exception object.
        """
        if error_policy not in {"raise", "store_none", "store_exception"}:
            raise ValueError(f"Unknown error_policy: {error_policy!r}")

        semaphore = asyncio.Semaphore(max_concurrency or MAX_CONCURRENCY)

        async def _one(messages: List[Dict]) -> Union[ParsedChatCompletion, BaseException]:
            async with semaphore:
                try:
                    return await self._call_once(messages)
                except BaseException as exc:
                    return exc

        results: List[Any] = await tqdm.gather(
            *(_one(m) for m in messages_list),
            desc="LLM generation",
        )

        failed = [(i, r) for i, r in enumerate(results) if isinstance(r, BaseException)]
        for idx, exc in failed:
            logger.error("LLM call %d failed: %s", idx, exc)
        logger.info("LLM generation: %d ok, %d failed", len(results) - len(failed), len(failed))

        self.token_stats = self._compute_token_stats(results)

        if error_policy == "raise" and failed:
            raise failed[0][1]
        if error_policy == "store_none":
            results = [None if isinstance(r, BaseException) else r for r in results]
        return results

    def _compute_token_stats(self, results: List[Any]) -> Dict[str, float]:
        """Aggregate completion/prompt token counts across the successful calls
        and log a one-line summary. Stats are based only on responses that came
        back (truncated/errored calls are not represented since their usage is
        not available via the SDK's ParsedChatCompletion)."""
        usages = [r.usage for r in results if hasattr(r, "usage") and r.usage is not None]
        if not usages:
            return {}

        completion = [u.completion_tokens for u in usages]
        prompt = [u.prompt_tokens for u in usages]
        # `*_sum` and `n_calls` let callers aggregate stats across multiple
        # call_llm invocations (e.g. one per batch) — see utils.batch.merge_token_stats.
        stats = {
            "completion_tokens_mean": sum(completion) / len(completion),
            "completion_tokens_max":  max(completion),
            "completion_tokens_min":  min(completion),
            "completion_tokens_sum":  sum(completion),
            "prompt_tokens_mean":     sum(prompt) / len(prompt),
            "prompt_tokens_max":      max(prompt),
            "prompt_tokens_min":      min(prompt),
            "prompt_tokens_sum":      sum(prompt),
            "n_calls":                len(usages),
        }
        logger.info(
            "Token usage over %d successful calls — "
            "completion: mean=%.1f, max=%d, min=%d | prompt: mean=%.1f, max=%d, min=%d",
            len(usages),
            stats["completion_tokens_mean"], stats["completion_tokens_max"], stats["completion_tokens_min"],
            stats["prompt_tokens_mean"],     stats["prompt_tokens_max"],     stats["prompt_tokens_min"],
        )
        return stats

    def _process_output(self, response: Union[ParsedChatCompletion, BaseException, None]) -> BaseModel:
        """Extract the parsed BaseModel from a ParsedChatCompletion and attach a confidence."""
        if response is None or isinstance(response, BaseException):
            return self.response_format(codable=False, nace2025=None, confidence=0.0)

        parsed: BaseModel = response.choices[0].message.parsed

        # JSON-Schema cannot express the cross-field constraint "codable=True ⇒ nace2025 not null",
        # so the LLM may return that combination. Treat it as not codable.
        if parsed.nace2025 is None:
            parsed.codable = False
            parsed.confidence = 0.0
            return parsed

        logprobs_obj = response.choices[0].logprobs
        token_logprobs = getattr(logprobs_obj, "content", None) if logprobs_obj else None
        parsed.confidence = self._compute_confidence(token_logprobs, parsed.nace2025) if token_logprobs else 0.0
        return parsed

    @staticmethod
    def _compute_confidence(token_logprobs: List[Any], target: str) -> float:
        """
        Find a window of consecutive tokens whose concatenation contains `target`,
        then return exp(mean(window_logprobs)). Returns 0.0 if no match.
        """
        if not target:
            return 0.0

        tokens = [item.token for item in token_logprobs]
        logprobs = [item.logprob for item in token_logprobs]

        for i in range(len(tokens)):
            concat = ""
            for j in range(i, len(tokens)):
                concat += tokens[j]
                normalized = concat.replace(" ", "").replace('"', "").replace(".", "")
                if target.replace(".", "") in normalized:
                    window = logprobs[i : j + 1]
                    return math.exp(sum(window) / len(window))
        return 0.0

    def process_outputs(self, outputs: List[Optional[ParsedChatCompletion]]) -> pd.DataFrame:
        """Process a list of LLM outputs into a structured DataFrame."""
        records = [self._process_output(output).model_dump() for output in outputs]
        df = pd.DataFrame.from_records(records)
        return self.postprocess_results(df)
