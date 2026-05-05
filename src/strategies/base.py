import asyncio
import logging
import math
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import BaseModel, TypeAdapter, ValidationError

from constants.vector_db import MAX_CONCURRENCY
from utils.data import fetch_mapping, get_file_system

logger = logging.getLogger(__name__)


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
        )
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

        self.response_format: Optional[BaseModel] = None
        self.sampling_params: Dict[str, Any] = {}

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
        Format the activity description from the row data.
        Adds precisions in case of agricultural activity.
        """
        activity = row.get("libelle").lower() if row.get("libelle").isupper() else row.get("libelle")

        if row.get("activ_sec_agri_et"):
            activity += f"\nPrécisions sur l'activité agricole : {row.get('activ_sec_agri_et').lower()}"

        if row.get("activ_nat_lib_et_1"):
            activity += f"\nAutre nature d'activité : {row.get('activ_nat_lib_et_1').lower()}"

        if row.get("lib_cj"):
            activity += f"\nCatégorie juridique de l'établissement : {row.get('lib_cj').lower()}"

        return activity

    async def call_llm(self, messages_list: List[List[Dict]]) -> List[ChatCompletion]:
        """
        Run all chat completions concurrently against the llm.lab API,
        rate-limited by self.semaphore.
        """
        response_format_arg = {
            "type": "json_schema",
            "json_schema": {
                "name": self.response_format.__name__,
                "schema": self.response_format.model_json_schema(),
                "strict": True,
            },
        }

        async def _one_call(messages: List[Dict]) -> ChatCompletion:
            async with self.semaphore:
                return await self.client.chat.completions.create(
                    model=self.generation_model,
                    messages=messages,
                    logprobs=True,
                    response_format=response_format_arg,
                    **self.sampling_params,
                )

        return await asyncio.gather(*(_one_call(m) for m in messages_list))

    def _parse_content(self, content: str) -> Optional[BaseModel]:
        try:
            return TypeAdapter(self.response_format).validate_json(content)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return None

    def _process_output(self, response: ChatCompletion) -> BaseModel:
        """Parse a single ChatCompletion into a response_format BaseModel with confidence."""
        content = response.choices[0].message.content
        parsed = self._parse_content(content)
        if parsed is None or getattr(parsed, "nace2025", None) is None:
            return self.response_format(codable=False, nace2025=None, confidence=0.0)

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

    def process_outputs(self, outputs: List[ChatCompletion]) -> pd.DataFrame:
        """Process a list of LLM outputs into a structured DataFrame."""
        records = [self._process_output(output).model_dump() for output in outputs]
        df = pd.DataFrame.from_records(records)
        return self.postprocess_results(df)
