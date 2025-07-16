import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from pydantic import BaseModel, TypeAdapter, ValidationError
from vllm import LLM
from vllm.outputs import RequestOutput

from constants.llm import (
    MODEL_TO_ARGS,
)
from utils.data import fetch_mapping, get_file_system

logger = logging.getLogger(__name__)


class EncodeStrategy(ABC):
    """
    Abstract base class for your encoding strategies (RAG or CAG).
    Provides common LLM handling and postprocessing hooks.
    """

    def __init__(
        self,
        generation_model: str = "Qwen/Qwen2.5-0.5B",
    ):
        self.fs = get_file_system()
        self.mapping = fetch_mapping()
        self.generation_model = generation_model

        model_args = MODEL_TO_ARGS.get(self.generation_model, {}).copy()

        # add mistral specific args
        if self.generation_model.startswith("mistralai"):
            mistral_args = {
                "tokenizer_mode": "mistral",
                "config_format": "mistral",
                "load_format": "mistral",
            }
            model_args.update(mistral_args)

        self.llm = LLM(
            model=self.generation_model,
            **model_args,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.response_format: Optional[BaseModel] = None

    @abstractmethod
    def get_prompts(self, data: pd.DataFrame, load_prompts_from_file: bool = False) -> List[List[Dict]]:
        """
        Each strategy defines how it builds prompts.
        """
        pass

    @property
    @abstractmethod
    def output_path(self) -> str:
        """
        Each strategy defines its output path.
        """
        pass

    def postprocess_results(self, df):
        """
        Default postprocess: remove dots from 'nace2025'.
        """
        df["nace2025"] = df["nace2025"].str.replace(".", "", regex=False)
        return df

    def save_results(self, df: pd.DataFrame, third: int) -> str:
        """
        Save the results to the specified output path.
        """
        output_path = self.output_path.format(third=f"{third}" if third else "", i="{i}")

        pq.write_to_dataset(
            pa.Table.from_pandas(df),
            root_path="/".join(output_path.split("/")[:-1]),
            partition_cols=["codable"],
            basename_template=output_path.split("/")[-1],
            existing_data_behavior="overwrite_or_ignore",
            filesystem=self.fs,
        )
        return output_path.format(i=0)

    def _format_activity_description(self, row: Any) -> str:
        """
        Format the activity description from the row data.
        """
        activity = row.get("libelle").lower() if row.get("libelle").isupper() else row.get("libelle")

        if row.get("activ_sec_agri_et"):
            activity += f"\nPrécisions sur l'activité agricole : {row.get('activ_sec_agri_et').lower()}"

        if row.get("activ_nat_lib_et"):
            activity += f"\nAutre nature d'activité : {row.get('activ_nat_lib_et').lower()}"

        return activity

    def call_llm(self, messages: List[List[Dict]], sampling_params: Any) -> List[RequestOutput]:
        return self.llm.chat(messages, sampling_params=sampling_params)

    def _parse_content(self, content: str) -> Optional[BaseModel]:
        try:
            return TypeAdapter(self.response_format).validate_json(content)
        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            return None

    def _process_output(self, output: RequestOutput) -> BaseModel:
        """
        Process the outputs from the LLM and return a list of BaseModel objects.
        """
        parsed = self._parse_content(output.outputs[0].text)
        if parsed is None or getattr(parsed, "nace2025", None) is None:
            return self.response_format(codable=False, nace2025=None, confidence=0.0)

        # We get the tokenized predicted NACE2025 code
        target_ids = (
            self.tokenizer(parsed.nace2025).get("input_ids")
            if isinstance(self.tokenizer(parsed.nace2025), dict)
            else self.tokenizer(parsed.nace2025).input_ids  # We need to deal with custom tokenizer (MistralAI...)
        )
        logprobs_tensor = self.extract_sequence_logprobs(output.outputs[0].logprobs, target_ids)

        # We set the confidence score based on the logprobs
        parsed.confidence = torch.exp(logprobs_tensor).mean().item()
        return parsed

    def extract_sequence_logprobs(self, logprobs: List[Dict[int, Any]], target_ids: List[int]) -> torch.Tensor:
        """
        Extracts logprobs for the exact target_ids sequence from the list of logprobs.

        Args:
            logprobs: List of dicts with {token_id: Logprob}.
            target_ids: The exact sequence of token IDs you want to find.

        Returns:
            Tensor of logprobs for the matched sequence, or empty tensor if not found.
        """
        ids_sequence = [list(tok.keys())[0] if tok else None for tok in logprobs]
        sequence_length = len(target_ids)

        for i in range(len(ids_sequence) - sequence_length + 1):
            window_ids = ids_sequence[i : i + sequence_length]
            if window_ids == target_ids:
                # Exact match found, extract corresponding logprobs
                window_logprobs = [list(logprobs[i + j].values())[0].logprob for j in range(sequence_length)]
                return torch.tensor(window_logprobs)

        # If no match found, return empty or fill with -inf
        return torch.full((sequence_length,), float("-inf"))

    def process_outputs(self, outputs: List[RequestOutput]) -> pd.DataFrame:
        records = [self._process_output(output).model_dump() for output in outputs]
        df = pd.DataFrame.from_records(records)
        return self.postprocess_results(df)
