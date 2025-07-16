import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langfuse import Langfuse
from pydantic import BaseModel, Field, model_validator
from tqdm.asyncio import tqdm
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

from constants.llm import (
    MAX_NEW_TOKEN,
    TEMPERATURE,
)
from constants.paths import URL_SIRENE4_AMBIGUOUS_CAG

from .base import EncodeStrategy

logger = logging.getLogger(__name__)


class CAGResponse(BaseModel):
    """Represents a response model for classification code assignment."""

    codable: bool = Field(
        description="""True if enough information is provided to decide classification code, False otherwise."""
    )

    nace2025: Optional[str] = Field(
        description="""NACE 2025 classification code. Empty if codable=False.""",
        default=None,
    )

    nace08_valid: Optional[bool] = Field(
        description="""True if the NACE08 classification seems valid with the description of the activity, False otherwise.""",
        default=None,
    )

    confidence: Optional[float] = Field(
        description="""Confidence score for the NACE2025 code, based on log probabilities. Rounded to 2 decimal places maximum.""",
        default=0.0,
    )

    @model_validator(mode="after")
    def check_nace2025_if_codable(self) -> BaseModel:
        if self.codable and not self.nace2025:
            raise ValueError("If codable=True, then nace2025 must not be None or empty.")
        return self


class CAGStrategy(EncodeStrategy):
    def __init__(
        self,
        generation_model: str = "Qwen/Qwen2.5-0.5B",
        reranker_model: str = None,
    ):
        super().__init__(generation_model)
        self.response_format = CAGResponse
        self.prompt_template = Langfuse().get_prompt("cag-classifier", label="production")
        self.sampling_params = SamplingParams(
            max_tokens=MAX_NEW_TOKEN,
            temperature=TEMPERATURE,
            seed=2025,
            logprobs=1,
            guided_decoding=GuidedDecodingParams(json=self.response_format.model_json_schema()),
        )

    async def get_prompts(self, data: pd.DataFrame, load_prompts_from_file: bool = False) -> List[List[Dict]]:
        tasks = [self.create_prompt(row) for row in data.to_dict(orient="records")]
        return await tqdm.gather(*tasks)

    @property
    def output_path(self):
        date = datetime.now().strftime("%Y-%m-%d--%H:%M")
        return f"{URL_SIRENE4_AMBIGUOUS_CAG}/{self.generation_model}/part-{{i}}-{{third}}--{date}.parquet"

    def postprocess_results(self, df):
        # Apply the base postprocessing first
        df = super().postprocess_results(df)
        # Then apply specific CAG postprocessing
        df["nace08_valid"] = df["nace08_valid"].fillna("undefined").astype(str)
        return df

    async def create_prompt(self, row: Dict[str, Any], top_k: int = 5) -> List[Dict]:
        activity = self._format_activity_description(row)
        nace08 = f"{row.get('apet_finale')[:2]}.{row.get('apet_finale')[2:]}"
        nace_old, proposed_codes, list_codes = self._format_documents(nace08)

        return self.prompt_template.compile(
            activity=activity,
            nace_old=nace08,
            proposed_codes=proposed_codes,
            list_proposed_codes=list_codes,
        )

    def _format_documents(self, nace08: str) -> Tuple[str, str, str]:
        nace2025_codes = next((m.naf2025 for m in self.mapping if m.code == nace08))
        nace08_code = next((m for m in self.mapping if m.code == nace08))

        nace_old = "\n\n".join([f"{c.code}: {c.label}" for c in [nace08_code]])
        list_codes = ", ".join([f"'{c.code}'" for c in nace2025_codes])
        proposed_codes = self.format_code(nace2025_codes)
        return nace_old, proposed_codes, list_codes

    def format_code(self, codes: list) -> str:
        return "\n\n".join([f"{c.code}: {c.label}\n{self.extract_info(c)}" for c in codes])

    def extract_info(self, code) -> str:
        info = [getattr(code, attr) for attr in ["include", "not_include", "notes"] if getattr(code, attr, None)]
        return "\n\n".join(info) if info else ""
