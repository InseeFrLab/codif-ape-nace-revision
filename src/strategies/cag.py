import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langfuse import Langfuse
from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm

from constants.data import NACE08_VAR
from constants.llm import MAX_NEW_TOKEN_FAST, MAX_NEW_TOKEN_THINKING, TEMPERATURE
from constants.paths import URL_PROMPTS_CAG
from utils.data import get_file_system, prompts_to_df

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


class CAGStrategy(EncodeStrategy):
    def __init__(
        self,
        generation_model: str = "gemma4-31b",
        prompt_name: str = "cag-classifier",
        prompt_label: str = "production",
        thinking: bool = False,
        max_new_tokens: Optional[int] = None,
    ):
        super().__init__(generation_model)
        self.response_format = CAGResponse
        self.prompt_template = Langfuse().get_prompt(prompt_name, label=prompt_label)

        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_TOKEN_THINKING if thinking else MAX_NEW_TOKEN_FAST
        self.thinking = thinking
        self.sampling_params = {
            "max_tokens": max_new_tokens,
            "temperature": TEMPERATURE,
            "seed": 2025,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": thinking}},
        }
        self.prompt_name = prompt_name
        self.prompt_label = prompt_label

    async def get_prompts(
        self, data: pd.DataFrame, load_prompts_from_file: bool = False,
        top_k: int = 5, save: bool = False,
    ) -> List[List[Dict]]:
        tasks = [self.create_prompt(row) for row in data.to_dict(orient="records")]
        prompts = await tqdm.gather(*tasks)
        if save:
            self._save_prompts(prompts)
        return prompts

    def postprocess_results(self, df):
        # Apply the base postprocessing first
        df = super().postprocess_results(df)
        # Then apply specific CAG postprocessing
        df["nace08_valid"] = df["nace08_valid"].fillna("undefined").astype(str)
        return df

    async def create_prompt(self, row: Dict[str, Any]) -> List[Dict]:
        activity = self._format_activity_description(row)
        apet = row.get(NACE08_VAR)
        nace08 = f"{apet[:2]}.{apet[2:]}"
        nace_old, proposed_codes, list_codes = self._format_documents(nace08)

        prompts = self.prompt_template.compile(
            activity=activity,
            nace_old=nace08,
            proposed_codes=proposed_codes,
            list_proposed_codes=list_codes,
        )
        return prompts

    def _save_prompts(
        self,
        prompts: List[List[Dict]],
    ) -> None:
        """Save prompts to a Parquet file.

        Args:
            prompts: List of conversations to save
            prompt_name: Name of the Langfuse prompt
            prompt_label: Label for the Langfuse prompt
        """
        fs = get_file_system()
        prompts_df: pd.DataFrame = prompts_to_df(prompts)
        prompts_df.to_parquet(
            URL_PROMPTS_CAG.format(prompt_name=self.prompt_name, prompt_label=self.prompt_label),
            filesystem=fs,
        )


    def _format_documents(self, nace08: str) -> Tuple[str, str, str]:
        """Format documents related to NACE classification codes.

        Args:
            nace08: The NACE08 code to format documents for.

        Returns:
            A tuple containing:
            - nace_old: Formatted string of the NACE08 code and label
            - proposed_codes: Formatted string of proposed NACE2025 codes with their details
            - list_codes: Comma-separated string of proposed NACE2025 codes
        """
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
