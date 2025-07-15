import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from langchain.schema import Document
from langfuse import Langfuse
from pydantic import BaseModel, Field, model_validator
from tqdm.asyncio import tqdm
from vllm.sampling_params import GuidedDecodingParams, SamplingParams

from constants.llm import (
    MAX_NEW_TOKEN,
    TEMPERATURE,
)
from constants.paths import URL_SIRENE4_AMBIGUOUS_RAG
from utils.data import load_prompts, save_prompts
from vector_db.loading import get_retriever

from .base import EncodeStrategy

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    """Represents a response model for classification code assignment."""

    codable: bool = Field(
        description="""True if enough information is provided to decide classification code, False otherwise."""
    )

    nace2025: Optional[str] = Field(
        description="""NACE 2025 classification code Empty if codable=False.""",
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


class RAGStrategy(EncodeStrategy):
    def __init__(
        self,
        generation_model: str = "Qwen/Qwen2.5-0.5B",
        reranker_model: str = None,
    ):
        super().__init__(generation_model)
        self.response_format = RAGResponse
        self.reranker_model = reranker_model
        self.db = get_retriever(os.getenv("COLLECTION_NAME"), self.reranker_model)
        self.prompt_template = Langfuse().get_prompt("rag-classifier", label="production")
        self.prompt_template_retriever = Langfuse().get_prompt("retriever", label="production")
        self.sampling_params = SamplingParams(
            max_tokens=MAX_NEW_TOKEN,
            temperature=TEMPERATURE,
            seed=2025,
            logprobs=1,
            guided_decoding=GuidedDecodingParams(json=self.response_format.model_json_schema()),
        )

    async def get_prompts(self, data: pd.DataFrame, load_prompts_from_file: bool = False) -> List[List[Dict]]:
        if load_prompts_from_file:
            prompts = load_prompts()
        else:
            tasks = [self.create_prompt(row) for row in data.to_dict(orient="records")]
            prompts = await tqdm.gather(*tasks)
            save_prompts(prompts)
        return prompts

    @property
    def output_path(self):
        date = datetime.now().strftime("%Y-%m-%d--%H:%M")
        return f"{URL_SIRENE4_AMBIGUOUS_RAG}/{self.generation_model}/part-{{i}}-{{third}}--{date}.parquet"

    # TODO: implement a method that create prompt and saves it to s3 in parquet and load it back when specified
    async def create_prompt(self, row: Dict[str, Any], top_k: int = 5) -> Dict:
        activity = self._format_activity_description(row)
        query = self.prompt_template_retriever.compile(
            activity_description=activity,
        )
        docs = await self.db.asimilarity_search(query, k=top_k)
        proposed_codes, list_codes = self._format_documents(docs)

        return self.prompt_template.compile(
            activity=activity,
            proposed_codes=proposed_codes,
            list_proposed_codes=list_codes,
        )

    def _format_documents(self, docs: List[Document]) -> Tuple[str, str]:
        proposed_codes = "\n\n".join(f"========\n{doc.page_content}" for doc in docs)
        list_codes = ", ".join(f"'{doc.metadata['code']}'" for doc in docs)
        return proposed_codes, list_codes
