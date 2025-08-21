import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from langchain.schema import Document
from langfuse import Langfuse
from pydantic import BaseModel, Field, model_validator
from tqdm.asyncio import tqdm
from vllm.sampling_params import GuidedDecodingParams, SamplingParams
from qdrant_client.http.models import SearchRequest
from qdrant_client.http.models import NamedVector
from math import ceil

from constants.llm import (
    MAX_NEW_TOKEN,
    TEMPERATURE,
)
from constants.paths import URL_SIRENE4_AMBIGUOUS_RAG
from constants.vector_db import MAX_CONCURRENCY
from utils.data import load_prompts, save_prompts
from vector_db.loading import get_retriever

from .base import EncodeStrategy

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    """Represents the RAG response model for classification code assignment."""

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
        prompt_name: str = "rag-classifier",
        prompt_label: str = "production",
        collection_name: str = os.getenv("COLLECTION_NAME"),
        reranker_model: str = None,
    ):
        super().__init__(generation_model)
        self.response_format = RAGResponse
        self.reranker_model = reranker_model
        self.collection_name = collection_name
        self.db = get_retriever(collection_name, self.reranker_model)
        self.prompt_name = prompt_name
        self.prompt_label = prompt_label
        self.prompt_template = Langfuse().get_prompt(self.prompt_name, label=self.prompt_label)
        self.prompt_template_retriever = Langfuse().get_prompt("retriever", label="production")
        self.sampling_params = SamplingParams(
            max_tokens=MAX_NEW_TOKEN,
            temperature=TEMPERATURE,
            seed=2025,
            logprobs=1,
            guided_decoding=GuidedDecodingParams(json=self.response_format.model_json_schema()),
        )
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)  # Max concurrency for API calls

    async def get_prompts(
        self,
        data: pd.DataFrame,
        load_prompts_from_file: bool = False,
        top_k: int = 5,
        batch_size: int = 128,
    ) -> List[List[Dict]]:
        """
        Generate prompts for each row of the dataframe by retrieving
        relevant documents from Qdrant and formatting them as chat messages.

        Args:
            data (pd.DataFrame): Input data containing activity descriptions.
            load_prompts_from_file (bool): If True, load prompts from disk instead of recomputing.
            top_k (int): Number of documents to retrieve per query.
            batch_size (int): Number of queries per batch when calling Qdrant.

        Returns:
            List[List[Dict]]: A list of conversations, one per row in the dataframe.
        """
        if load_prompts_from_file:
            return load_prompts(self.prompt_name, self.prompt_label)

        if data.empty:
            raise ValueError("Input data is empty")

        activities, queries = self._prepare_queries(data)

        # Embedding queries
        embeddings = await self.db.embeddings.aembed_documents(queries)

        # Batch search in Qdrant
        results = self._search_qdrant(embeddings, top_k, batch_size)

        # Build prompts from retrieved docs
        prompts = self._build_prompts(activities, results)

        # Persist prompts for later reuse
        save_prompts(prompts, self.prompt_name, self.prompt_label)
        return prompts

    def _prepare_queries(self, data: pd.DataFrame):
        """
        Convert dataframe rows into activity descriptions and queries.

        Args:
            data (pd.DataFrame): Input dataframe.

        Returns:
            tuple: (activities, queries)
                - activities (list[str]): Formatted activity descriptions.
                - queries (list[str]): Queries compiled for embeddings.
        """
        rows = data.to_dict(orient="records")
        activities = [self._format_activity_description(row) for row in rows]
        queries = [self.prompt_template_retriever.compile(activity_description=a) for a in activities]
        return activities, queries

    def _search_qdrant(
        self,
        embeddings: List[List[float]],
        top_k: int,
        batch_size: int,
    ):
        """
        Run batched search requests in Qdrant.

        Args:
            embeddings (List[List[float]]): List of embedding vectors.
            top_k (int): Number of results per query.
            batch_size (int): Number of queries per request batch.

        Returns:
            List[List[ScoredPoint]]: Search results grouped by query.
        """
        search_requests = [
            SearchRequest(
                vector=NamedVector(name=self.db.vector_name, vector=vec),
                limit=top_k,
                with_payload=True,
            )
            for vec in embeddings
        ]

        results = []
        num_chunks = ceil(len(search_requests) / batch_size)
        for chunk in tqdm(
            self._chunked(search_requests, batch_size),
            total=num_chunks,
            desc="Processing Qdrant requests",
            unit="batch",
        ):
            res = self.db.client.search_batch(
                collection_name=self.collection_name,
                requests=chunk,
            )
            results.extend(res)

        return results

    def _build_prompts(
        self,
        activities: List[str],
        results,
    ) -> List[List[Dict]]:
        """
        Build final prompts from retrieved documents and activities.

        Args:
            activities (List[str]): Activity descriptions.
            results (List[List[ScoredPoint]]): Search results from Qdrant.

        Returns:
            List[List[Dict]]: Final prompts as chat message dictionaries.
        """
        prompts: List[List[Dict]] = []
        for activity, docs in zip(activities, results):
            langchain_docs = [
                Document(
                    page_content=d.payload["page_content"],
                    metadata=d.payload.get("metadata", {}),
                )
                for d in docs
            ]
            proposed_codes, list_codes = self._format_documents(langchain_docs)

            convo: List[Dict] = self.prompt_template.compile(
                activity=activity,
                proposed_codes=proposed_codes,
                list_proposed_codes=list_codes,
            )
            prompts.append(convo)
        return prompts

    @staticmethod
    def _chunked(seq, size):
        """Split a list into chunks of given size."""
        for i in range(0, len(seq), size):
            yield seq[i : i + size]

    @property
    def output_path(self) -> str:
        """
        Returns a Parquet output path template including model name and timestamp.
        Placeholders {i} and {third} must be filled later.
        """
        date = datetime.now().strftime("%Y-%m-%d--%H:%M")
        return f"{URL_SIRENE4_AMBIGUOUS_RAG}/{self.generation_model}/part-{{i}}-{{third}}--{date}.parquet"

    def _format_documents(self, docs: List[Document]) -> Tuple[str, str]:
        """
        Formats retrieved documents into two string representations.

        Args:
            docs: A list of LangChain Document objects with metadata.

        Returns:
            A tuple of:
                - A formatted string containing document content blocks.
                - A comma-separated list of classification codes.
        """
        proposed_codes = "\n\n".join(f"========\n{doc.page_content}" for doc in docs)
        list_codes = ", ".join(f"'{doc.metadata['code']}'" for doc in docs)
        return proposed_codes, list_codes
