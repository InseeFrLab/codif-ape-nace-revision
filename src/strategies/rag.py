import asyncio
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
from utils.strategies import chunked
from vector_db.loading import get_retriever

from .base import EncodeStrategy

logger = logging.getLogger(__name__)

# Deal with Qdrant API temporary disconnections
import random
import httpx
import inspect
from qdrant_client.http.exceptions import ResponseHandlingException
MAX_RETRIES = 5
BASE_DELAY = 2  # secondes


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
        self, data: pd.DataFrame, load_prompts_from_file: bool = False, top_k: int = 5, batch_size: int = 128,
    ) -> List[List[Dict]]:

        if load_prompts_from_file:
            return load_prompts(self.prompt_name, self.prompt_label)

        rows = data.to_dict(orient="records")

        # 1. Construire toutes les activity descriptions
        activities = [self._format_activity_description(row) for row in rows]

        # 2. Compiler toutes les queries
        queries = [
            self.prompt_template_retriever.compile(activity_description=activity)
            for activity in activities
        ]

        # 3. Embedding en batch (beaucoup plus rapide qu'un par un)
        embeddings = await self.db.embeddings.aembed_documents(queries)

        # 4. Construire une requête batch pour Qdrant
        search_requests = [
            SearchRequest(
                vector=NamedVector(name=self.db.vector_name, vector=vec),
                limit=top_k,
                with_payload=True
            )
            for vec in embeddings
        ]

        num_chunks = (len(search_requests) + batch_size - 1) // batch_size  # Calcul du nombre de chunks

        # 5. Requête batch au client Qdrant (un seul appel réseau par batch!)
        results = []
        for chunk in tqdm(chunked(search_requests, batch_size), total=num_chunks*batch_size, desc="Processing Qdrant requests"):
            res = self.db.client.search_batch(
                collection_name=self.collection_name,
                requests=chunk,
            )
            results.extend(res)

        # 6. Construire les prompts avec les docs retrouvés
        prompts = []
        for row, activity, docs in zip(rows, activities, results):
            # docs = List[ScoredPoint] renvoyés par Qdrant
            proposed_codes, list_codes = self._format_documents([
                Document(
                    page_content=d.payload["page_content"],
                    metadata=d.payload.get("metadata", {}),
                )
                for d in docs
            ])
            prompt = self.prompt_template.compile(
                activity=activity,
                proposed_codes=proposed_codes,
                list_proposed_codes=list_codes,
            )
            prompts.append(prompt)

        # 7. Sauvegarder et retourner
        save_prompts(prompts, self.prompt_name, self.prompt_label)
        return prompts

    @property
    def output_path(self) -> str:
        """
        Returns a Parquet output path template including model name and timestamp.
        Placeholders {i} and {third} must be filled later.
        """
        date = datetime.now().strftime("%Y-%m-%d--%H:%M")
        return f"{URL_SIRENE4_AMBIGUOUS_RAG}/{self.generation_model}/part-{{i}}-{{third}}--{date}.parquet"

    async def _retry_with_backoff(self, coro, *args, retries=5, backoff_in_seconds=1, **kwargs):
        if not callable(coro):
            raise TypeError(f"Expected a callable coroutine, got {type(coro)}")
        if not inspect.iscoroutinefunction(coro):
            raise TypeError(f"Expected an async coroutine function, got {coro}")

        for attempt in range(retries):
            try:
                return await coro(*args, **kwargs)
            except Exception as e:
                wait_time = backoff_in_seconds * (2 ** attempt)
                print(f"Attempt {attempt+1} failed: {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
        raise RuntimeError(f"Failed after {retries} retries")

    async def create_prompt(self, row: Dict[str, Any], top_k: int = 5) -> List[Dict]:
        """
        Creates a prompt from a data row by retrieving similar documents.

        Args:
            row: A dictionary representing a single activity description row.
            top_k: Number of top documents to retrieve based on similarity.

        Returns:
            Filled prompt fields ready to be used for generation.
        """
        # try:
        async with self.semaphore:
            activity = self._format_activity_description(row)
            query = self.prompt_template_retriever.compile(
                activity_description=activity,
            )
            docs = await self._retry_with_backoff(self.db.asimilarity_search, query, k=top_k)
            # docs = await self.db.asimilarity_search(query, k=top_k)
            proposed_codes, list_codes = self._format_documents(docs)
        # except Exception as e:
        #     print("=====row=======")
        #     print(row)
        #     print("=====query=======")
        #     print(query)
        #     raise e

        return self.prompt_template.compile(
            activity=activity,
            proposed_codes=proposed_codes,
            list_proposed_codes=list_codes,
        )

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
