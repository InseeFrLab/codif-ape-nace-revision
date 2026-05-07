import logging
import os
from dataclasses import dataclass
from typing import Iterable, List, Mapping

from openai import AsyncOpenAI, OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.exceptions import UnexpectedResponse

logger = logging.getLogger(__name__)

# Number of texts sent per embeddings request to the LLM Lab API.
EMBED_BATCH_SIZE = 16
# Number of points sent per Qdrant upsert call.
UPSERT_BATCH_SIZE = 16


def _vector_to_api_model(vector_name: str) -> str:
    """Convert a Qdrant named-vector key into the slug expected by the LLM Lab
    embeddings API. Example: 'Qwen/Qwen3-Embedding-8B' -> 'qwen3-embedding-8b'.
    The heuristic strips the org/repo prefix and lowercases the result."""
    return vector_name.rsplit("/", 1)[-1].lower()


class Embeddings:
    """Thin wrapper around the OpenAI-compatible LLM Lab API exposing
    embed_documents (sync) and aembed_documents (async)."""

    def __init__(self, model: str, base_url: str, api_key: str, batch_size: int = EMBED_BATCH_SIZE):
        self.model = model
        self.batch_size = batch_size
        self._sync = OpenAI(base_url=base_url, api_key=api_key)
        self._async = AsyncOpenAI(base_url=base_url, api_key=api_key)

    def _explain(self, exc: Exception) -> RuntimeError:
        """Wrap an OpenAI/HTTP exception with a hint about the model slug,
        the most common source of opaque 500s on this endpoint."""
        return RuntimeError(
            f"Embedding API call failed (model={self.model!r}, "
            f"base_url={self._sync.base_url!s}). "
            f"If the server rejects this slug, override it via the "
            f"EMBEDDING_MODEL_API_NAME env var (e.g. 'qwen3-embedding-8b'). "
            f"Underlying error: {type(exc).__name__}: {exc}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            try:
                resp = self._sync.embeddings.create(model=self.model, input=chunk)
            except Exception as e:
                raise self._explain(e) from e
            out.extend(d.embedding for d in resp.data)
        return out

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            try:
                resp = await self._async.embeddings.create(model=self.model, input=chunk)
            except Exception as e:
                raise self._explain(e) from e
            out.extend(d.embedding for d in resp.data)
        return out

    def healthcheck(self) -> int:
        """One-shot embedding call used to fail fast on misconfiguration.
        Returns the embedding dimension on success."""
        resp = self._sync.embeddings.create(model=self.model, input=["healthcheck"])
        return len(resp.data[0].embedding)


@dataclass
class VectorDB:
    """Bundle of the Qdrant client, embedding client, named-vector key,
    and collection name used by the retriever."""

    client: QdrantClient
    embeddings: Embeddings
    vector_name: str
    collection_name: str


def get_qdrant_client() -> QdrantClient:
    """Initialize and return the Qdrant client."""
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        port=443,
        https=True,
    )


def get_embedding_model_name(client: QdrantClient, collection_name: str) -> str:
    """Retrieve the named-vector key (i.e. embedding model name) from a Qdrant collection."""
    try:
        info = client.get_collection(collection_name=collection_name)
        return next(iter(info.config.params.vectors.keys()))
    except Exception as e:
        raise RuntimeError(f"Error retrieving embedding model: {e}")


def get_embedding_model(model_name: str) -> Embeddings:
    """Initialize the embedding client against the LLM Lab OpenAI-compatible API.
    The Qdrant-side vector_name (e.g. 'Qwen/Qwen3-Embedding-8B') is translated
    into the API slug (e.g. 'qwen3-embedding-8b'); EMBEDDING_MODEL_API_NAME
    overrides this translation if the heuristic does not fit a given provider."""
    api_model = os.getenv("EMBEDDING_MODEL_API_NAME") or _vector_to_api_model(model_name)
    if api_model != model_name:
        logger.info(
            f"Embedding model: API name={api_model!r} (Qdrant vector key={model_name!r})."
        )
    else:
        logger.info(f"Embedding model: {api_model!r}.")
    return Embeddings(
        model=api_model,
        base_url=os.environ["LLMLAB_URL"],
        api_key=os.environ["LLMLAB_API_KEY"],
    )


def create_vector_db(
    docs: Iterable[Mapping],
    embedding_model: Embeddings,
    collection_name: str,
) -> VectorDB:
    """Embed the provided docs and upsert them into a Qdrant collection that
    holds a single named vector. Each doc must be a mapping with
    'page_content' (str) and 'metadata' (dict) keys."""
    logger.info("🧠 Creating Qdrant vector DB with embeddings")
    client = get_qdrant_client()
    vector_name = os.getenv("EMBEDDING_MODEL")

    docs = list(docs)
    texts = [d["page_content"] for d in docs]
    metadatas = [d.get("metadata", {}) for d in docs]
    vectors = embedding_model.embed_documents(texts)
    dim = len(vectors[0])

    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={vector_name: qm.VectorParams(size=dim, distance=qm.Distance.COSINE)},
        )
        logger.info(f"Collection '{collection_name}' created (dim={dim}, vector='{vector_name}').")
    except UnexpectedResponse as e:
        if e.status_code == 409:
            logger.info(f"Collection '{collection_name}' already exists — reusing it.")
        else:
            raise

    points = [
        qm.PointStruct(
            id=i,
            vector={vector_name: vec},
            payload={"page_content": text, "metadata": meta},
        )
        for i, (text, meta, vec) in enumerate(zip(texts, metadatas, vectors))
    ]
    for start in range(0, len(points), UPSERT_BATCH_SIZE):
        client.upsert(collection_name=collection_name, points=points[start : start + UPSERT_BATCH_SIZE])
    logger.info(f"Upserted {len(points)} points into '{collection_name}'.")

    return VectorDB(client, embedding_model, vector_name, collection_name)


def get_vector_db(collection_name: str) -> VectorDB:
    """Build a VectorDB handle over an existing Qdrant collection."""
    client = get_qdrant_client()
    vector_name = get_embedding_model_name(client, collection_name)
    embeddings = get_embedding_model(vector_name)
    return VectorDB(client, embeddings, vector_name, collection_name)


def get_retriever(collection_name: str) -> VectorDB:
    """Initialize the retriever from an existing Qdrant collection."""
    return get_vector_db(collection_name)
