import logging
import os

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def create_vector_db(docs, embedding_model: OpenAIEmbeddings, collection_name: str) -> QdrantVectorStore:
    logger.info("🧠 Creating Qdrant vector DB with embeddings")
    return QdrantVectorStore.from_documents(
        docs,
        embedding_model,
        collection_name=collection_name,
        vector_name=os.getenv("EMBEDDING_MODEL"),
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        port="443",
        https=True,
    )


def get_qdrant_client() -> QdrantClient:
    """Initialize and return the Qdrant client."""
    return QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        port=443,
        https=True,
    )


def get_embedding_model_name(client: QdrantClient, collection_name: str) -> str:
    """Retrieve the embedding model name from the Qdrant collection."""
    try:
        collection_info = client.get_collection(collection_name=collection_name)
        return next(iter(collection_info.config.params.vectors.keys()))
    except Exception as e:
        raise RuntimeError(f"Error retrieving embedding model: {e}")


def get_embedding_model(model_name: str) -> OpenAIEmbeddings:
    """Initialize the embedding model against the llm.lab OpenAI-compatible API."""
    return OpenAIEmbeddings(
        model=model_name,
        openai_api_base=os.environ["LLMLAB_URL"],
        openai_api_key=os.environ["LLMLAB_API_KEY"],
        tiktoken_enabled=False,
        embedding_ctx_length=8192,
    )


def get_vector_db(collection_name: str) -> QdrantVectorStore:
    """Initialize the Qdrant Vector Store from the existing collection."""
    client = get_qdrant_client()
    emb_model_name = get_embedding_model_name(client, collection_name)
    emb_model = get_embedding_model(emb_model_name)
    return QdrantVectorStore.from_existing_collection(
        embedding=emb_model,
        collection_name=collection_name,
        vector_name=emb_model_name,
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY"),
        port=443,
        https=True,
    )


def get_retriever(collection_name: str):
    """Initialize the retriever from a vector database."""
    return get_vector_db(collection_name)
