import logging
import os

from langchain_community.document_loaders import DataFrameLoader

import config
from vector_db.loading import create_vector_db, get_embedding_model
from vector_db.notices_nace import fetch_nace2025_labels

config.setup()
logger = logging.getLogger(__name__)


def main(collection_name: str, excluded_fields: list[str] | None = None):
    labels = fetch_nace2025_labels(excluded_fields)

    # Load documents
    docs = DataFrameLoader(labels, page_content_column="content").load()

    # Initialize embedding model
    emb_model = get_embedding_model(os.getenv("EMBEDDING_MODEL"))

    _ = create_vector_db(docs, emb_model, collection_name)

    logging.info(f"Qdrant DB has been created in collection '{collection_name}'.")


if __name__ == "__main__":
    main(collection_name="embeddings_qwen_light", excluded_fields=["include", "not_include", "notes"])
    # main(collection_name=os.getenv("COLLECTION_NAME"))
