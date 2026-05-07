import logging
import os

import config
from vector_db.loading import create_vector_db, get_embedding_model
from vector_db.notices_nace import fetch_nace2025_labels

config.setup()
logger = logging.getLogger(__name__)


def main(collection_name: str, excluded_fields: list[str] | None = None):
    labels = fetch_nace2025_labels(excluded_fields)

    # Build a list of {page_content, metadata} mappings, the format expected
    # by create_vector_db. The 'content' column becomes page_content; every
    # other column lands in metadata.
    docs = [
        {"page_content": row.pop("content"), "metadata": row}
        for row in labels.to_dict(orient="records")
    ]

    # Initialize embedding model
    emb_model = get_embedding_model(os.getenv("EMBEDDING_MODEL"))

    _ = create_vector_db(docs, emb_model, collection_name)

    logging.info(f"Qdrant DB has been created in collection '{collection_name}'.")


if __name__ == "__main__":
    # main(collection_name="embeddings_qwen_semi_light", excluded_fields=["not_include", "notes"])
    # main(collection_name="embeddings_qwen_light", excluded_fields=["include", "not_include", "notes"])
    main(collection_name="embeddings_qwen")
    # main(collection_name=os.getenv("COLLECTION_NAME"))
