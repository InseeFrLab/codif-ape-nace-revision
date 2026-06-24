import argparse
import logging
import os

import config
from constants.paths import URL_EXPLANATORY_NOTES, URL_MAPPING_TABLE
from vector_db.loading import create_vector_db, get_embedding_model
from vector_db.notices_nace import fetch_nace2025_labels

config.setup()
logger = logging.getLogger(__name__)


def main(collection_name: str, model_name: str, excluded_fields: list[str] | None = None):
    logger.info("===== STEP 1: build vector DB =====")
    logger.info("INPUT  : %s", URL_MAPPING_TABLE)
    logger.info("INPUT  : %s", URL_EXPLANATORY_NOTES)
    logger.info("OUTPUT : Qdrant collection '%s' at %s", collection_name, os.getenv("QDRANT_URL"))

    labels = fetch_nace2025_labels(excluded_fields)

    # Build a list of {page_content, metadata} mappings, the format expected
    # by create_vector_db. The 'content' column becomes page_content; every
    # other column lands in metadata.
    docs = [
        {"page_content": row.pop("content"), "metadata": row}
        for row in labels.to_dict(orient="records")
    ]

    # Initialize embedding model
    emb_model = get_embedding_model(model_name)

    _ = create_vector_db(docs, emb_model, collection_name, model_name=model_name)

    logging.info(f"Qdrant DB has been created in collection '{collection_name}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build a Qdrant vector database from NACE 2025 labels."
    )
    parser.add_argument(
        "--collection_name",
        type=str,
        default="embeddings_qwen",
        help="Name of the Qdrant collection to create.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="embedding model name.",
    )
    parser.add_argument(
        "--excluded_fields",
        type=str,
        nargs="*",
        default=None,
        help="Fields to exclude from the NACE labels (e.g. include not_include notes).",
    )
    args = parser.parse_args()

    main(
        collection_name=args.collection_name, 
        model_name=args.model_name, 
        excluded_fields=args.excluded_fields
    )
