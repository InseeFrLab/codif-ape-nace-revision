# Interactive pipeline for debugging — run line by line in a REPL / Jupyter / VSCode.
# Each section can be executed independently. Intermediate variables (labels, docs,
# emb_model, db) stay available in the namespace.

import logging
import os

os.chdir("codif-ape-nace-revision/src")
import config
from vector_db.loading import create_vector_db, get_embedding_model
from vector_db.notices_nace import fetch_nace2025_labels

config.setup()


# =============================================================================
# Parameters — edit here for interactive runs
# =============================================================================
collection_name      = "embeddings_qwen_test"
excluded_fields      = None    # e.g. ["not_include", "notes"] or ["include", "not_include", "notes"]
embedding_model_name = "qwen3-embedding-8b"


# =============================================================================
# Step 1 — Fetch NACE 2025 labels
# =============================================================================
logging.info("Fetching NACE 2025 labels ==========================")
labels = fetch_nace2025_labels(excluded_fields)
# Inspect: labels.head(), labels.columns, labels["content"].iloc[0]


# =============================================================================
# Step 2 — Build docs (page_content + metadata mappings expected by create_vector_db)
# =============================================================================
logging.info("Building docs ==========================")
docs = [
    {"page_content": row.pop("content"), "metadata": row}
    for row in labels.to_dict(orient="records")
]
# Inspect: docs[0], len(docs)


# =============================================================================
# Step 3 — Initialize embedding model
# =============================================================================
logging.info("Initializing embedding model ==========================")
emb_model = get_embedding_model(embedding_model_name)
# Sanity check: emb_model.embed_documents(["hello"])


# =============================================================================
# Step 4 — Create vector DB
# =============================================================================
logging.info("Creating vector DB ==========================")
db = create_vector_db(docs, emb_model, collection_name, model_name=embedding_model_name)
logging.info(f"Qdrant DB has been created in collection '{collection_name}'.")
# Inspect: db.client.get_collection(collection_name)
