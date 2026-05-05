"""
Test connection sur llm.lab
Test available models
"""

import os
#os.chdir("codif-ape-nace-revision")
import sys
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path  

load_dotenv()  

# --------------------------------------------------
# Check env variables
# --------------------------------------------------

_REQUIRED_ENV = ["LLMLAB_API_KEY", "LLMLAB_URL"]
_missing = [v for v in _REQUIRED_ENV if not os.getenv(v)]
if _missing:
    sys.exit(f"Variables d'environnement manquantes : {', '.join(_missing)}")

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
def make_client() -> OpenAI:
    return OpenAI(
        base_url=os.environ["LLMLAB_URL"],
        api_key=os.environ["LLMLAB_API_KEY"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_list_models(client: OpenAI) -> list[str]:
    """Vérifie que le serveur répond et liste les modèles disponibles."""
    models = client.models.list()
    ids = [m.id for m in models.data]
    assert ids, "Aucun modèle disponible sur le serveur"
    print(f"[OK] Modèles disponibles : {ids}")
    return ids



def test_embedding(client: OpenAI, model: str = "qwen3-embedding-8b") -> None:
    """Checks that the embedding model returns valid vectors for a few sample texts."""
    texts = [
        "Boulangerie artisanale spécialisée dans le pain au levain.",
        "Développement de logiciels de gestion pour les PME.",
        "Transport routier de marchandises à longue distance.",
    ]
    response = client.embeddings.create(model=model, input=texts)
    assert len(response.data) == len(texts), "Nombre d'embeddings incorrect"
    for i, embedding_obj in enumerate(response.data):
        vec = embedding_obj.embedding
        assert isinstance(vec, list) and len(vec) > 0, f"Embedding {i} vide ou invalide"
    dim = len(response.data[0].embedding)
    print(f"[OK] Embeddings {model} — {len(texts)} vecteurs de dimension {dim}")


if __name__ == "__main__":
    client = make_client()
    print(test_list_models(client))
    test_embedding(client, model="gwen3-embedding-8b")