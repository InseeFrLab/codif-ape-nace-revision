"""
Test connection sur llm.lab
Test available models
"""

import os
#os.chdir("codif-ape-nace-revision")
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(override=True)

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


_RAG_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "codable":    {"type": "boolean"},
        "nace2025":   {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "confidence": {"anyOf": [{"type": "number"}, {"type": "null"}]},
    },
    "required": ["codable"],
    "title": "RAGResponse",
    "additionalProperties": False,
}

_RAG_PROMPT = [
    {
        "role": "system",
        "content": (
            "Tu es un expert de la Nomenclature statistique des Activités économiques "
            "dans la Communauté Européenne (NACE) chargé de classifier l'activité principale "
            "des entreprises selon la dernière version.\n"
            "Réponds uniquement avec un objet JSON valide, sans texte supplémentaire."
        ),
    },
    {
        "role": "user",
        "content": (
            "# Activité principale de l'entreprise :\n"
            "Boulangerie artisanale\n\n"
            "# Codes APE 2025 proposés :\n"
            "========\n"
            "10.71A: Fabrication industrielle de pain et de pâtisserie fraîche\n"
            "========\n"
            "10.71B: Cuisson de produits de boulangerie-pâtisserie en magasin\n"
            "========\n"
            "47.24Z: Commerce de détail de pain, pâtisserie et confiserie en magasin spécialisé\n\n"
            "# Instructions :\n"
            "Sélectionne le code APE 2025 parmi : '10.71A', '10.71B', '47.24Z'.\n"
            "Réponds en JSON : {\"codable\": <bool>, \"nace2025\": <code ou null>, \"confidence\": <float>}"
        ),
    },
]


def test_rag_completion(client: OpenAI, model: str) -> dict:
    """Checks that the LLM returns a valid structured JSON response for a RAG-style prompt."""
    response = client.chat.completions.create(
        model=model,
        messages=_RAG_PROMPT,
        max_tokens=100,
        temperature=0.01,
        seed=2025,
        logprobs=True,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "RAGResponse",
                "schema": _RAG_RESPONSE_SCHEMA,
                "strict": True,
            },
        },
    )
    content = response.choices[0].message.content
    parsed = json.loads(content)
    assert "codable" in parsed, "Champ 'codable' manquant"
    assert parsed.get("nace2025") in {"10.71A", "10.71B", "47.24Z", None}, f"Code inattendu : {parsed.get('nace2025')}"
    print(f"[OK] RAG generation — model={model}, response={parsed}")
    return parsed


if __name__ == "__main__":
    client = make_client()
    print(test_list_models(client))
    test_embedding(client, model="qwen3-embedding-8b")
    test_rag_completion(client, model="gemma4-26b-moe")