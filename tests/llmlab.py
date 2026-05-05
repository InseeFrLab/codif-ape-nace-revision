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


# def test_completion(client: OpenAI, model: str) -> str:
#     """Vérifie qu'une requête simple aboutit."""
#     response = client.chat.completions.create(
#         model=model,
#         messages=[{"role": "user", "content": "Réponds uniquement avec le mot 'ok'."}],
#         max_tokens=10,
#         temperature=0,
#     )
#     content = response.choices[0].message.content
#     assert content, "Réponse vide"
#     print(f"[OK] Complétion simple : {content!r}")
#     return content


# def test_json_schema(client: OpenAI, model: str) -> ReponseFormat:
#     """Vérifie que le serveur respecte un json_schema Pydantic."""
#     response = client.chat.completions.create(
#         model=model,
#         messages=[
#             {
#                 "role": "user",
#                 "content": (
#                     "Produit : lait entier bio 1L. "
#                     "Peux-tu lui attribuer un code COICOP ? "
#                     "Réponds en JSON."
#                 ),
#             }
#         ],
#         max_tokens=256,
#         temperature=0,
#         response_format={
#             "type": "json_schema",
#             "json_schema": {
#                 "name": "ReponseFormat",
#                 "schema": ReponseFormat.model_json_schema(),
#                 "strict": True,
#             },
#         },
#     )
#     raw = response.choices[0].message.content
#     parsed = ReponseFormat.model_validate_json(raw)
#     print(f"[OK] json_schema — codable={parsed.codable}, code={parsed.code_predict!r}, confidence={parsed.confidence:.2f}")
#     return parsed



if __name__ == "__main__":
    client = make_client()
    print(test_list_models(client))