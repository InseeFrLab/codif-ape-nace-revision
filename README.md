# CAG vs RAG: NACE 2008 → NACE 2025 recodification

LLM-based recoding of business activity descriptions from NACE 2008 to NACE 2025, comparing a **Retrieval-Augmented Generation (RAG)** strategy against a **Context-Augmented Generation (CAG)** strategy.

## Repository structure

```
.
├── src/
│   ├── 1_build_vector_db.py          # Build Qdrant collection (RAG only)
│   ├── 2_encode_unambiguous.py       # Rule-based recoding for unambiguous cases
│   ├── 3_encode_ambiguous.py         # LLM-based recoding (RAG or CAG)
│   ├── 4_ensemble_predictions.py     # Majority-vote ensemble across multiple LLM runs
│   ├── 5_build_nace2025_sirene4.py   # Build the final NACE 2025 dataset
│   ├── strategies/                   # rag.py, cag.py, base.py
│   ├── vector_db/                    # Qdrant client + NACE notices loader
│   ├── evaluation/                   # Metrics + reports
│   ├── mappings/, constants/, config/, utils/
│   └── analyse_errors.qmd, analyse_metrics.qmd
├── argo-workflows/                   # Argo workflow specs
├── tests/
├── pyproject.toml
└── .env.example
```

## Setup

Requires Python 3.12+ and [`uv`](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env   # then fill in the secrets
uv run pre-commit install
```

## Running `3_encode_ambiguous.py`

**RAG** (retrieves top-k NACE notices from a Qdrant collection before generation):

```bash
uv run src/3_encode_ambiguous.py \
  --strategy rag \
  --llm_name qwen3-6-35b-moe \
  --collection_name embeddings_qwen \
  --experiment_name NACE2025_DATASET \
  --top_k 5 \
  --third 1
```

**CAG** (the full NACE notice catalog is passed in the prompt, no retrieval):

```bash
uv run src/3_encode_ambiguous.py \
  --strategy cag \
  --llm_name qwen3-6-35b-moe \
  --experiment_name NACE2025_DATASET \
  --third 1
```

Common flags: `--sample_size N` for a subset, `--thinking` to enable LLM reasoning mode, `--only_annotated true` to restrict to annotated cases. See `python src/3_encode_ambiguous.py --help` for the full list.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
