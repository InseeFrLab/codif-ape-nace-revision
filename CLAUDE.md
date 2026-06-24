# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

LLM-based recoding of French business activity descriptions from NACE 2008 (NAF rev2) to NACE 2025 (NAF rev3). Compares two generation strategies: **RAG** (Retrieval-Augmented Generation via Qdrant vector DB) and **CAG** (Context-Augmented Generation, full catalog in prompt).

The pipeline runs in two modes:
- **prod** — recode an arbitrary input file, produce the NACE 2025 dataset (no labels needed).
- **eval** — run on the SIRENE 4 extraction filtered to human-annotated rows, compute accuracy metrics.

## Commands

```bash
uv sync                          # install deps
cp .env.example .env             # fill in secrets
uv run pre-commit install

# Step 0 — validate an input file (prod) before running
uv run src/0_input_validation.py --mode prod --input_url s3://.../input.parquet

# Step 3 — encode ambiguous (CAG, prod, resumable in batches)
uv run src/3_encode_ambiguous.py \
  --strategy cag --mode prod \
  --input_url s3://.../input.parquet \
  --llm_name qwen3-6-35b-moe \
  --experiment_name NACE2025_DATASET \
  --job_id job-001 --batch_size 1000

# Step 3 — eval mode (metrics on annotated rows; no input_url)
uv run src/3_encode_ambiguous.py --strategy rag --mode eval \
  --llm_name qwen3-6-35b-moe --collection_name embeddings_qwen \
  --experiment_name NACE2025_DATASET --top_k 5

# Step 4 — ensemble / consolidate
uv run src/4_ensemble_predictions.py --run_ids id1,id2,id3 --mode prod --export

# Lint
uv run ruff check src/ && uv run ruff format src/ && uv run vulture src/
```

`ruff` line length is 120 characters.

## Pipeline Steps

| Step | Script | Purpose |
|------|--------|---------|
| 0 | `0_input_validation.py` | Gate: required columns present + value formats (NAF08 regex, non-null id/libelle) |
| 1 | `1_build_vector_db.py` | Create Qdrant collection from NACE 2025 labels (RAG only) |
| 2 | `2_encode_unambiguous.py` | Rule-based mapping for univocal NACE 2008 → 2025 codes (prod) |
| 3 | `3_encode_ambiguous.py` | LLM-based encoding of ambiguous cases (batched, resumable) |
| 4 | `4_ensemble_predictions.py` | Majority-vote ensemble across LLM runs / consolidate |
| 5 | `5_build_nace2025_sirene4.py` | Combine univocal + ambiguous + ground truth into final dataset (prod) |

Every step logs its inputs/outputs with explicit `INPUT  :` / `OUTPUT :` lines.

## Architecture

### Strategy Pattern (`src/strategies/`)

Both strategies inherit from `EncodeStrategy` (`base.py`), which handles async LLM calls, retries, confidence scoring (token log-probabilities), and Pydantic response validation. `_format_activity_description` builds the prompt's activity text from column names declared in `constants/data.py` (never hardcoded).

- **RAGStrategy** (`rag.py`): queries Qdrant for top-k NACE 2025 notices, injects them into the prompt.
- **CAGStrategy** (`cag.py`): embeds the full NACE 2025 catalog directly in the prompt, no retrieval.

### Batched & resumable encoding (step 3)

Step 3 processes data in batches of `--batch_size` (default 1000). Each batch builds prompts → `call_llm` (concurrency unchanged) → writes prompts+results as one Parquet part under a stable, job-scoped S3 dir (`utils/batch.py`). Re-running with the **same `--job_id`** skips batches whose results part already exists → a crashed multi-hour run resumes where it stopped. Omitting `--job_id` generates a fresh timestamped id (no resume). Each run writes its MLflow run id to S3 (`URL_RUN_ID`) so the ensemble step can find it.

### Key Modules

- `src/constants/data.py` — **input schema column names** (`ID_VAR`, `NACE08_VAR`, `ACTIVITY_LABEL_VAR`, `ACTIVITY_PRECISION_VARS`, `VAR_TO_KEEP`). Single source of truth; `VAR_TO_KEEP` splats `ACTIVITY_PRECISION_VARS` so every precision column is always loaded.
- `src/constants/paths.py` — all S3 paths.
- `src/utils/batch.py` — batch paths, resume detection, per-batch IO, token-stat aggregation.
- `src/utils/data.py` — S3/DuckDB loading, `get_ambiguous_data` (input_url=None ⇒ eval/annotated, else prod), `write_run_id_to_s3`.
- `src/evaluation/evaluator.py` — accuracy/mapping/position metrics vs Label Studio ground truth.

### External Services

LLM Lab (OpenAI-compatible API) · Qdrant (RAG) · MLflow (tracking) · Langfuse (tracing + `"production"` prompts) · S3/MinIO · DuckDB.

## Environment Variables

See `.env.example`: `LLMLAB_*`, `MLFLOW_TRACKING_*`, `QDRANT_*`, `LANGFUSE_*`, `AWS_*`.

## Argo Workflows (`argo-workflows/`)

- `relabel.yaml` — full pipeline DAG (steps 0→5). Parameter **declarations** only (no values).
- `params.yaml` — parameter **values**; the only file to edit per run.
- `argo_helper.md` — submit/monitor/resume procedure with the `argo` CLI.

```
0 validate ──┬─> 1 build-vector-db (rag, opt-in) ─┐
             ├─> 2 encode-unambiguous (prod) ──────┼─> 5 build-final (prod)
             └─> 3 encode (1 per model) ─> aggregate ─> 4 ensemble ─┘
```
Conditional execution: build-vector-db (rag + opt-in), encode-unambiguous & build-final (prod only), ensemble (multi-model, or always in prod). Submit with `argo submit relabel.yaml --parameter-file params.yaml -n <ns>`.

## Refactor state (handoff)

Recent work, decisions made:
- **prod/eval modes** added to step 3 (`--mode`); replaced the old `--only_annotated`. eval loads annotated rows + computes metrics; prod loads `--input_url`, skips eval.
- **Resumable batching** in step 3 (`--job_id` required for resume, `--batch_size`); LLM call mechanics unchanged.
- **Step 4** parametrized (`--run_ids`, `--mode`, `--export`); removed hardcoded `RUN_IDS`/`EXPORT_FINAL`. Single-model voting = identity export.
- **Step 0** (input validation) added.
- **Coherence fixes**: defined `URL_SIRENE4_UNIVOCAL`/`URL_SIRENE4_NACE2025`; fixed step 5 imports (`src.`→relative, `cache_models`→`data`); step 2/5 accept `--input_url`; step 3 passes `--collection_name` (RAG).
- **Column names centralized** in `constants/data.py`. Resolved a latent bug where `activ_nat_lib_et_1`/`lib_cj` were used but never loaded — now included in `VAR_TO_KEEP` (option 2). `lib_cj` must be the *label*, not the `cj` code.
- **Argo**: single `relabel.yaml` covering all steps; params externalized to `params.yaml`.

Known caveats (not yet addressed): step 5 reads `URL_SIRENE4_AMBIGUOUS_FINAL` as a directory (cross-day files could mix); univocal/final output paths are fixed (concurrent prod runs would collide).
