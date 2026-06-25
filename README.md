# CAG vs RAG: NACE 2008 → NACE 2025 recodification

Recode labelled French business records into **NACE 2025** (NAF rev3) using LLMs. Each record carries a free-text **activity description** and a **NACE 2008** (NAF rev2) code whose mapping to NACE 2025 is **not always bijective**: when one 2008 code maps to several 2025 candidates, the LLM disambiguates from the activity description; the univocal cases are resolved by rule.

Two generation strategies are compared: **Retrieval-Augmented Generation (RAG)** (top-k NACE notices retrieved from Qdrant) vs **Context-Augmented Generation (CAG)** (short list of NACE 2025 candidates added in the prompt, based on NAF 2008 pre-existing labels).

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


## Argo workflow

The pipeline runs as a single Argo workflow (`argo-workflows/relabel.yaml`) in the **`projet-ape`** namespace. The DAG chains steps 0→5:

- **0 validate** — check required columns and value formats of the input file.
- **1 build-vector-db** — build the Qdrant collection (RAG only, opt-in).
- **2 encode-unambiguous** — rule-based recoding of univocal NACE 2008 → 2025 codes.
- **3 encode** — LLM recoding of ambiguous cases, one branch per model, batched and **resumable**.
- **4 ensemble** — majority-vote consolidation across model runs.
- **5 build-final** — assemble the final NACE 2025 dataset.

It runs in two modes: **prod** (recode an arbitrary input file) or **eval** (compute accuracy metrics on human-annotated rows).

Only `argo-workflows/params.yaml` is edited per run. A run is scoped by a single `job-id`, which roots every output under `s3://projet-ape/NAF-revision/workflow_relabel/<job-id>/`. Re-submitting with the same `job-id` resumes a crashed run (completed step-3 batches are skipped). See `argo-workflows/argo_helper.md` for the submit/monitor/resume procedure.

## Work in progress

This workflow is **not finalized**: it still carries a fair amount of methodological exploration (RAG vs CAG, multi-model voting, eval tooling). Results so far point toward a production pipeline that is **CAG-only** and relies on a **single LLM** rather than a majority vote across several models. As a consequence, the code is more complex than the target pipeline will need — once the methodology is locked in, the RAG strategy, the ensemble/voting step, and the remaining exploratory branches can be pruned to leave a leaner CAG-only, single-model pipeline.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
