# Input extraction (stays in place; never written by the pipeline).
URL_SIRENE4_EXTRACTION = "s3://projet-ape/extractions/20250825_sirene4.parquet"

# -----------------------------------------------------------------------------
# Workflow run root — EVERY intermediate and final output of one pipeline run
# lives under this job-scoped directory, one sub-folder per step. job_id scopes
# the whole run so concurrent runs never collide.
#   {job_id}/univocal/  — step 2 (rule-based rewrites)
#   {job_id}/ambiguous/ — step 3 (LLM, one sub-dir per model: results/ + prompts/)
#   {job_id}/ensemble/  — step 4 (majority-vote consolidation)
#   {job_id}/final/     — step 5 (assembled NACE 2025 dataset)
#   {job_id}/run_ids/   — step 3 publishes its MLflow run id for the aggregate step
# All are .format(job_id=..., [llm_name=...]) at runtime.
# -----------------------------------------------------------------------------
URL_WORKFLOW_ROOT = "s3://projet-ape/NAF-revision/workflow_relabel/{job_id}"
URL_WORKFLOW_UNIVOCAL = URL_WORKFLOW_ROOT + "/univocal/sirene4_univoques.parquet"
URL_WORKFLOW_AMBIGUOUS = URL_WORKFLOW_ROOT + "/ambiguous"
URL_WORKFLOW_ENSEMBLE = URL_WORKFLOW_ROOT + "/ensemble/sirene4_ambiguous.parquet"
URL_WORKFLOW_FINAL = URL_WORKFLOW_ROOT + "/final/sirene4_nace2025.parquet"
URL_RUN_ID = URL_WORKFLOW_ROOT + "/run_ids/{llm_name}.txt"

URL_MAPPING_TABLE = "s3://projet-ape/NAF-revision/table-correspondance-naf2025.xls"
URL_EXPLANATORY_NOTES = "s3://projet-ape/NAF-revision/Notes explicatives NACE et NAF.xlsx"
URL_GROUND_TRUTH = (
    "s3://projet-ape/label-studio/annotation-campaign-2024/rev-NAF2025/preprocessed/training_data_NAF2025.parquet"
)
URL_PROMPTS_RAG = "s3://projet-ape/NAF-revision/prompts/{collection}/prompts-rag-{prompt_name}-{prompt_label}.parquet"
URL_PROMPTS_CAG = "s3://projet-ape/NAF-revision/prompts/cag/prompts-rag-{prompt_name}-{prompt_label}.parquet"
