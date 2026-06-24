URL_SIRENE4_EXTRACTION = "s3://projet-ape/extractions/20250825_sirene4.parquet"
URL_SIRENE4_AMBIGUOUS_CAG = "s3://projet-ape/NAF-revision/relabeled-data-cag"
URL_SIRENE4_AMBIGUOUS_RAG = "s3://projet-ape/NAF-revision/relabeled-data-rag"
URL_SIRENE4_UNIVOCAL = "s3://projet-ape/NAF-revision/relabeled-data/sirene4_univoques.parquet"
URL_SIRENE4_AMBIGUOUS_FINAL = "s3://projet-ape/NAF-revision/relabeled-data/"
URL_SIRENE4_NACE2025 = "s3://projet-ape/NAF-revision/relabeled-data/sirene4_nace2025.parquet"
URL_MAPPING_TABLE = "s3://projet-ape/NAF-revision/table-correspondance-naf2025.xls"
URL_EXPLANATORY_NOTES = "s3://projet-ape/NAF-revision/Notes explicatives NACE et NAF.xlsx"
URL_GROUND_TRUTH = (
    "s3://projet-ape/label-studio/annotation-campaign-2024/rev-NAF2025/preprocessed/training_data_NAF2025.parquet"
)
URL_PROMPTS_RAG = "s3://projet-ape/NAF-revision/prompts/{collection}/prompts-rag-{prompt_name}-{prompt_label}.parquet"
URL_PROMPTS_CAG = "s3://projet-ape/NAF-revision/prompts/cag/prompts-rag-{prompt_name}-{prompt_label}.parquet"
URL_RUN_ID = "s3://projet-ape/NAF-revision/run_ids/{experiment_name}/{llm_name}.txt"
