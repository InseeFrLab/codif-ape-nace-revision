# Interactive pipeline for debugging — run line by line in a REPL / Jupyter / VSCode.
# Each section can be executed independently. Intermediate variables (data, prompts,
# generation_outputs, results, metrics, df_eval) stay available in the namespace.

import asyncio
import logging
import os
import tempfile
import time

import mlflow
import nest_asyncio

os.chdir("codif-ape-nace-revision/src")
import config
from constants.data import VAR_TO_KEEP
from constants.paths import URL_SIRENE4_EXTRACTION
from evaluation.evaluator import Evaluator
from strategies.cag import CAGStrategy
from strategies.rag import RAGStrategy
from utils.data import get_ambiguous_data

config.setup()
nest_asyncio.apply()  # allow multiple asyncio.run() calls in the same interpreter


# =============================================================================
# Parameters — edit here for interactive runs
# =============================================================================
strategy_cls       = RAGStrategy        # CAGStrategy or RAGStrategy
experiment_name    = "Test"             
run_name           = None
collection_name    = "embeddings_qwen"               # only used for RAG # embeddings_qwen
llm_name           = "gemma4-26b-moe"
third              = None
prompts_from_file  = False
save_prompts       = False
prompt_name        = "rag-classifier"   # "rag-classifier" for RAG
prompt_label       = "production"
top_k              = 5               # None for CAG, e.g. 5 for RAG
sample_size        = 50
only_annotated     = True


# =============================================================================
# Step 1 — Initialize strategy
# =============================================================================
logging.info("Initializing strategy ==========================")
kwargs = {
    "generation_model": llm_name,
    "prompt_name":      prompt_name,
    "prompt_label":     prompt_label,
}
if strategy_cls is RAGStrategy:
    kwargs["collection_name"] = collection_name
strategy = strategy_cls(**kwargs)


# =============================================================================
# Step 2 — Load ambiguous data
# =============================================================================
logging.info("Loading ambiguous data ==========================")
data = get_ambiguous_data(strategy.mapping, third, only_annotated, VAR_TO_KEEP)
if sample_size is not None:
    data = data.head(n=sample_size).reset_index(drop=True)


# =============================================================================
# Step 3 — Retrieve prompts
# =============================================================================
logging.info("Retrieving prompts ==========================")
_t0 = time.time()
prompts = asyncio.run(strategy.get_prompts(
    data,
    load_prompts_from_file=prompts_from_file,
    top_k=top_k,
    save=save_prompts,
))
retrieval_time_mn = (time.time() - _t0) / 60
logging.info(f"Retrieved {len(prompts)} prompts in {retrieval_time_mn:.2f} min")


# =============================================================================
# Step 4 — LLM generation
# =============================================================================
logging.info("Starting LLM generation ==========================")
_t0 = time.time()
generation_outputs = asyncio.run(strategy.call_llm(prompts))
generation_time_mn = (time.time() - _t0) / 60
logging.info(f"Generated {len(generation_outputs)} outputs in {generation_time_mn:.2f} min")

generation_outputs[2]

# =============================================================================
# Step 5 — Process outputs and merge with data
# =============================================================================
processed_outputs = strategy.process_outputs(generation_outputs)
results = data.merge(processed_outputs, left_index=True, right_index=True)


# =============================================================================
# Step 6 — Evaluate
# =============================================================================
metrics, df_eval = Evaluator().evaluate(results, prompts)
metrics.update({
    "num_coded":         int(results["codable"].sum()),
    "num_not_coded":     int(len(results) - results["codable"].sum()),
    "pct_not_coded":     round((len(results) - results["codable"].sum()) / len(results) * 100, 2),
    "retrieval_time_mn": round(retrieval_time_mn, 1),
    "generation_time_mn": round(generation_time_mn, 1),
})
print(metrics)


# # =============================================================================
# # Step 7 — MLflow logging (optional — comment out if you only want to debug)
# # =============================================================================
# mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
# mlflow.set_experiment(experiment_name)
# with mlflow.start_run(run_name=run_name):
#     output_path = strategy.save_results(results, third=None)
#     params = {
#         "LLM_MODEL":              llm_name,
#         "TEMPERATURE":            strategy.sampling_params["temperature"],
#         "input_path":             URL_SIRENE4_EXTRACTION,
#         "output_path":            output_path,
#         "strategy":               "cag" if isinstance(strategy, CAGStrategy) else "rag",
#         "top_k":                  top_k,
#         "URL_SIRENE4_EXTRACTION": URL_SIRENE4_EXTRACTION,
#     }
#     if hasattr(strategy, "db"):
#         params["COLLECTION_NAME"] = collection_name
#         params["EMBEDDING_MODEL"] = getattr(strategy.db, "model_name", None)

#     mlflow.log_params(params)
#     for metric, value in metrics.items():
#         mlflow.log_metric(metric, value)

#     with tempfile.TemporaryDirectory() as tmpdir:
#         file_path = os.path.join(tmpdir, "df_eval.csv")
#         df_eval.to_csv(file_path, index=False)
#         mlflow.log_artifact(file_path, artifact_path="dataframes")

