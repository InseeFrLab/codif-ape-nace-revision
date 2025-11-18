# Download du modèle (commande bash)
# export MODEL_NAME=Qwen/Qwen3-32B
# uv run huggingface-cli download $MODEL_NAME

import asyncio
import logging
import os
import tempfile
# os.chdir('./codif-ape-nace-revision/src')
import time
import mlflow
import gc
import torch
import config
from utils.data import fetch_mapping, get_file_system, df_to_prompts
import pandas as pd
import vllm
from vllm.sampling_params import GuidedDecodingParams, SamplingParams
from constants.llm import MODEL_TO_ARGS
from strategies.cag import CAGResponse

config.setup()

llm_name = "Qwen/Qwen3-32B"

logging.info("Import des prompts =======")

fs = get_file_system()
url = "s3://projet-ape/NAF-revision/prompts/test/prompts-cag.parquet"
prompts_df = pd.read_parquet(
    url,
    filesystem=fs,
)
prompts = df_to_prompts(prompts_df)
logging.info(f"Nombre total de prompts: {len(prompts)} =======")


logging.info("Initialisation du LLM =======")

model_args = {
    'max_model_len': 25000, 
    'gpu_memory_utilization': 0.95, 
    'enable_prefix_caching': False
}

llm = vllm.LLM(
    model=llm_name,
    **model_args,
)

response_format = CAGResponse

sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.01,
    seed=2025,
    logprobs=1,
    guided_decoding=GuidedDecodingParams(
        json=response_format.model_json_schema()
    ),
)

logging.info("Génération =======")

outputs = llm.chat(prompts[:10], sampling_params=sampling_params)



# messages = [
#     {"role": "system", "content": "Tu es un assistant utile."},
#     {"role": "user", "content": "Quelle est la capitale de la France ?"}
# ]

# # Conversion en prompt texte avec le template du modèle
# prompt_text = strategy.tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,  # Important : retourne du texte, pas des tokens
#     add_generation_prompt=True  # Ajoute le prompt de génération (ex: "Assistant:")
# )


#prompts = prompts[:20000]

# # Paramètre de batching
# BATCH_SIZE = 2048
# logging.info(f"Taille des batches: {BATCH_SIZE}")

# all_generation_outputs = []
# total_generation_time_mn = 0.0

# # Boucle sur les batchs
# for i in range(0, len(prompts), BATCH_SIZE):
#     batch_prompts = prompts[i:i + BATCH_SIZE]
#     logging.info(f"🚀 Traitement du batch {i // BATCH_SIZE + 1} / {len(prompts) // BATCH_SIZE + 1} "
#           f"({len(batch_prompts)} prompts)")

#     if i > 0:
#         logging.info("Init du llm pour libérer la RAM")
#         strategy.initialize_llm()
#     else: 
#         logging.info("Pas d'init du llm pour le premier batch")

#     # Génération pour ce batch
#     logging.info("Début de l'inférence ===========")
#     generation_outputs, generation_time_mn = _generate_outputs(strategy, batch_prompts)

#     # Sauvegarde des résultats intermédiaires (optionnel mais recommandé)
#     all_generation_outputs.extend(generation_outputs)
#     total_generation_time_mn += generation_time_mn

#     # Libération mémoire pour éviter la montée continue de RAM
#     logging.info("Cleaning du LLM")
#     strategy.cleanup_llm()
#     del batch_prompts
#     del generation_outputs
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

logging.info(f"✅ Génération terminée pour {len(prompts)} prompts.")
logging.info(f"⏱ Temps total de génération : {total_generation_time_mn:.2f} minutes")




