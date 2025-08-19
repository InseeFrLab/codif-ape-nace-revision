#!/bin/bash

#Qwen/Qwen3-0.6B
# export HF_HOME=/home/onyxia/work/.cache
# export MODEL_NAME=Qwen/Qwen2-7B-Instruct
# uv run huggingface-cli download $MODEL_NAME


uv run src/encode_ambiguous_test.py \
    --strategy rag \
    --experiment_name NACE2025_DATASET \
    --collection_name embeddings_qwen_light \
    --llm_name Qwen/Qwen2-7B-Instruct \
    --third 1

uv run src/encode_ambiguous_test.py \
    --strategy rag \
    --experiment_name NACE2025_DATASET \
    --collection_name embeddings_qwen_semi_light \
    --llm_name Qwen/Qwen2-7B-Instruct \
    --third 1

uv run src/encode_ambiguous_test.py \
    --strategy rag \
    --experiment_name NACE2025_DATASET \
    --collection_name embeddings_qwen \
    --llm_name Qwen/Qwen2-7B-Instruct \
    --third 1 \
