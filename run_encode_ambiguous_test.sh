#!/bin/bash

#Qwen/Qwen3-0.6B
export HF_HOME=/home/onyxia/work/.cache
export MODEL_NAME=tiiuae/falcon-7b-instruct
uv run huggingface-cli download $MODEL_NAME


uv run src/encode_ambiguous_test.py \
    --strategy rag \
    --experiment_name NACE2025_DATASET \
    --collection_name embeddings_qwen_light \
    --llm_name mosaicml/mpt-7b-instruct \
    --third 1 \
    --sample_size 50

uv run src/encode_ambiguous_test.py \
    --strategy rag \
    --experiment_name NACE2025_DATASET \
    --collection_name embeddings_qwen_semi_light \
    --llm_name mosaicml/mpt-7b-instruct \
    --third 1 \
    --sample_size 50

uv run src/encode_ambiguous_test.py \
    --strategy rag \
    --experiment_name NACE2025_DATASET \
    --collection_name embeddings_qwen \
    --llm_name mosaicml/mpt-7b-instruct \
    --third 1 \
    --sample_size 50
