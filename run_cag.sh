#!/bin/bash


collections="embeddings_qwen"
llm_name="Qwen/Qwen3-0.6B"
top_k=5

uv run huggingface-cli download "$llm_name"

uv run src/encode_ambiguous.py \
    --strategy cag \
    --collection_name "$collections" \
    --llm_name "$llm_name" \
    --third 1 \
    --top_k "$top_k" \
    --sample_size 100

