#!/bin/bash

llm_name="Qwen/Qwen3-0.6B"
top_k=5

uv run huggingface-cli download "$llm_name"

uv run src/encode_ambiguous.py \
    --strategy cag \
    --llm_name "$llm_name" \
    --top_k "$top_k" \
    --save_prompts \
    --sample_size 1000

#    --third 1 \

