#!/bin/bash

llm_name="Qwen/Qwen3-0.6B"
# llm_name="Qwen/Qwen3-32B"
# llm_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# llm_name="mistralai/Mistral-Small-3.2-24B-Instruct-2506"
# llm_name="openai/gpt-oss-20b"

#only_annotated="false"

uv run huggingface-cli download "$llm_name"

# on construit un tableau au lieu d’une seule chaîne
CMD=(uv run src/encode_ambiguous.py --strategy cag --experiment_name Test --llm_name "$llm_name")

if [[ "${only_annotated:-false}" == "true" ]]; then
    CMD+=("--only_annotated")
fi

echo "Executing: ${CMD[*]}"
"${CMD[@]}"


# top_k=5


# uv run src/encode_ambiguous.py \
#     --strategy cag \
#     --llm_name "$llm_name" \
#     --save_prompts \
#     --sample_size 100 \
#     --only_annotated False

# #    --top_k "$top_k" \
# #    --third 1 \

