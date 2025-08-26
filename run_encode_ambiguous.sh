#!/bin/bash

# -------------------------------
# Config
# -------------------------------
# List of models to test
models=(
  #"Qwen/Qwen3-0.6B"
  "Qwen/Qwen2-7B-Instruct"
)

# List of collections
collections=(
  #"embeddings_qwen_light"
  "embeddings_qwen_semi_light"
  #"embeddings_qwen"
)

# List of top_k values
top_ks=(
  4
#  7
#  10
)

# -------------------------------
# Compute the total number of runs
# -------------------------------
total_runs=$(( ${#models[@]} * ${#collections[@]} * ${#top_ks[@]} ))
echo ">>> Total number of runs planned: $total_runs"
echo "----------------------------------------------"

# -------------------------------
# Loops
# -------------------------------
run_id=1
for model in "${models[@]}"; do
  # Download the generation model, once per model
  echo ">>> Downloading model $model"
  uv run huggingface-cli download "$model"

  for collection in "${collections[@]}"; do
    for top_k in "${top_ks[@]}"; do
      echo ""
      echo ">>> Run $run_id / $total_runs"
      echo "    Model      : $model"
      echo "    Collection : $collection"
      echo "    top_k      : $top_k"
      echo "    reranker_model      : BAAI/bge-reranker-large"
      echo "----------------------------------------------"

      uv run src/encode_ambiguous.py \
          --strategy rag \
          --experiment_name NACE2025_DATASET \
          --collection_name "$collection" \
          --llm_name "$model" \
          --top_k "$top_k" \
          --reranker_model BAAI/bge-reranker-large

      run_id=$((run_id+1))
    done
  done
done
