#!/bin/bash

set -e

VLLM_ENV="/path/to/your/vllm/venv/"
EVAL_ENV="/path/to/your/search-agent/venv/"

CONFIGS=(
    "qwen2.5_7b_instruct_32k_bm25.yaml"
    "qwen2.5_7b_instruct_32k_dense.yaml"
    "qwen2.5_7b_instruct_128k_bm25.yaml"
    "qwen2.5_7b_instruct_128k_dense.yaml"
    "qwen2.5_32b_instruct_32k_bm25.yaml"
    "qwen2.5_32b_instruct_32k_dense.yaml"
    "qwen2.5_32b_instruct_128k_bm25.yaml"
    "qwen2.5_32b_instruct_128k_dense.yaml"
    "qwen3_8b_32k_bm25_thinking.yaml"
    "qwen3_8b_32k_dense_thinking.yaml"
    "qwen3_8b_128k_bm25_thinking.yaml"
    "qwen3_8b_128k_dense_thinking.yaml"
    "qwen3_8b_32k_bm25_no_thinking.yaml"
    "qwen3_8b_32k_dense_no_thinking.yaml"
    "qwen3_8b_128k_bm25_no_thinking.yaml"
    "qwen3_8b_128k_dense_no_thinking.yaml"
    "qwen3_32b_32k_bm25_thinking.yaml"
    "qwen3_32b_32k_dense_thinking.yaml"
    "qwen3_32b_128k_bm25_thinking.yaml"
    "qwen3_32b_128k_dense_thinking.yaml"
    "qwen3_32b_32k_bm25_no_thinking.yaml"
    "qwen3_32b_32k_dense_no_thinking.yaml"
    "qwen3_32b_128k_bm25_no_thinking.yaml"
    "qwen3_32b_128k_dense_no_thinking.yaml"
    "qwen3_30b_a3b_32k_bm25_thinking.yaml"
    "qwen3_30b_a3b_32k_dense_thinking.yaml"
    "qwen3_30b_a3b_128k_bm25_thinking.yaml"
    "qwen3_30b_a3b_128k_dense_thinking.yaml"
    "qwen3_30b_a3b_32k_bm25_no_thinking.yaml"
    "qwen3_30b_a3b_32k_dense_no_thinking.yaml"
    "qwen3_30b_a3b_128k_bm25_no_thinking.yaml"
    "qwen3_30b_a3b_128k_dense_no_thinking.yaml"
    "llama3.1_8b_32k_bm25.yaml"
    "llama3.1_8b_32k_dense.yaml"
    "llama3.1_8b_128k_bm25.yaml"
    "llama3.1_8b_128k_dense.yaml"
)


PORT=8000

for config in "${CONFIGS[@]}"; do

    echo ">>> $eval_cmd"
    bash scripts/evaluation/run_log.sh "configs/${config}"

    echo ">>> Finished task for $config."
    echo "-----------------------------------------"
done

echo "All tasks completed."
