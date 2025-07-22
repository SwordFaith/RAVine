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

    log_path="/path/to/your/logs/$(basename "$config" .yaml).log"

    echo ">>> Activating VLLM environment"
    source "$VLLM_ENV"

    echo ">>> Starting server with $config, logging to $log_path"
    bash scripts/server/vllm.sh "configs/${config}" > "$log_path" 2>&1 &
    SERVER_PID=$!


    echo ">>> Waiting for server on port $PORT to be ready..."
    until curl -s "http://localhost:$PORT/v1/models" >/dev/null; do
        sleep 5
    done

    echo ">>> Switching to evaluation environment"
    source "$EVAL_ENV"

    echo ">>> $eval_cmd"
    bash scripts/evaluation/run.sh "configs/${config}"

    echo ">>> Evaluation complete. Shutting down server..."
    kill $SERVER_PID
    sleep 20
    if ps -p $SERVER_PID > /dev/null; then
        kill -9 $SERVER_PID
    fi

    echo ">>> Finished task for $config."
    echo "-----------------------------------------"
done

echo "All tasks completed."
