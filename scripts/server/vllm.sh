#!/bin/bash

CONFIG=$1

eval $(python3 -c "
import yaml, sys
def format_value(v):
    if isinstance(v, bool):
        return '\"true\"' if v else '\"false\"'
    elif isinstance(v, str):
        return f'\"{v}\"'
    else:
        return str(v)
with open(sys.argv[1]) as f:
    data = yaml.safe_load(f)
    for k, v in data.items():
        print(f'{k}={format_value(v)}')
" "$CONFIG")

CMD=(vllm serve "$agent_model_name"
    --tensor-parallel-size "$tensor_parallel_size"
    --dtype "$dtype"
    --api-key "$api_key"
    --enable-auto-tool-choice
    --tool-call-parser "$tool_call_parser"
    --max-model-len "$max_model_len"
)

[ -n "$chat_template" ] && CMD+=(--chat-template "$chat_template")

[ "$enable_reasoning" = "true" ] && CMD+=(--enable-reasoning)

[ "$rope_scaling" = "true" ] && CMD+=(--rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}')

echo "Running: ${CMD[@]}"

exec "${CMD[@]}"
