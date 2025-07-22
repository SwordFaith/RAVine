export JAVA_HOME=${YOUR_JAVA_DIR}/jdk-21.0.7
export CLASSPATH=.:${JAVA_HOME}/lib
export PATH=${CLASSPATH}:${JAVA_HOME}/bin:$PATH
export JVM_PATH=${YOUR_JAVA_DIR}/jdk-21.0.7/lib/server/libjvm.so
# replace YOUR_JAVA_DIR with your actual Java directory

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


CMD=(python -m src.run_eval
    --nuggets_path "$nuggets_path"
    --qrels_path "$qrels_path"
    --agent_model_name "$agent_model_name"
    --corpus_name "$corpus_name"
    --index_path "$index_path"
    --mapper_path "$mapper_path"
    --search_client_type "$search_client_type"
    --eval_model "$eval_model"
)

[ -n "$corpus_path"] && CMD+=(--corpus_path "$corpus_path")
[ -n "$log_dir" ] && CMD+=(--log_dir "$log_dir")
[ -n "$lucene_index_path" ] && CMD+=(--lucene_index_path "$lucene_index_path")
[ -n "$embedder_name" ] && CMD+=(--embedder_name "$embedder_name")
[ -n "$latex_table" ] && CMD+=(--latex_table "$latex_table")
[ "$enable_reasoning" = "true" ] && CMD+=(--enable_thinking)

echo "Running: ${CMD[@]}"

exec "${CMD[@]}"
