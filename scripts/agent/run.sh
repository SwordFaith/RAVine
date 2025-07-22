export JAVA_HOME=${YOUR_JAVA_DIR}/jdk-21.0.7
export CLASSPATH=.:${JAVA_HOME}/lib
export PATH=${CLASSPATH}:${JAVA_HOME}/bin:$PATH
export JVM_PATH=${YOUR_JAVA_DIR}/jdk-21.0.7/lib/server/libjvm.so
# replace YOUR_JAVA_DIR with your actual Java directory


MODEL_NAME="/path/to/your/model/"
CORPUS_NAME="msmarco-v2.1-doc"
MAPPER_PATH="/path/to/url2doc/mapper/file/"
LUCENE_INDEX_PATH="/path/to/bm25/index/"
DENSE_INDEX_PATH="/path/to/dense/index/"
EMBEDDER_NAME="/path/to/embedder/model/"


python -m src.run_agent \
    --model_name ${MODEL_NAME} \
    --corpus_name ${CORPUS_NAME} \
    --index_path ${DENSE_INDEX_PATH} \
    --mapper_path ${MAPPER_PATH} \
    --embedder_name ${EMBEDDER_NAME} \
    --lucene_index_path ${LUCENE_INDEX_PATH} \
    --search_client dense


# running in thinking mode
# python -m src.run_agent \
#     --model_name ${QWEN2_5_7B_INSTRUCT} \
#     --corpus_name ${CORPUS_NAME} \
#     --index_path ${LUCENE_INDEX_PATH} \
#     --mapper_path ${MAPPER_PATH} \
#     --lucene_index_path ${LUCENE_INDEX_PATH} \
#     --search_client bm25 \
#     --enable_thinking
