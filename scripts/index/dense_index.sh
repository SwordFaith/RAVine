export JAVA_HOME=${YOUR_JAVA_DIR}/jdk-21.0.7
export CLASSPATH=.:${JAVA_HOME}/lib
export PATH=${CLASSPATH}:${JAVA_HOME}/bin:$PATH
export JVM_PATH=${YOUR_JAVA_DIR}/jdk-21.0.7/lib/server/libjvm.so
# replace YOUR_JAVA_DIR with your actual Java directory

EMBEDDER_NAME="/path/to/embedder/model/"

python scripts/index/preprocess.py

CUDA_VISIBLE_DEVICES=0 python -m pyserini.encode \
    input   --corpus {shard_0_file_path} \
            --fields text \
            --delimiter "\n" \
            --shard-id 0 \
            --shard-num 4 \
    output  --embeddings {shard_0_embedding_file_path} \
            --to-faiss \
    encoder --encoder ${EMBEDDER_NAME} \
            --fields text \
            --batch 128 > shard_0.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -m pyserini.encode \
    input   --corpus {shard_1_file_path} \
            --fields text \
            --delimiter "\n" \
            --shard-id 1 \
            --shard-num 4 \
    output  --embeddings {shard_1_embedding_file_path} \
            --to-faiss \
    encoder --encoder ${EMBEDDER_NAME} \
            --fields text \
            --batch 128 > shard_1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -m pyserini.encode \
    input   --corpus {shard_2_file_path} \
            --fields text \
            --delimiter "\n" \
            --shard-id 2 \
            --shard-num 4 \
    output  --embeddings {shard_2_embedding_file_path} \
            --to-faiss \
    encoder --encoder ${EMBEDDER_NAME} \
            --fields text \
            --batch 128 > shard_2.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -m pyserini.encode \
    input   --corpus {shard_3_file_path} \
            --fields text \
            --delimiter "\n" \
            --shard-id 3 \
            --shard-num 4 \
    output  --embeddings {shard_3_embedding_file_path} \
            --to-faiss \
    encoder --encoder ${EMBEDDER_NAME} \
            --fields text \
            --batch 128 > shard_3.log 2>&1 &

wait

# merge
python -m pyserini.index.merge_faiss_indexes \
    --prefix {full_index_file_path}- \
    --shard-num 4
