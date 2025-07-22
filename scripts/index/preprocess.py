from pyserini.search.lucene import LuceneSearcher
from tqdm import tqdm
import json

def extract_docs():
    corpus_path = '/path/to/lucene-index/'
    output_path = '/path/to/msmarco-v2.1-doc.jsonl'
    searcher = LuceneSearcher(corpus_path)
    num_docs = searcher.num_docs

    with open(output_path, 'w') as file:
        for i in tqdm(range(num_docs)):
            doc = json.loads(searcher.doc(i).raw())
            file.write(json.dumps(doc) + '\n')
    
    return num_docs

def sharding_indexing_input(num_docs, num_shards=4, shard_size=2800000):
    corpus_path = '/path/to/msmarco-v2.1-doc.jsonl'
    output_path = '/path/to/msmarco-v2.1-doc-shard-{shard_id}.jsonl'

    with open(corpus_path, 'r') as file:
        lines = [json.loads(line) for line in file]
    
    shard_id = 0
    offset = 0
    while shard_id < num_shards and offset < num_docs:
        end = min(offset + shard_size, num_docs)
        shard_lines = lines[offset: end]

        with open(output_path.format(shard_id=shard_id), 'w') as file:
            for line in shard_lines:
                file.write(json.dumps({
                    'id': line['docid'],
                    'text': line['body'],
                }) + '\n')
        
        print(f'shard {shard_id} finished.')

        offset = end
        shard_id += 1



if __name__ == '__main__':
    num_docs = extract_docs()
    sharding_indexing_input(num_docs)

