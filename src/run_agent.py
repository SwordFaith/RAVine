from src.agent import AgenticSearcher
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--corpus_name', type=str, required=True)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--embedder_name', type=str) # only for dense index
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--mapper_path', type=str, required=True)
    parser.add_argument('--lucene_index_path', type=str)
    parser.add_argument('--search_client', type=str, choices=['bm25', 'dense'])
    parser.add_argument('--enable_thinking', action='store_true')
    args = parser.parse_args()

    agentic_searcher = AgenticSearcher(
        model_name=args.model_name,
        corpus_name=args.corpus_name,
        corpus_path=args.corpus_path,
        index_path=args.index_path,
        mapper_path=args.mapper_path,
        search_client_type=args.search_client,
        log_dir=None,
        embedder_name=args.embedder_name,
        lucene_index_path=args.lucene_index_path,
        enable_thinking=args.enable_thinking
    )
    agentic_searcher.run(
        # input='how much does the chinese government bowlderize its citizens',
        input='how does i-5 relate to redlining in the puget sound',
        input_id='0',
        print_logs=True
    )


if __name__ == '__main__':
    main()


