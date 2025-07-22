from src.evaluator import FullEvaluator
import os
import json
import argparse

def append_line_to_file(file_path, line):
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topics_path', type=str, required=True)
    parser.add_argument('--nuggets_path', type=str, required=True)
    parser.add_argument('--qrels_path', type=str, required=True)
    parser.add_argument('--agent_model_name', type=str, required=True)
    parser.add_argument('--corpus_name', type=str, required=True)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--embedder_name', type=str) # only for dense index
    parser.add_argument('--index_path', type=str, required=True)
    parser.add_argument('--mapper_path', type=str, required=True)
    parser.add_argument('--lucene_index_path', type=str)
    parser.add_argument('--search_client_type', type=str, choices=['bm25', 'dense'], required=True)
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--eval_model', type=str, required=True)
    parser.add_argument('--log_dir', type=str)
    parser.add_argument('--latex_table', type=str)
    args = parser.parse_args()

    evalutator = FullEvaluator(
        nuggets_path=args.nuggets_path,
        qrels_path=args.qrels_path,
        agent_model_name=args.agent_model_name,
        corpus_name=args.corpus_name,
        corpus_path=args.corpus_path,
        index_path=args.index_path,
        mapper_path=args.mapper_path,
        search_client_type=args.search_client_type,
        embedder_name=args.embedder_name,
        lucene_index_path=args.lucene_index_path,
        enable_thinking=args.enable_thinking,
        eval_model=args.eval_model,
        log_dir=args.log_dir
    )
    result = evalutator.run_eval()
    print(result)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    with open(os.path.join(args.log_dir, 'final_eval.json'), 'w') as file:
        json.dump(result, file, indent=4)
    
    if args.latex_table:
        line = ' & '.join([
            f"{args.agent_model_name}",
            f"{(result['task_finish_rate'] * 100.):.1f}",
            f"{(result['final_task_completeness'] * 100.):.1f}",
            f"{(result['avg_citation_recall'] * result['task_finish_rate'] * 100.):.1f}",
            f"{(result['avg_citation_precision'] * result['task_finish_rate'] * 100.):.1f}",
            f"{result['avg_latency']:.1f}",
            f"{result['avg_cost']:.2f}",
            f"{result['avg_iteration_turns']:.1f}",
            f"{(result['macro_avg_search_precision'] * 100.):.1f}",
            f"{(result['macro_avg_search_recall'] * 100.):.1f}",
            f"{(result['avg_avg_search_gain'] * 100.):.1f}",
            f"{(result['micro_avg_fetch_url_error'] * 100.):.1f}",
            f"{(result['micro_avg_fetch_precision'] * 100.):.1f}",
        ])
        line += ' \\\\'
        append_line_to_file(args.latex_table, line)


if __name__ == '__main__':
    main()