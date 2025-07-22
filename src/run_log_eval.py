from src.evaluator import LogEvaluator
import os
import argparse

def append_line_to_file(file_path, line):
    dir_path = os.path.dirname(file_path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nuggets_path', type=str, required=True)
    parser.add_argument('--qrels_path', type=str, required=True)
    parser.add_argument('--agent_model_name', type=str, required=True)
    parser.add_argument('--corpus_name', type=str, required=True)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--mapper_path', type=str, required=True)
    parser.add_argument('--enable_thinking', action='store_true')
    parser.add_argument('--eval_model', type=str, required=True)
    parser.add_argument('--log_dir', type=str)
    args = parser.parse_args()

    evalutator = LogEvaluator(
        nuggets_path=args.nuggets_path,
        qrels_path=args.qrels_path,
        agent_model_name=args.agent_model_name,
        corpus_name=args.corpus_name,
        corpus_path=args.corpus_path,
        mapper_path=args.mapper_path,
        enable_thinking=args.enable_thinking,
        eval_model=args.eval_model,
        log_dir=args.log_dir
    )
    result = evalutator.run_eval()
    print(result)


if __name__ == '__main__':
    main()