from src.nuggetizer import Nuggetizer

def main():
    model_name = 'gemini-2.5-flash-preview-05-20-nothinking'
    embedding_model = 'Alibaba-NLP/gte-modernbert-base'
    nuggetizer = Nuggetizer(
        model=model_name,
        embedding_model=embedding_model,
    )
    input_file = '/path/to/your/input_file.jsonl'  # Update with your actual input file path
    output_file = '/path/to/your/output_file.jsonl'  # Update with your actual output file path
    
    nuggetizer.nuggetization(input_file=input_file, output_file=output_file)


if __name__ == "__main__":
    main()
