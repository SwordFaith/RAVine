# Details of the Project

In this note, we dicuss the role of each main file in this repo. If you want to make modifications to our framework, we might recommend you to read this note.

There are three folders which mostly contributes to our implementation:
- `\src`: This directory includes the most important implementation codes, including agentic LLM, nuggetization, evaluation, web search API, etc.
- `\scripts`: This directory includes execution scripts for various tasks, such as starting the vllm service, running or evaluating agentic LLMs with search, building the search engine index, etc.
- `\configs`: This directory includes configuration files and chat templates. The configuration files include models (agents), vllm services, data paths, experimental settings, etc.



## `\src`
- `\agent`: Implementation of agentic llm with search.
    - `history.py`: Implementation of the history context class used to record and manage the iteration process of each run of the agentic LLM.
    - `prompts.py`: Prompt template for instructing LLMs to perform agentic retrieval-augmented generation.
    - `searcher.py`: Implementation class of agentic llm with search.
- `\data`: Definition of data types and evaluation indicators.
    - `metrics.py`: Definition classes of evaluation metrics.
    - `types.py`: Definition of basic data types used in `\src`. This reference is from [repo-nuggetizer](https://github.com/castorini/nuggetizer).
- `\evaluator`: Implementation of evaluator.
    - `evaluator.py`: The main evaluator, which can be used by providing models, data, etc.
    - `log_evaluator.py`: Log-based evaluator, which requires additional running logs. This is for the purpose of reproduction.
- `\llm`: Implementation of LLM class, the basis of agentic LLM.
    - `client.py`: LLM client class based on OpenAI chat completion.
- `\nuggetizer`: Implementation of nuggetization.
    - `embedder.py`: The class of the embedding model used when building and querying dense index.
    - `handler.py`: The class for calling LLM-as-Judge, which is referenced from [repo-nuggetizer](https://github.com/castorini/nuggetizer).
    - `nuggetizer.py`: The class for nuggets extracting, merging, scoring, and assignment, which is the main file for nuggetization.
    - `prompts.py`: Prompt template for instructing LLM-as-Judge to create, merge, score and assign nuggets.
    - `utils.py`: Some utilities for nuggetization.
- `\tool`: Definition and implementation of various tools.
    - `\web_search`: Definition and implementation of web search tools.
        - `client.py`: Client class for web search tools, including clients for the dense index and bm25 index.
        - `function.py`: Pythonic function implementation and OpenAI function calling definition of search tool and fetch tool.
        - `url2doc.py`: A mapper of URLs and web pages in the corpus, for use in tools, evaluation, etc.
- `nuggetize.py`: Main program of nuggets collection.
- `run_agent.py`: Run entry of agentic llm with search.
- `run_eval.py`: Main program for evaluation.
- `run_log_eval.py`: Main program for evaluation reproduction from logs.


## `\scripts`

- `\agent\run.sh`: Scripts for running the agentic llms with search.
- `\evaluation`: Evaluation start point.
    - `run.sh`: Run the evaluation of a single model configuration.
    - `run_log.sh`: Reproduce the evaluation based on logs.
    - `run_all_eval.sh`: Run the evaluation of multiple model configurations.
    - `run_all_eval_log.sh`: Reproduce the evaluation of multiple model configurations based on logs.
- `\index`: Scripts for index building.
    - `dense_index.sh`: Script for corpus preprocessing, index building by sharding, and index merging.
    - `preprocess.py`: Corpus preprocessing code, including web pages extracting and sharding.
- `\nuggetization\run.sh`: Scripts for running nuggetization (create, merge, score).
- `\server\vllm.sh`: Scripts for running vllm server.

## `\configs`

- `{model_name}_{context_length}_{index}_{thinking}.yaml`: The configuration file of each model, including the upper limit of context length, index type, whether thinking, etc.
- `\chat`: Some chat templates that may be used.
