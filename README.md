# RAVine

RAVine (a Reality-Aligned eValuation framework for agentic LLMs with search), is a comprehensive evaluation system for agentic search, encompassing the web environment, benchmark datasets, and a novel evaluation method, serving as a full-process, reproducible, and goal-aligned evaluation sandbox.

## Features


- 🎯**More Precise**: RAVine provides a more precise and attributable method for nuggets (claim-level ground truth) extraction, the final generated report evaluation is more accurate.
- ⚙️**More Comprehensive**: RAVine not only focuses on end-to-end result evaluation, but also designs detailed search process performance indicators.
- 🚀**Full Sandbox**: We have packaged the search tools and evaluation framework. You only need to provide an LLM to run or evaluate the agent search on RAVine.


## Datasets

We have uploaded the running evaluation data, nuggets, corpus index, etc. to hugging face. You can access them and modify the corresponding path variables in the running script:
- Queries & Nuggets: https://huggingface.co/datasets/sapphirex/RAVine-nuggets
- Raw Qrels: https://huggingface.co/datasets/sapphirex/RAVine-qrels
- Dense Index: https://huggingface.co/datasets/sapphirex/RAVine-dense-index
- URL-Docid Mapper: https://huggingface.co/datasets/sapphirex/RAVine-mapper
- Running logs (for reproduction): https://huggingface.co/datasets/sapphirex/RAVine-logs


## How to run/evaluate?

First, install the operating environment. This project mainly uses two environments, and we recommend using `uv` to install the following two environments separately:
- Env for vllm: the vllm version we use is `0.9.0.1`, run `pip install vllm==0.9.0.1`.
- Env for `/src`: the operating environment of our main program has been exported to `requirements_agent.txt`. Run `uv pip install -r requirements_agent.txt`.



Second, write the configuration file, which is related to the selection and setting of the model, operating environment, index, and file path. You can find examples at `configs/` and write your own config here.


Third, run the vllm service and then run the main program. Example instructions are as follows:
```
bash scripts/server/vllm.sh your_config_file # run the server of agentic llm
bash scripts/evaluation/run.sh your_config_file
```
If you need to run multiple configurations at once, we provide the corresponding scripts. Please write the configurations to be run into `/scripts/evaluation/run_all_eval.sh` and then run it.


<!-- ## Experimental Results -->


## Citation

