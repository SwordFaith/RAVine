# RAVine

RAVine (a Reality-Aligned eValuation framework for agentic LLMs with search), is a comprehensive evaluation system for agentic search, encompassing the web environment, benchmark datasets, and a novel evaluation method, serving as a full-process, reproducible, and goal-aligned evaluation sandbox.

## Features


- 🎯**More Precise**: RAVine provides a more precise and attributable method for nuggets (claim-level ground truth) extraction, the final generated report evaluation is more accurate.
- ⚙️**More Comprehensive**: RAVine not only focuses on end-to-end result evaluation, but also designs detailed search process performance indicators.
- 💰**Lower Cost**: RAVine provides local search APIs and reduces the cost of calling LLM-Judge(Gemini-2.5-Flash) to ~0.01$ per evaluation data.
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

For more detailed steps, see [Evaluation_and_reproduction](https://github.com/SwordFaith/RAVine/blob/main/docs/evaluation_and_reproduction.md).

## Experimental Results

Table Description:
- "Rate" denotes the Task Completion Rate.
- "Comp." refers to the score of Task Completeness.
- "Rec." and "Prec." represent Recall and Precision, respectively.
- "URL Err." denotes the URL Error.
- Latency is measured in seconds, and cost is measured in dollars.
- Symbols (↑) and (↓) indicate that higher or lower values are preferred, respectively.
- Models not marked with (Thinking) either run without thinking or lack support for the thinking mode.
- Bold values indicate the best performance for each corresponding metric in the column.


Evaluation results on RAVine, with a maximum context length of 32k and the index built by gte-modernbert-base:

<table>
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th colspan="4" style="text-align:center;">Report Quality</th>
      <th colspan="3" style="text-align:center;">Efficiency</th>
      <th colspan="3" style="text-align:center;">Search</th>
      <th colspan="2" style="text-align:center;">Fetch</th>
    </tr>
    <tr>
      <th style="text-align:center;">Rate (↑)</th>
      <th style="text-align:center;">Comp. (↑)</th>
      <th style="text-align:center;">Rec. (↑)</th>
      <th style="text-align:center;">Prec. (↑)</th>
      <th style="text-align:center;">Latency (↓)</th>
      <th style="text-align:center;">Cost (↓)</th>
      <th style="text-align:center;">Turns</th>
      <th style="text-align:center;">Prec. (↑)</th>
      <th style="text-align:center;">Rec. (↑)</th>
      <th style="text-align:center;">Gain (↑)</th>
      <th style="text-align:center;">URL Err. (↓)</th>
      <th style="text-align:center;">Prec. (↑)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-7B-Instruct</td>
      <td style="text-align:center;">19.0</td>
      <td style="text-align:center;">6.8</td>
      <td style="text-align:center;">1.9</td>
      <td style="text-align:center;">1.8</td>
      <td style="text-align:center;">7.1</td>
      <td style="text-align:center;">0.01</td>
      <td style="text-align:center;">3.1</td>
      <td style="text-align:center;">18.7</td>
      <td style="text-align:center;">5.7</td>
      <td style="text-align:center;">4.7</td>
      <td style="text-align:center;">8.8</td>
      <td style="text-align:center;">19.8</td>
    </tr>
    <tr>
      <td>Qwen2.5-32B-Instruct</td>
      <td style="text-align:center;">71.4</td>
      <td style="text-align:center;">23.0</td>
      <td style="text-align:center;"><strong>14.9</strong></td>
      <td style="text-align:center;"><strong>16.5</strong></td>
      <td style="text-align:center;">40.3</td>
      <td style="text-align:center;">0.03</td>
      <td style="text-align:center;">4.0</td>
      <td style="text-align:center;"><strong>21.1</strong></td>
      <td style="text-align:center;">6.5</td>
      <td style="text-align:center;">4.4</td>
      <td style="text-align:center;">1.4</td>
      <td style="text-align:center;">28.7</td>
    </tr>
    <tr>
      <td>Qwen3-8B (Thinking)</td>
      <td style="text-align:center;">86.9</td>
      <td style="text-align:center;">37.8</td>
      <td style="text-align:center;">10.4</td>
      <td style="text-align:center;">12.1</td>
      <td style="text-align:center;">13.9</td>
      <td style="text-align:center;">0.03</td>
      <td style="text-align:center;">6.6</td>
      <td style="text-align:center;">19.7</td>
      <td style="text-align:center;">6.6</td>
      <td style="text-align:center;">5.1</td>
      <td style="text-align:center;">8.9</td>
      <td style="text-align:center;">27.3</td>
    </tr>
    <tr>
      <td>Qwen3-8B</td>
      <td style="text-align:center;">28.6</td>
      <td style="text-align:center;">12.4</td>
      <td style="text-align:center;">4.8</td>
      <td style="text-align:center;">6.1</td>
      <td style="text-align:center;">11.2</td>
      <td style="text-align:center;">0.06</td>
      <td style="text-align:center;">9.3</td>
      <td style="text-align:center;">19.3</td>
      <td style="text-align:center;">5.9</td>
      <td style="text-align:center;">5.0</td>
      <td style="text-align:center;">2.4</td>
      <td style="text-align:center;">23.8</td>
    </tr>
    <tr>
      <td>Qwen3-32B (Thinking)</td>
      <td style="text-align:center;"><strong>98.8</strong></td>
      <td style="text-align:center;"><strong>43.5</strong></td>
      <td style="text-align:center;">11.7</td>
      <td style="text-align:center;">15.1</td>
      <td style="text-align:center;">19.6</td>
      <td style="text-align:center;">0.02</td>
      <td style="text-align:center;">2.8</td>
      <td style="text-align:center;">19.2</td>
      <td style="text-align:center;">5.0</td>
      <td style="text-align:center;">4.0</td>
      <td style="text-align:center;">8.9</td>
      <td style="text-align:center;">22.2</td>
    </tr>
    <tr>
      <td>Qwen3-32B</td>
      <td style="text-align:center;">85.7</td>
      <td style="text-align:center;">38.0</td>
      <td style="text-align:center;">12.8</td>
      <td style="text-align:center;">12.6</td>
      <td style="text-align:center;">14.6</td>
      <td style="text-align:center;">0.08</td>
      <td style="text-align:center;">8.5</td>
      <td style="text-align:center;">19.1</td>
      <td style="text-align:center;">6.3</td>
      <td style="text-align:center;">5.0</td>
      <td style="text-align:center;">8.1</td>
      <td style="text-align:center;">20.2</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B (Thinking)</td>
      <td style="text-align:center;">81.0</td>
      <td style="text-align:center;">35.6</td>
      <td style="text-align:center;">10.6</td>
      <td style="text-align:center;">14.2</td>
      <td style="text-align:center;">33.0</td>
      <td style="text-align:center;">0.10</td>
      <td style="text-align:center;">6.6</td>
      <td style="text-align:center;">19.7</td>
      <td style="text-align:center;">6.2</td>
      <td style="text-align:center;">3.6</td>
      <td style="text-align:center;">10.3</td>
      <td style="text-align:center;">29.3</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B</td>
      <td style="text-align:center;">77.4</td>
      <td style="text-align:center;">30.9</td>
      <td style="text-align:center;">11.3</td>
      <td style="text-align:center;">14.2</td>
      <td style="text-align:center;">15.7</td>
      <td style="text-align:center;">0.07</td>
      <td style="text-align:center;">7.3</td>
      <td style="text-align:center;">16.8</td>
      <td style="text-align:center;">6.2</td>
      <td style="text-align:center;">3.4</td>
      <td style="text-align:center;"><strong>0.6</strong></td>
      <td style="text-align:center;"><strong>30.4</strong></td>
    </tr>
    <tr>
      <td>LLaMA-3.1-8B-Instruct</td>
      <td style="text-align:center;">96.4</td>
      <td style="text-align:center;">24.0</td>
      <td style="text-align:center;">3.1</td>
      <td style="text-align:center;">3.1</td>
      <td style="text-align:center;">7.3</td>
      <td style="text-align:center;">0.02</td>
      <td style="text-align:center;">2.7</td>
      <td style="text-align:center;">12.1</td>
      <td style="text-align:center;"><strong>8.8</strong></td>
      <td style="text-align:center;"><strong>6.6</strong></td>
      <td style="text-align:center;">36.8</td>
      <td style="text-align:center;">15.8</td>
    </tr>
  </tbody>
</table>



Evaluation results on RAVine, with a maximum context length of 128k and the index built by gte-modernbert-base:

<table>
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th colspan="4" style="text-align:center;">Report Quality</th>
      <th colspan="3" style="text-align:center;">Efficiency</th>
      <th colspan="3" style="text-align:center;">Search</th>
      <th colspan="2" style="text-align:center;">Fetch</th>
    </tr>
    <tr>
      <th style="text-align:center;">Rate (↑)</th>
      <th style="text-align:center;">Comp. (↑)</th>
      <th style="text-align:center;">Rec. (↑)</th>
      <th style="text-align:center;">Prec. (↑)</th>
      <th style="text-align:center;">Latency (↓)</th>
      <th style="text-align:center;">Cost (↓)</th>
      <th style="text-align:center;">Turns</th>
      <th style="text-align:center;">Prec. (↑)</th>
      <th style="text-align:center;">Rec. (↑)</th>
      <th style="text-align:center;">Gain (↑)</th>
      <th style="text-align:center;">URL Err. (↓)</th>
      <th style="text-align:center;">Prec. (↑)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-7B-Instruct</td>
      <td style="text-align:center;">1.2</td>
      <td style="text-align:center;">0.3</td>
      <td style="text-align:center;">0.0</td>
      <td style="text-align:center;">0.0</td>
      <td style="text-align:center;">4.5</td>
      <td style="text-align:center;">0.01</td>
      <td style="text-align:center;">1.6</td>
      <td style="text-align:center;">7.4</td>
      <td style="text-align:center;">2.6</td>
      <td style="text-align:center;">2.3</td>
      <td style="text-align:center;"><strong>0.0</strong></td>
      <td style="text-align:center;"><strong>33.3</strong></td>
    </tr>
    <tr>
      <td>Qwen2.5-32B-Instruct</td>
      <td style="text-align:center;">61.9</td>
      <td style="text-align:center;">24.5</td>
      <td style="text-align:center;">9.7</td>
      <td style="text-align:center;">11.8</td>
      <td style="text-align:center;">17.9</td>
      <td style="text-align:center;">0.03</td>
      <td style="text-align:center;">3.8</td>
      <td style="text-align:center;"><strong>19.7</strong></td>
      <td style="text-align:center;">6.0</td>
      <td style="text-align:center;">4.3</td>
      <td style="text-align:center;">3.6</td>
      <td style="text-align:center;">28.3</td>
    </tr>
    <tr>
      <td>Qwen3-8B (Thinking)</td>
      <td style="text-align:center;">91.7</td>
      <td style="text-align:center;">41.9</td>
      <td style="text-align:center;">8.3</td>
      <td style="text-align:center;">10.3</td>
      <td style="text-align:center;">65.8</td>
      <td style="text-align:center;">0.40</td>
      <td style="text-align:center;">23.3</td>
      <td style="text-align:center;">12.9</td>
      <td style="text-align:center;">6.3</td>
      <td style="text-align:center;">2.2</td>
      <td style="text-align:center;">5.6</td>
      <td style="text-align:center;">25.1</td>
    </tr>
    <tr>
      <td>Qwen3-8B</td>
      <td style="text-align:center;">26.2</td>
      <td style="text-align:center;">10.1</td>
      <td style="text-align:center;">2.6</td>
      <td style="text-align:center;">3.8</td>
      <td style="text-align:center;">139.5</td>
      <td style="text-align:center;">2.52</td>
      <td style="text-align:center;">113.3</td>
      <td style="text-align:center;">8.0</td>
      <td style="text-align:center;">6.3</td>
      <td style="text-align:center;">2.3</td>
      <td style="text-align:center;">4.1</td>
      <td style="text-align:center;">23.8</td>
    </tr>
    <tr>
      <td>Qwen3-32B (Thinking)</td>
      <td style="text-align:center;"><strong>100.0</strong></td>
      <td style="text-align:center;"><strong>45.2</strong></td>
      <td style="text-align:center;">8.4</td>
      <td style="text-align:center;">9.6</td>
      <td style="text-align:center;">23.0</td>
      <td style="text-align:center;">0.02</td>
      <td style="text-align:center;">2.7</td>
      <td style="text-align:center;">18.6</td>
      <td style="text-align:center;">5.2</td>
      <td style="text-align:center;">4.1</td>
      <td style="text-align:center;"><strong>0.0</strong></td>
      <td style="text-align:center;">8.6</td>
    </tr>
    <tr>
      <td>Qwen3-32B</td>
      <td style="text-align:center;">82.1</td>
      <td style="text-align:center;">35.0</td>
      <td style="text-align:center;"><strong>13.2</strong></td>
      <td style="text-align:center;">11.9</td>
      <td style="text-align:center;">22.7</td>
      <td style="text-align:center;">0.42</td>
      <td style="text-align:center;">14.8</td>
      <td style="text-align:center;">15.0</td>
      <td style="text-align:center;">7.0</td>
      <td style="text-align:center;">3.2</td>
      <td style="text-align:center;">6.5</td>
      <td style="text-align:center;">20.3</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B (Thinking)</td>
      <td style="text-align:center;">81.0</td>
      <td style="text-align:center;">36.8</td>
      <td style="text-align:center;">10.9</td>
      <td style="text-align:center;"><strong>12.0</strong></td>
      <td style="text-align:center;">46.9</td>
      <td style="text-align:center;">0.43</td>
      <td style="text-align:center;">12.1</td>
      <td style="text-align:center;">19.0</td>
      <td style="text-align:center;">6.3</td>
      <td style="text-align:center;">4.7</td>
      <td style="text-align:center;">3.8</td>
      <td style="text-align:center;">12.1</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B</td>
      <td style="text-align:center;">46.4</td>
      <td style="text-align:center;">16.9</td>
      <td style="text-align:center;">6.1</td>
      <td style="text-align:center;">6.7</td>
      <td style="text-align:center;">54.2</td>
      <td style="text-align:center;">0.64</td>
      <td style="text-align:center;">16.7</td>
      <td style="text-align:center;">18.5</td>
      <td style="text-align:center;">6.4</td>
      <td style="text-align:center;">4.8</td>
      <td style="text-align:center;">1.9</td>
      <td style="text-align:center;">23.1</td>
    </tr>
    <tr>
      <td>LLaMA-3.1-8B-Instruct</td>
      <td style="text-align:center;">98.8</td>
      <td style="text-align:center;">25.8</td>
      <td style="text-align:center;">2.8</td>
      <td style="text-align:center;">3.3</td>
      <td style="text-align:center;">4.4</td>
      <td style="text-align:center;">0.02</td>
      <td style="text-align:center;">2.7</td>
      <td style="text-align:center;">12.3</td>
      <td style="text-align:center;"><strong>8.0</strong></td>
      <td style="text-align:center;"><strong>6.3</strong></td>
      <td style="text-align:center;">58.2</td>
      <td style="text-align:center;">10.9</td>
    </tr>
  </tbody>
</table>



 Evaluation results on RAVine, with a maximum context length of 32k and the index built by BM25:


<table>
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th colspan="4" style="text-align:center;">Report Quality</th>
      <th colspan="3" style="text-align:center;">Efficiency</th>
      <th colspan="3" style="text-align:center;">Search</th>
      <th colspan="2" style="text-align:center;">Fetch</th>
    </tr>
    <tr>
      <th style="text-align:center;">Rate (↑)</th>
      <th style="text-align:center;">Comp. (↑)</th>
      <th style="text-align:center;">Rec. (↑)</th>
      <th style="text-align:center;">Prec. (↑)</th>
      <th style="text-align:center;">Latency (↓)</th>
      <th style="text-align:center;">Cost (↓)</th>
      <th style="text-align:center;">Turns</th>
      <th style="text-align:center;">Prec. (↑)</th>
      <th style="text-align:center;">Rec. (↑)</th>
      <th style="text-align:center;">Gain (↑)</th>
      <th style="text-align:center;">URL Err. (↓)</th>
      <th style="text-align:center;">Prec. (↑)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-7B-Instruct</td>
      <td style="text-align:center;">22.6</td>
      <td style="text-align:center;">6.4</td>
      <td style="text-align:center;">4.4</td>
      <td style="text-align:center;">6.3</td>
      <td style="text-align:center;">6.9</td>
      <td style="text-align:center;">0.02</td>
      <td style="text-align:center;">2.9</td>
      <td style="text-align:center;">24.6</td>
      <td style="text-align:center;">12.3</td>
      <td style="text-align:center;">6.9</td>
      <td style="text-align:center;">27.1</td>
      <td style="text-align:center;">29.2</td>
    </tr>
    <tr>
      <td>Qwen2.5-32B-Instruct</td>
      <td style="text-align:center;">70.2</td>
      <td style="text-align:center;">26.2</td>
      <td style="text-align:center;">13.0</td>
      <td style="text-align:center;">17.1</td>
      <td style="text-align:center;">16.6</td>
      <td style="text-align:center;">0.04</td>
      <td style="text-align:center;">3.9</td>
      <td style="text-align:center;"><strong>24.7</strong></td>
      <td style="text-align:center;">12.9</td>
      <td style="text-align:center;">6.9</td>
      <td style="text-align:center;">5.1</td>
      <td style="text-align:center;"><strong>49.5</strong></td>
    </tr>
    <tr>
      <td>Qwen3-8B (Thinking)</td>
      <td style="text-align:center;">63.1</td>
      <td style="text-align:center;">25.9</td>
      <td style="text-align:center;">8.3</td>
      <td style="text-align:center;">11.8</td>
      <td style="text-align:center;">10.9</td>
      <td style="text-align:center;">0.04</td>
      <td style="text-align:center;">5.6</td>
      <td style="text-align:center;">21.4</td>
      <td style="text-align:center;">10.8</td>
      <td style="text-align:center;">4.6</td>
      <td style="text-align:center;">6.6</td>
      <td style="text-align:center;">28.0</td>
    </tr>
    <tr>
      <td>Qwen3-8B</td>
      <td style="text-align:center;">15.5</td>
      <td style="text-align:center;">7.9</td>
      <td style="text-align:center;">3.5</td>
      <td style="text-align:center;">4.5</td>
      <td style="text-align:center;">6.1</td>
      <td style="text-align:center;">0.14</td>
      <td style="text-align:center;">13.2</td>
      <td style="text-align:center;">18.5</td>
      <td style="text-align:center;">9.3</td>
      <td style="text-align:center;">3.9</td>
      <td style="text-align:center;">1.8</td>
      <td style="text-align:center;">34.6</td>
    </tr>
    <tr>
      <td>Qwen3-32B (Thinking)</td>
      <td style="text-align:center;">91.7</td>
      <td style="text-align:center;"><strong>40.9</strong></td>
      <td style="text-align:center;">10.2</td>
      <td style="text-align:center;">12.5</td>
      <td style="text-align:center;">20.6</td>
      <td style="text-align:center;">0.04</td>
      <td style="text-align:center;">3.3</td>
      <td style="text-align:center;">20.3</td>
      <td style="text-align:center;">6.9</td>
      <td style="text-align:center;">4.8</td>
      <td style="text-align:center;"><strong>0.0</strong></td>
      <td style="text-align:center;">31.6</td>
    </tr>
    <tr>
      <td>Qwen3-32B</td>
      <td style="text-align:center;">57.1</td>
      <td style="text-align:center;">26.7</td>
      <td style="text-align:center;"><strong>13.2</strong></td>
      <td style="text-align:center;">14.5</td>
      <td style="text-align:center;">10.1</td>
      <td style="text-align:center;">0.09</td>
      <td style="text-align:center;">7.5</td>
      <td style="text-align:center;">23.6</td>
      <td style="text-align:center;">13.8</td>
      <td style="text-align:center;">6.3</td>
      <td style="text-align:center;">9.8</td>
      <td style="text-align:center;">34.0</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B (Thinking)</td>
      <td style="text-align:center;">76.2</td>
      <td style="text-align:center;">32.6</td>
      <td style="text-align:center;">12.6</td>
      <td style="text-align:center;"><strong>18.5</strong></td>
      <td style="text-align:center;">23.3</td>
      <td style="text-align:center;">0.09</td>
      <td style="text-align:center;">6.0</td>
      <td style="text-align:center;">22.5</td>
      <td style="text-align:center;">11.4</td>
      <td style="text-align:center;">5.5</td>
      <td style="text-align:center;"><strong>0.0</strong></td>
      <td style="text-align:center;">45.7</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B</td>
      <td style="text-align:center;">58.3</td>
      <td style="text-align:center;">23.2</td>
      <td style="text-align:center;">12.3</td>
      <td style="text-align:center;">12.7</td>
      <td style="text-align:center;">9.1</td>
      <td style="text-align:center;">0.13</td>
      <td style="text-align:center;">9.1</td>
      <td style="text-align:center;">17.9</td>
      <td style="text-align:center;">11.3</td>
      <td style="text-align:center;">4.5</td>
      <td style="text-align:center;">1.1</td>
      <td style="text-align:center;">48.9</td>
    </tr>
    <tr>
      <td>LLaMA-3.1-8B-Instruct</td>
      <td style="text-align:center;"><strong>97.6</strong></td>
      <td style="text-align:center;">22.1</td>
      <td style="text-align:center;">3.0</td>
      <td style="text-align:center;">2.5</td>
      <td style="text-align:center;">4.5</td>
      <td style="text-align:center;">0.02</td>
      <td style="text-align:center;">2.6</td>
      <td style="text-align:center;">21.1</td>
      <td style="text-align:center;"><strong>14.8</strong></td>
      <td style="text-align:center;"><strong>8.6</strong></td>
      <td style="text-align:center;">46.8</td>
      <td style="text-align:center;">25.5</td>
    </tr>
  </tbody>
</table>



 Evaluation results on RAVine, with a maximum context length of 128k and the index built by BM25:


<table>
  <thead>
    <tr>
      <th rowspan="2"></th>
      <th colspan="4" style="text-align:center;">Report Quality</th>
      <th colspan="3" style="text-align:center;">Efficiency</th>
      <th colspan="3" style="text-align:center;">Search</th>
      <th colspan="2" style="text-align:center;">Fetch</th>
    </tr>
    <tr>
      <th style="text-align:center;">Rate (↑)</th>
      <th style="text-align:center;">Comp. (↑)</th>
      <th style="text-align:center;">Rec. (↑)</th>
      <th style="text-align:center;">Prec. (↑)</th>
      <th style="text-align:center;">Latency (↓)</th>
      <th style="text-align:center;">Cost (↓)</th>
      <th style="text-align:center;">Turns</th>
      <th style="text-align:center;">Prec. (↑)</th>
      <th style="text-align:center;">Rec. (↑)</th>
      <th style="text-align:center;">Gain (↑)</th>
      <th style="text-align:center;">URL Err. (↓)</th>
      <th style="text-align:center;">Prec. (↑)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Qwen2.5-7B-Instruct</td>
      <td style="text-align:center;">7.1</td>
      <td style="text-align:center;">1.4</td>
      <td style="text-align:center;">0.0</td>
      <td style="text-align:center;">0.0</td>
      <td style="text-align:center;">21.2</td>
      <td style="text-align:center;">0.01</td>
      <td style="text-align:center;">1.8</td>
      <td style="text-align:center;">10.5</td>
      <td style="text-align:center;">3.2</td>
      <td style="text-align:center;">2.1</td>
      <td style="text-align:center;">66.7</td>
      <td style="text-align:center;">0.0</td>
    </tr>
    <tr>
      <td>Qwen2.5-32B-Instruct</td>
      <td style="text-align:center;">73.8</td>
      <td style="text-align:center;">28.7</td>
      <td style="text-align:center;"><strong>16.8</strong></td>
      <td style="text-align:center;"><strong>20.4</strong></td>
      <td style="text-align:center;">16.8</td>
      <td style="text-align:center;">0.04</td>
      <td style="text-align:center;">3.9</td>
      <td style="text-align:center;"><strong>26.9</strong></td>
      <td style="text-align:center;">15.1</td>
      <td style="text-align:center;">7.5</td>
      <td style="text-align:center;">5.3</td>
      <td style="text-align:center;"><strong>43.0</strong></td>
    </tr>
    <tr>
      <td>Qwen3-8B (Thinking)</td>
      <td style="text-align:center;">90.5</td>
      <td style="text-align:center;">40.8</td>
      <td style="text-align:center;">14.4</td>
      <td style="text-align:center;">17.8</td>
      <td style="text-align:center;">26.7</td>
      <td style="text-align:center;">0.23</td>
      <td style="text-align:center;">13.4</td>
      <td style="text-align:center;">15.8</td>
      <td style="text-align:center;">12.8</td>
      <td style="text-align:center;">3.8</td>
      <td style="text-align:center;">4.8</td>
      <td style="text-align:center;">32.5</td>
    </tr>
    <tr>
      <td>Qwen3-8B</td>
      <td style="text-align:center;">19.0</td>
      <td style="text-align:center;">6.3</td>
      <td style="text-align:center;">5.2</td>
      <td style="text-align:center;">5.4</td>
      <td style="text-align:center;">39.2</td>
      <td style="text-align:center;">1.87</td>
      <td style="text-align:center;">81.0</td>
      <td style="text-align:center;">7.4</td>
      <td style="text-align:center;">12.2</td>
      <td style="text-align:center;">1.4</td>
      <td style="text-align:center;">5.0</td>
      <td style="text-align:center;">38.4</td>
    </tr>
    <tr>
      <td>Qwen3-32B (Thinking)</td>
      <td style="text-align:center;"><strong>100.0</strong></td>
      <td style="text-align:center;"><strong>47.5</strong></td>
      <td style="text-align:center;">10.5</td>
      <td style="text-align:center;">12.3</td>
      <td style="text-align:center;">22.6</td>
      <td style="text-align:center;">0.03</td>
      <td style="text-align:center;">2.8</td>
      <td style="text-align:center;">17.6</td>
      <td style="text-align:center;">9.5</td>
      <td style="text-align:center;">6.0</td>
      <td style="text-align:center;">8.5</td>
      <td style="text-align:center;">19.1</td>
    </tr>
    <tr>
      <td>Qwen3-32B</td>
      <td style="text-align:center;">76.2</td>
      <td style="text-align:center;">31.1</td>
      <td style="text-align:center;">13.6</td>
      <td style="text-align:center;">17.6</td>
      <td style="text-align:center;">20.6</td>
      <td style="text-align:center;">0.40</td>
      <td style="text-align:center;">13.3</td>
      <td style="text-align:center;">17.3</td>
      <td style="text-align:center;">15.5</td>
      <td style="text-align:center;">4.8</td>
      <td style="text-align:center;">9.2</td>
      <td style="text-align:center;">29.7</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B (Thinking)</td>
      <td style="text-align:center;">75.0</td>
      <td style="text-align:center;">35.0</td>
      <td style="text-align:center;">10.8</td>
      <td style="text-align:center;">12.8</td>
      <td style="text-align:center;">58.9</td>
      <td style="text-align:center;">0.81</td>
      <td style="text-align:center;">18.5</td>
      <td style="text-align:center;">18.7</td>
      <td style="text-align:center;">13.5</td>
      <td style="text-align:center;">5.3</td>
      <td style="text-align:center;"><strong>3.3</strong></td>
      <td style="text-align:center;">29.6</td>
    </tr>
    <tr>
      <td>Qwen3-30B-A3B</td>
      <td style="text-align:center;">50.0</td>
      <td style="text-align:center;">18.5</td>
      <td style="text-align:center;">13.1</td>
      <td style="text-align:center;">14.7</td>
      <td style="text-align:center;">35.3</td>
      <td style="text-align:center;">0.37</td>
      <td style="text-align:center;">12.3</td>
      <td style="text-align:center;">20.5</td>
      <td style="text-align:center;">10.0</td>
      <td style="text-align:center;">4.7</td>
      <td style="text-align:center;">6.5</td>
      <td style="text-align.center;">27.1</td>
    </tr>
    <tr>
      <td>LLaMA-3.1-8B-Instruct</td>
      <td style="text-align:center;">97.6</td>
      <td style="text-align:center;">23.4</td>
      <td style="text-align:center;">4.6</td>
      <td style="text-align:center;">5.0</td>
      <td style="text-align:center;">9.5</td>
      <td style="text-align:center;">0.02</td>
      <td style="text-align:center;">2.6</td>
      <td style="text-align:center;">23.2</td>
      <td style="text-align:center;"><strong>15.6</strong></td>
      <td style="text-align:center;"><strong>10.2</strong></td>
      <td style="text-align:center;">28.6</td>
      <td style="text-align:center;">28.6</td>
    </tr>
  </tbody>
</table>



## Citation

If you find RAVine useful in your research, please cite our paper:

```bibtex
@article{xu2025ravine,
  title={RAVine: Reality-Aligned Evaluation for Agentic Search},
  author={Xu, Yilong and Long, Xiang and Zheng, Zhi and Gao, Jinhua},
  journal={arXiv preprint arXiv:2507.16725},
  year={2025},
  url={https://arxiv.org/abs/2507.16725}
}
```
