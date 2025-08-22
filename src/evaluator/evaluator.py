''' Implementation of the agentic research evaluation module '''

from src.data import Query, ScoredNugget, AssignedScoredNugget
from src.tool.web_search import UrlDocMapper
from src.data import RunMetric, ModelMetric
from src.nuggetizer import Nuggetizer
from src.agent import AgenticSearcher
from nltk.tokenize import sent_tokenize
from collections import defaultdict
from dataclasses import asdict
from tqdm import tqdm
import os
import re
import time
import json
import nltk
# nltk.data.path.insert(0, '{path/to/nltk/data}')


class FullEvaluator:
    
    def __init__(self,
                 nuggets_path: str,
                 qrels_path: str,
                 agent_model_name: str,
                 corpus_name: str,
                 index_path: str,
                 mapper_path: str,
                 search_client_type: str,
                 corpus_path: str=None,
                 embedder_name: str=None,
                 lucene_index_path: str=None,
                 enable_thinking: bool=True,
                 eval_model: str='gpt-4.1',
                 log_dir: str=None,):
        self.nuggets_path = nuggets_path
        self.qrels_path = qrels_path
        self.agent_model_name = agent_model_name
        self.corpus_name = corpus_name
        self.corpus_path = corpus_path

        self.agentic_searcher = AgenticSearcher(
            agent_model_name, corpus_name, index_path, mapper_path, search_client_type, log_dir, corpus_path, embedder_name,
            lucene_index_path, enable_thinking
        )
        self.enable_thinking = enable_thinking

        # price (dollar)
        self._search_cost = 0.01 # 150 / 15000  referenced from SerpAPI (https://serpapi.com/pricing)
        # referenced from together.ai (https://www.together.ai/pricing)
        if 'llama' in self.agent_model_name.lower():
            if '8b' in self.agent_model_name.lower():
                self._prompt_token_cost = 0.0000002  # 0.20 per 1M tokens (llama* reference model)
                self._completion_token_cost = 0.0000002
        elif 'qwen' in self.agent_model_name.lower():
            if '8b' in self.agent_model_name.lower() or '7b' in self.agent_model_name.lower():
                self._prompt_token_cost = 0.0000002  # 0.20 per 1M tokens (other model)
                self._completion_token_cost = 0.0000002
            elif '32b' in self.agent_model_name.lower():
                self._prompt_token_cost = 0.0000008  # 0.80 per 1M tokens (other model)
                self._completion_token_cost = 0.0000008
            elif '30b' in self.agent_model_name.lower():
                self._prompt_token_cost = 0.0000006  # 0.60 per 1M tokens (moe model)
                self._completion_token_cost = 0.0000006        

        # the max number of citations to consider during the calculation
        self.max_num_citations = 3

        # the max number of relevant webs during the calculation of search recall
        self.max_num_rel_webs = 3

        # the max number of nuggets (truncate the nugget list to reduce evaluation cost)
        self.max_num_nuggets = 60

        self._load_topics()
        self._load_nuggets_qrels()
        self._load_raw_qrels()

        self.nuggetizer = Nuggetizer(model=eval_model)
        self.url_mapper = UrlDocMapper(corpus_name, mapper_path, corpus_path)

        self.run_log_dir = log_dir

    def _load_topics(self):
        self.topics: list[Query] = []
        with open(self.nuggets_path, 'r') as file:
            for line in file:
                instance = json.loads(line)
                self.topics.append(Query(qid=instance['qid'], text=instance['query']))
    
    def _load_nuggets_qrels(self):
        self.nuggets: dict[str, list[ScoredNugget]] = {}
        self.qrels: dict[str, list] = {} # {'qid': [nugget1_rel_doc_list, nugget2_rel_doc_list, ...]}
        self.flat_qrels = defaultdict(set) # {'qid': [docid1, docid2, ...]}

        with open(self.nuggets_path, 'r') as file:
            for line in file:
                # [{'qid': '', 'query': '', 'nuggets': [{'text': '', 'importance': '', 'segids': [], 'docids': []}, ...]}]
                instance = json.loads(line)
                self.nuggets[instance['qid']] = [
                    ScoredNugget(text=nugget['text'],
                                 importance=nugget['importance'],
                                 docids=nugget['docids'])
                    for nugget in instance['nuggets'][:self.max_num_nuggets]
                ]
                self.qrels[instance['qid']] = [
                    nugget['docids'] for nugget in instance['nuggets'][:self.max_num_nuggets]
                ]
                self.flat_qrels[instance['qid']].update([
                    docid
                    for nugget in instance['nuggets'][:self.max_num_nuggets] for docid in nugget['docids']
                ])
    
    def _load_raw_qrels(self):
        self.raw_flat_qrels = defaultdict(list) # {'qid': [docid1, docid2, ...]}
        with open(self.qrels_path, 'r') as file:
            for line in file:
                item = json.loads(line)
                if int(item['label']) >= 1:
                    self.raw_flat_qrels[item['qid']].append(item['docid'])
    
    @staticmethod
    def parse_sentence_citations(report: str):
        # parse citations at sentence-level
        sentences = sent_tokenize(report)

        parsed_output = []
        citation_pattern = re.compile(r'\[([^\]]+?)\]\((https?://[^\)]+?)\)') # parse citations: [title](url)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            citations = []
            for match in citation_pattern.finditer(sentence):
                title, url = match.groups()
                citations.append({'title': title, 'url': url})
            parsed_output.append({
                'sentence': sentence,
                'citations': citations
            })

        return parsed_output
    
    @staticmethod
    def parse_block_citations(report: str):
        citation_pattern = re.compile(r'\((?:\[\s*.*?\s*\]\(.*?\))(?:\s*;\s*\[.*?\]\(.*?\))*\)')
        blocks = []
        last_citation_end = 0

        for match in citation_pattern.finditer(report):
            citation_text = match.group(0)
            citation_start, citation_end = match.span()
            # extract block: the content from the previous citation to the current citation
            raw_block = report[last_citation_end: citation_start]
            block_text = raw_block.strip(" \n\t.")  # remove leading/trailing spaces and punctuation such as periods
            if block_text:
                citations = re.findall(r'\[(.*?)\]\((.*?)\)', citation_text)
                citation_list = [{'title': title.strip(), 'url': url.strip()} for title, url in citations]
                blocks.append((block_text, citation_list))
            last_citation_end = citation_end
        
        if last_citation_end < len(report):
            # process the last block
            tail_block = report[last_citation_end:].strip(" \n\t.")
            if tail_block:
                blocks.append((tail_block, []))
        
        return blocks # [(block_text. citation_list), ...]
    
    def run_eval(self):
        result_records = [] 
        mm = ModelMetric()

        for topic in tqdm(self.topics):
            rm = RunMetric()

            start_time = time.time()
            history = self.agentic_searcher.run(input=topic.text, input_id=topic.qid)
            end_time = time.time()

            ### system-level efficiency assessment
            rm.latency = end_time - start_time
            for message in history.eval:
                if message['role'] == 'assistant':
                    rm.iteration_turns += 1.0
                    rm.cost += self._prompt_token_cost * message['prompt_tokens']
                    rm.cost += self._completion_token_cost * message['completion_tokens']
                    rm.new_tokens_count += message['completion_tokens']
                elif message['role'] == 'tool':
                    tool_type = message['name']
                    if tool_type == 'web_search':
                        rm.cost += self._search_cost
                    elif tool_type == 'web_fetch':
                        pass # web_fetch is free
            mm.avg_latency += rm.latency
            mm.avg_cost += rm.cost
            mm.avg_new_tokens_count += rm.new_tokens_count
            mm.avg_iteration_turns += rm.iteration_turns


            ### tool-level correctness assessment
            is_first_tool_call = True
            qrels = self.qrels[topic.qid]
            flat_qrels = self.flat_qrels[topic.qid]
            raw_flat_qrels = self.raw_flat_qrels[topic.qid]
            docids_in_process = set() # record searched docids in iterative process
            total_num_seen_web_pages = 0. # record the total seen web pages by the search tool
            for message in history.eval:
                if message['role'] == 'assistant' and message['tool_calls']:
                    for tool_call in message['tool_calls']:
                        rm.num_tool_call += 1.0

                        if is_first_tool_call:
                            is_first_tool_call = False
                            if tool_call.function.name != 'web_search':
                                rm.tool_choice_error += 1.0
                        
                        tool_args = json.loads(tool_call.function.arguments)
                        if tool_call.function.name == 'web_search':
                            rm.num_search_call += 1.0
                            if not ('query' in tool_args and isinstance(tool_args['query'], str) and 'num_results' in tool_args and isinstance(tool_args['num_results'], int)):
                                rm.search_params_error += 1.0
                            elif not set(tool_args.keys()).issubset(['query', 'num_results']):
                                rm.search_extra_params_error += 1.0
                            else:
                                # correct search tool call
                                rm.num_correct_search_call += 1.0
                        elif tool_call.function.name == 'web_fetch':
                            rm.num_fetch_call += 1.0
                            if not ('url' in tool_args and isinstance(tool_args['url'], str)):
                                rm.fetch_params_error += 1.0
                            elif not set(tool_args.keys()).issubset(['url']):
                                rm.fetch_extra_params_error += 1.0
                            elif not self.url_mapper.get_docid_by_url(tool_args['url']):
                                rm.fetch_url_error += 1.0
                            else:
                                # correct fetch tool call
                                rm.num_correct_fetch_call += 1.0
                                if self.url_mapper.get_docid_by_url(tool_args['url']) in flat_qrels:
                                    rm.fetch_precision += 1.0
                                if self.url_mapper.get_docid_by_url(tool_args['url']) in raw_flat_qrels:
                                    rm.raw_fetch_precision += 1.0
                        else:
                            # wrong tool name
                            rm.tool_name_error += 1.0
                        
                elif message['role'] == 'tool':
                    if message['name'] == 'web_search':
                        raw_result = message['raw_result'] # [{title, headings, url}]
                        if raw_result is None:
                            rm.search_gain_list.append(0.0)
                            rm.raw_search_gain_list.append(0.0)
                            continue
                        total_num_seen_web_pages += len(raw_result)
                        if len(rm.search_gain_list) == 0:
                            # first search call
                            # see how many qrels `docids_in_process` hits, and then divide it by the total number of qrels
                            docids_in_process.update([self.url_mapper.get_docid_by_url(item['url']) for item in raw_result])
                            rm.search_gain_list.append(sum([1.0 for docid in docids_in_process if docid in flat_qrels]))
                            rm.raw_search_gain_list.append(sum([1.0 for docid in docids_in_process if docid in raw_flat_qrels]))
                        else:
                            # See how many new qrels are hit compared to `docids_in_process`, and then divide it by the total number of qrels
                            seen_rel_docids = [docid for docid in docids_in_process if docid in flat_qrels]
                            seen_raw_rel_docids = [docid for docid in docids_in_process if docid in raw_flat_qrels]
                            docids_in_process.update([self.url_mapper.get_docid_by_url(item['url']) for item in raw_result])
                            rm.search_gain_list.append(sum([1.0 for docid in docids_in_process if docid in flat_qrels and docid not in seen_rel_docids]))
                            rm.raw_search_gain_list.append(sum([1.0 for docid in docids_in_process if docid in raw_flat_qrels and docid not in seen_raw_rel_docids]))
            
            # calculate search recall using the number of hits in qrels in `docids_in_process` and take cap
            for nugget_docids in qrels:
                num_hits = len(set(nugget_docids) & docids_in_process)
                if len(nugget_docids) > self.max_num_rel_webs:
                    rm.search_recall += min(num_hits / self.max_num_rel_webs, 1)
                else:
                    rm.search_recall += (num_hits / len(nugget_docids))
            rm.search_recall /= len(qrels)
            rm.raw_search_recall = sum([1.0 for docid in docids_in_process if docid in raw_flat_qrels]) / len(raw_flat_qrels)

            # calculate search precision using the number of hits in qrels in `docids_in_process` and `total_web_pages`
            # discourage repeated searches of the same web page to avoid reward hacking
            rm.search_precision = sum([1.0 for docid in docids_in_process if docid in flat_qrels]) / total_num_seen_web_pages if total_num_seen_web_pages > 0 else 0
            rm.raw_search_precision = sum([1.0 for docid in docids_in_process if docid in raw_flat_qrels]) / total_num_seen_web_pages if total_num_seen_web_pages > 0 else 0

            # use `search_gain_list` and `raw_search_gain_list` to calculate `search_gain` and `raw_search_gain` respectively, and take cap
            search_gain_base = 0.
            used_docids = set()
            for nugget_docids in qrels:
                overlap = len(used_docids & set(nugget_docids))
                unused = len(nugget_docids) - overlap
                search_gain_base += min(unused, self.max_num_rel_webs)
                used_docids.update(nugget_docids)
            rm.search_gain_list = [x / search_gain_base for x in rm.search_gain_list]
            rm.raw_search_gain_list = [x / len(raw_flat_qrels) for x in rm.raw_search_gain_list]
            rm.avg_search_gain = sum(rm.search_gain_list) / len(rm.search_gain_list) if len(rm.search_gain_list) > 0 else 0
            rm.avg_raw_search_gain = sum(rm.raw_search_gain_list) / len(rm.raw_search_gain_list) if len(rm.raw_search_gain_list) > 0 else 0

            
            mm.micro_avg_search_params_error += rm.search_params_error
            mm.micro_avg_search_extra_params_error += rm.search_extra_params_error
            mm.micro_avg_fetch_params_error += rm.fetch_params_error
            mm.micro_avg_fetch_extra_params_error += rm.fetch_extra_params_error
            mm.micro_avg_fetch_url_error += rm.fetch_url_error

            mm.macro_avg_search_params_error += ((rm.search_params_error / rm.num_search_call) if rm.num_search_call > 0 else 0)
            mm.macro_avg_search_extra_params_error += ((rm.search_extra_params_error / rm.num_search_call) if rm.num_search_call > 0 else 0)
            mm.macro_avg_fetch_params_error += ((rm.fetch_params_error / rm.num_fetch_call) if rm.num_fetch_call > 0 else 0)
            mm.macro_avg_fetch_extra_params_error += ((rm.fetch_extra_params_error / rm.num_fetch_call) if rm.num_fetch_call > 0 else 0)
            mm.macro_avg_fetch_url_error += ((rm.fetch_url_error / rm.num_fetch_call) if rm.num_fetch_call > 0 else 0)

            mm.micro_avg_num_correct_search_call += rm.num_correct_search_call
            mm.micro_avg_num_correct_fetch_call += rm.num_correct_fetch_call
            mm.macro_avg_num_correct_search_call += ((rm.num_correct_search_call / rm.num_search_call) if rm.num_search_call > 0 else 0)
            mm.macro_avg_num_correct_fetch_call += ((rm.num_correct_fetch_call / rm.num_fetch_call) if rm.num_fetch_call > 0 else 0)

            mm.macro_avg_search_recall += rm.search_recall
            mm.macro_avg_search_precision += rm.search_precision
            mm.micro_avg_fetch_precision += rm.fetch_precision
            mm.macro_avg_fetch_precision += ((rm.fetch_precision / rm.num_fetch_call) if rm.num_fetch_call > 0 else 0)

            mm.macro_avg_raw_search_recall += rm.raw_search_recall
            mm.macro_avg_raw_search_precision += rm.raw_search_precision
            mm.micro_avg_raw_fetch_precision += rm.raw_fetch_precision
            mm.macro_avg_raw_fetch_precision += ((rm.raw_fetch_precision / rm.num_fetch_call) if rm.num_fetch_call > 0 else 0)

            mm.avg_avg_search_gain += rm.avg_search_gain
            mm.avg_avg_raw_search_gain += rm.avg_raw_search_gain

            mm.avg_tool_choice_error += rm.tool_choice_error
            mm.avg_num_search_call += rm.num_search_call
            mm.avg_num_fetch_call += rm.num_fetch_call
            mm.micro_avg_tool_name_error += rm.tool_name_error
            mm.macro_avg_tool_name_error += ((rm.tool_name_error / rm.num_tool_call) if rm.num_tool_call > 0 else 0)
            mm.total_num_search_call += rm.num_search_call
            mm.total_num_fetch_call += rm.num_fetch_call
            mm.total_num_tool_call += rm.num_tool_call


            ### final result (report) extraction
            report = history.report

            if report is None:
                # unfinished run -> failed
                rm.is_failed = True
                mm.num_failed_runs += 1.0
                result_records.append({
                    'is_failed': rm.is_failed,
                    'qid': topic.qid,
                    'query': topic.text,
                    'task_completeness': 0.,
                    'citation_recall': 0.,
                    'citation_precision': 0.,
                    'internal_knowledge_completeness': 0,
                    'latency': rm.latency,
                    'cost': rm.cost,
                    'new_tokens_count': rm.new_tokens_count,
                    'iteration_turns': rm.iteration_turns,
                    'search_params_error': rm.search_params_error,
                    'search_extra_params_error': rm.search_extra_params_error,
                    'fetch_params_error': rm.fetch_params_error,
                    'fetch_extra_params_error': rm.fetch_extra_params_error,
                    'fetch_url_error': rm.fetch_url_error,
                    'tool_choice_error': rm.tool_choice_error,
                    'tool_name_error': rm.tool_name_error,
                    'search_recall': rm.search_recall,
                    'search_precision': rm.search_precision,
                    'fetch_precision': rm.fetch_precision,
                    'raw_search_recall': rm.raw_search_recall,
                    'raw_search_precision': rm.raw_search_precision,
                    'raw_fetch_precision': rm.raw_fetch_precision,
                    'search_gain_list': rm.search_gain_list,
                    'raw_search_gain_list': rm.raw_search_gain_list,
                    'avg_search_gain': rm.avg_search_gain,
                    'avg_raw_search_gain': rm.avg_raw_search_gain,
                    'num_search_call': rm.num_search_call,
                    'num_fetch_call': rm.num_fetch_call,
                    'num_tool_call': rm.num_tool_call,
                    'num_correct_search_call': rm.num_correct_search_call,
                    'num_correct_fetch_call': rm.num_correct_fetch_call,
                    'report': None,
                    'blocks': None,
                    'block_nuggets_assignment': None,
                    'global_nuggets_assignment': None,
                })
                continue
            else:
                rm.report = report


            ### end-to-end system-level quality assessment: information coverage evaluation and citation evaluation
            if self.enable_thinking:
                history = history.parse_thinking()

            rm.blocks = self.parse_block_citations(rm.report) # [(block_text, citations), ...]
            rm.global_nuggets_assignment = ['not_support' for _ in self.nuggets[topic.qid]] # record whether the nugget is hit or not (for information coverage evaluation)

            num_cited_blocks = 0.
            for block_text, citations in rm.blocks:
                assigned_nuggets: list[AssignedScoredNugget] = self.nuggetizer.assign(query=topic, block_text=block_text, nuggets=self.nuggets[topic.qid])
                rm.block_nuggets_assignment.append([asdict(assigned_nugget) for assigned_nugget in assigned_nuggets]) # use AssignedScoredNugget(**data) to reload
                local_nuggets_hit_map = [] # record whether the assigned nugget is hit or not (for citation evaluation)
                for i, nugget in enumerate(assigned_nuggets):
                    if nugget.importance == 'vital':
                        if nugget.assignment == 'support':
                            rm.global_nuggets_assignment[i] = 'support'
                            local_nuggets_hit_map.append(1)
                        elif nugget.assignment == 'partial_support':
                            rm.global_nuggets_assignment[i] = 'partial_support' if rm.global_nuggets_assignment[i] != 'support' else 'support'
                            local_nuggets_hit_map.append(1)
                        elif nugget.assignment == 'not_support':
                            local_nuggets_hit_map.append(0)
                        else:
                            local_nuggets_hit_map.append(0) # error: wrong assignment label
                    elif nugget.importance == 'okay':
                        if nugget.assignment == 'support':
                            rm.global_nuggets_assignment[i] = 'support'
                            local_nuggets_hit_map.append(1)
                        elif nugget.assignment == 'partial_support':
                            rm.global_nuggets_assignment[i] = 'partial_support' if rm.global_nuggets_assignment[i] != 'support' else 'support'
                            local_nuggets_hit_map.append(1)
                        elif nugget.assignment == 'not_support':
                            local_nuggets_hit_map.append(0)
                        else:
                            local_nuggets_hit_map.append(0) # error: wrong assignment label
                    else:
                        # error: wrong nugget label
                        rm.global_nuggets_assignment[i] = 'skip'
                        local_nuggets_hit_map.append(0)

                ########## citations recall & precision ##########
                if len(citations) == 0:
                    # usually, the last block has no citations
                    continue
                num_cited_blocks += 1.0


                # calculating the weight of gold citation
                original_gold_docids = set(docid for nugget, hit in zip(assigned_nuggets, local_nuggets_hit_map) if hit == 1 for docid in nugget.docids)
                gold_docids = original_gold_docids & docids_in_process # Only those seen by the model will be the gold citation candidates
                gold_docids_weights = [] # weight for each gold_docid
                for gold_docid in gold_docids:
                    gold_docids_weights.append(sum([1.0 for nugget, hit in zip(assigned_nuggets, local_nuggets_hit_map) if hit == 1 and gold_docid in nugget.docids]))

                pred_docids = set(self.url_mapper.get_docid_by_url(item['url']) for item in citations)
                hit_docids = set(pred_docids) & set(gold_docids)
                gold_dict = dict(zip(gold_docids, gold_docids_weights))
                if len(gold_docids) > self.max_num_citations:
                    # hitting 3 gold citations does not necessarily mean full marks, only choosing the best 3 is full marks
                    top_k_weights = sorted(gold_docids_weights, reverse=True)[:self.max_num_citations]
                    total_top_k_weight = sum(top_k_weights)
                    hit_weight = sum(gold_dict[docid] for docid in hit_docids)
                    recall = min(hit_weight / total_top_k_weight, 1.0) if total_top_k_weight > 0 else 0.0
                else:
                    total_weight = sum(gold_docids_weights)
                    hit_weight = sum(gold_dict[docid] for docid in hit_docids)
                    recall = min(hit_weight / total_weight, 1.0) if total_weight > 0 else 0.0
                
                precision = len(hit_docids) / len(pred_docids) if len(pred_docids) > 0 else 0
                rm.citation_recall += recall
                rm.citation_precision += precision

                if len(gold_docids) == 0 and len(original_gold_docids) > 0:
                    # this means that no relevant web pages were found during the search, but nuggets were still hit
                    rm.internal_knowledge_completeness += 1.0
        

            rm.citation_recall = rm.citation_recall / num_cited_blocks if num_cited_blocks > 0 else 0
            rm.citation_precision = rm.citation_precision / num_cited_blocks if num_cited_blocks > 0 else 0
            rm.internal_knowledge_completeness = rm.internal_knowledge_completeness / num_cited_blocks if num_cited_blocks > 0 else 0
            mm.avg_citation_recall += rm.citation_recall
            mm.avg_citation_precision += rm.citation_precision
            mm.avg_internal_knowledge_completeness += rm.internal_knowledge_completeness
            
            num_vital_nuggets = 0.
            num_okay_nuggets = 0.
            for nugget, hit in zip(self.nuggets[topic.qid], rm.global_nuggets_assignment):
                if nugget.importance == 'vital':
                    num_vital_nuggets += 1.0
                    if hit == 'support':
                        rm.task_completeness += 1.0
                    elif hit == 'partial_support':
                        rm.task_completeness += 0.5
                    else:
                        pass
                elif nugget.importance == 'okay':
                    num_okay_nuggets += 1.0
                    if hit == 'support':
                        rm.task_completeness += 1.0 * 0.5
                    elif hit == 'partial_support':
                        rm.task_completeness += 0.5 * 0.5
                    else:
                        pass

            rm.task_completeness = rm.task_completeness / (num_vital_nuggets + 0.5 * num_okay_nuggets) if (num_vital_nuggets + 0.5 * num_okay_nuggets) > 0 else 0
            mm.avg_task_completeness += rm.task_completeness

            result_records.append({
                'is_failed': rm.is_failed,
                'qid': topic.qid,
                'query': topic.text,
                'task_completeness': rm.task_completeness,
                'citation_recall': rm.citation_recall,
                'citation_precision': rm.citation_precision,
                'internal_knowledge_completeness': rm.internal_knowledge_completeness,
                'latency': rm.latency,
                'cost': rm.cost,
                'new_tokens_count': rm.new_tokens_count,
                'iteration_turns': rm.iteration_turns,
                'search_params_error': rm.search_params_error,
                'search_extra_params_error': rm.search_extra_params_error,
                'fetch_params_error': rm.fetch_params_error,
                'fetch_extra_params_error': rm.fetch_extra_params_error,
                'fetch_url_error': rm.fetch_url_error,
                'tool_choice_error': rm.tool_choice_error,
                'tool_name_error': rm.tool_name_error,
                'search_recall': rm.search_recall,
                'search_precision': rm.search_precision,
                'fetch_precision': rm.fetch_precision,
                'raw_search_recall': rm.raw_search_recall,
                'raw_search_precision': rm.raw_search_precision,
                'raw_fetch_precision': rm.raw_fetch_precision,
                'search_gain_list': rm.search_gain_list,
                'raw_search_gain_list': rm.raw_search_gain_list,
                'avg_search_gain': rm.avg_search_gain,
                'avg_raw_search_gain': rm.avg_raw_search_gain,
                'num_search_call': rm.num_search_call,
                'num_fetch_call': rm.num_fetch_call,
                'num_tool_call': rm.num_tool_call,
                'num_correct_search_call': rm.num_correct_search_call,
                'num_correct_fetch_call': rm.num_correct_fetch_call,
                'report': rm.report,
                'blocks': rm.blocks,
                'block_nuggets_assignment': rm.block_nuggets_assignment,
                'global_nuggets_assignment': rm.global_nuggets_assignment,
            })
        
        # save the evaluation results
        if self.run_log_dir:
            if not os.path.exists(self.run_log_dir):
                os.makedirs(self.run_log_dir)
            with open(os.path.join(self.run_log_dir, f'eval.jsonl'), 'w') as file:
                for record in result_records:
                    file.write(json.dumps(record) + '\n')
        

        ### sum up the evaluation results
        mm.num_run = len(self.topics)
        mm.num_valid_run = len(self.topics) - mm.num_failed_runs # `valid` means that the final report is generated correctly
        mm.task_finish_rate = mm.num_valid_run / mm.num_run

        mm.avg_task_completeness /= mm.num_valid_run if mm.num_valid_run > 0 else 0.0
        mm.final_task_completeness = mm.avg_task_completeness * mm.task_finish_rate # consider failed runs in calculation
        mm.avg_citation_recall /= mm.num_valid_run if mm.num_valid_run > 0 else 0.0
        mm.avg_citation_precision /= mm.num_valid_run if mm.num_valid_run > 0 else 0.0
        mm.avg_internal_knowledge_completeness /= mm.num_valid_run if mm.num_valid_run > 0 else 0.0

        mm.avg_latency /= mm.num_run
        mm.avg_cost /= mm.num_run
        mm.avg_new_tokens_count /= mm.num_run
        mm.avg_iteration_turns /= mm.num_run

        mm.avg_tool_choice_error /= mm.num_run
        mm.avg_num_search_call /= mm.num_run
        mm.avg_num_fetch_call /= mm.num_run
        mm.micro_avg_tool_name_error /= mm.total_num_tool_call if mm.total_num_tool_call > 0 else 0.0
        mm.macro_avg_tool_name_error /= mm.num_run

        mm.macro_avg_search_precision /= mm.num_run
        mm.micro_avg_fetch_precision /= mm.total_num_fetch_call if mm.total_num_fetch_call > 0 else 0.0
        mm.macro_avg_fetch_precision /= mm.num_run
        mm.macro_avg_raw_search_precision /= mm.num_run
        mm.micro_avg_raw_fetch_precision /= mm.total_num_fetch_call if mm.total_num_fetch_call > 0 else 0.0
        mm.macro_avg_raw_fetch_precision /= mm.num_run

        mm.macro_avg_search_recall /= mm.num_run
        mm.macro_avg_raw_search_recall /= mm.num_run

        mm.avg_avg_search_gain /= mm.num_run
        mm.avg_avg_raw_search_gain /= mm.num_run

        mm.micro_avg_num_correct_search_call /= mm.total_num_search_call if mm.total_num_search_call > 0 else 0.0
        mm.micro_avg_num_correct_fetch_call /= mm.total_num_fetch_call if mm.total_num_fetch_call > 0 else 0.0
        mm.macro_avg_num_correct_search_call /= mm.num_run
        mm.macro_avg_num_correct_fetch_call /= mm.num_run

        mm.micro_avg_search_params_error /= mm.total_num_search_call if mm.total_num_search_call > 0 else 0.0
        mm.micro_avg_search_extra_params_error /= mm.total_num_search_call if mm.total_num_search_call > 0 else 0.0
        mm.micro_avg_fetch_params_error /= mm.total_num_fetch_call if mm.total_num_fetch_call > 0 else 0.0
        mm.micro_avg_fetch_extra_params_error /= mm.total_num_fetch_call if mm.total_num_fetch_call > 0 else 0.0
        mm.micro_avg_fetch_url_error /= mm.total_num_fetch_call if mm.total_num_fetch_call > 0 else 0.0

        mm.macro_avg_search_params_error /= mm.num_run
        mm.macro_avg_search_extra_params_error /= mm.num_run
        mm.macro_avg_fetch_params_error /= mm.num_run
        mm.macro_avg_fetch_extra_params_error /= mm.num_run
        mm.macro_avg_fetch_url_error /= mm.num_run

        return {
            'num_run': mm.num_run,
            'num_valid_run': mm.num_valid_run,
            'num_failed_runs': mm.num_failed_runs,
            'task_finish_rate': mm.task_finish_rate, # table
            'avg_task_completeness': mm.avg_task_completeness,
            'final_task_completeness': mm.final_task_completeness, # table
            'avg_citation_recall': mm.avg_citation_recall, # table
            'avg_citation_precision': mm.avg_citation_precision, # table
            'avg_internal_knowledge_completeness': mm.avg_internal_knowledge_completeness,
            'avg_weighted_internal_knowledge_completeness': mm.avg_internal_knowledge_completeness * mm.task_finish_rate,
            'avg_latency': mm.avg_latency, # table
            'avg_cost': mm.avg_cost, # table
            'avg_new_tokens_count': mm.avg_new_tokens_count,
            'avg_iteration_turns': mm.avg_iteration_turns, # table
            'avg_tool_choice_error': mm.avg_tool_choice_error,
            'avg_num_search_call': mm.avg_num_search_call,
            'avg_num_fetch_call': mm.avg_num_fetch_call,
            'micro_avg_tool_name_error': mm.micro_avg_tool_name_error,
            'macro_avg_tool_name_error': mm.macro_avg_tool_name_error,
            'macro_avg_search_precision': mm.macro_avg_search_precision, # table
            'micro_avg_fetch_precision': mm.micro_avg_fetch_precision, # table
            'macro_avg_fetch_precision': mm.macro_avg_fetch_precision,
            'macro_avg_raw_search_precision': mm.macro_avg_raw_search_precision,
            'micro_avg_raw_fetch_precision': mm.micro_avg_raw_fetch_precision,
            'macro_avg_raw_fetch_precision': mm.macro_avg_raw_fetch_precision,
            'macro_avg_search_recall': mm.macro_avg_search_recall, # table
            'macro_avg_raw_search_recall': mm.macro_avg_raw_search_recall,
            'avg_avg_search_gain': mm.avg_avg_search_gain, # table
            'avg_avg_raw_search_gain': mm.avg_avg_raw_search_gain,
            'macro_avg_search_f1': 2 * mm.macro_avg_search_precision * mm.macro_avg_search_recall / (mm.macro_avg_search_precision + mm.macro_avg_search_recall),
            'macro_avg_raw_search_f1': 2 * mm.macro_avg_raw_search_precision * mm.macro_avg_raw_search_recall / (mm.macro_avg_raw_search_precision + mm.macro_avg_raw_search_recall),
            'micro_avg_num_correct_search_call': mm.micro_avg_num_correct_search_call,
            'micro_avg_num_correct_fetch_call': mm.micro_avg_num_correct_fetch_call,
            'macro_avg_num_correct_search_call': mm.macro_avg_num_correct_search_call,
            'macro_avg_num_correct_fetch_call': mm.macro_avg_num_correct_fetch_call,
            'micro_avg_search_params_error': mm.micro_avg_search_params_error,
            'micro_avg_search_extra_params_error': mm.micro_avg_search_extra_params_error,
            'micro_avg_fetch_params_error': mm.micro_avg_fetch_params_error,
            'micro_avg_fetch_extra_params_error': mm.micro_avg_fetch_extra_params_error,
            'micro_avg_fetch_url_error': mm.micro_avg_fetch_url_error, # table
            'macro_avg_search_params_error': mm.macro_avg_search_params_error,
            'macro_avg_search_extra_params_error': mm.macro_avg_search_extra_params_error,
            'macro_avg_fetch_params_error': mm.macro_avg_fetch_params_error,
            'macro_avg_fetch_extra_params_error': mm.macro_avg_fetch_extra_params_error,
            'macro_avg_fetch_url_error': mm.macro_avg_fetch_url_error,
            'total_num_search_call': mm.total_num_search_call,
            'total_num_fetch_call': mm.total_num_fetch_call,
            'total_num_tool_call': mm.total_num_tool_call,
        }


