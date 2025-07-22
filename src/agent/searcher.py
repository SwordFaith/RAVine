''' Implementation of agentic research '''

from .prompts import system_prompt, plan_prompt_template
from .history import AgenticMessageHistory
from src.tool.web_search import (
    BM25Client,
    DenseClient,
    search_tool_v2,
    fetch_tool_v2,
    search,
    fetch
)
from src.llm import LLMChatClient
import os
import json


def custom_serializer(obj):
    try:
        # try to convert to dict
        return obj.__dict__
    except AttributeError:
        # forced to be converted to string
        return str(obj)


class AgenticSearcher:
    
    def __init__(self,
                 model_name :str,
                 corpus_name: str,
                 index_path: str,
                 mapper_path: str,
                 search_client_type: str,
                 log_dir: str=None,
                 corpus_path: str=None,
                 embedder_name: str=None,
                 lucene_index_path: str=None,
                 enable_thinking: bool=True):
        self.model_name = model_name
        self.llm = LLMChatClient(model_name=model_name)

        self.search_client_type = search_client_type
        if search_client_type == 'bm25':
            self.search_client = BM25Client(corpus_name, index_path, mapper_path, corpus_path)
        elif search_client_type == 'dense':
            self.search_client = DenseClient(embedder_name, corpus_name, index_path, mapper_path, lucene_index_path, corpus_path)
        
        self.tools = [search_tool_v2, fetch_tool_v2]
        self.tool_func_mapper = {
            'web_search': search,
            'web_fetch': fetch,
        }

        self.log_dir = log_dir
        if self.log_dir is not None:
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
    
    def run(self, input: str, input_id: str, print_logs: bool=False):
        num_turns = 0
        history = AgenticMessageHistory(
            system_prompt, plan_prompt_template.format(question=input)
        )

        completion = self.llm.chat(history.chat, tools=self.tools)
        history.add(completion)
        if print_logs:
            print('-' * 20 + f'Iteration Turn #{num_turns}' + '-' * 20)
            print(completion.choices[0].message.content)
            print('\n')
        num_turns += 1

        while completion.choices[0].message.tool_calls:
            if print_logs:
                print(completion.choices[0].message.tool_calls)
                print('\n')

            for tool_call in completion.choices[0].message.tool_calls:
                try:
                    tool_func = self.tool_func_mapper[tool_call.function.name] # model may call non-existent tool
                    tool_args = json.loads(tool_call.function.arguments)
                    result, raw_result = tool_func(self.search_client, **tool_args)
                    history.add({
                        'role': 'tool',
                        'content': result,
                        'tool_call_id': tool_call.id,
                        'name': tool_call.function.name,
                        'raw_result': raw_result,
                    })
                    if print_logs:
                        print(result + '\n')
                except Exception as e:
                    history.add({
                        'role': 'tool',
                        'content': f'[Tool Calling Error]: {e}',
                        'tool_call_id': tool_call.id,
                        'name': tool_call.function.name,
                        'raw_result': None
                    })
                    if print_logs:
                        print(f'[Tool Calling Error]: {e}\n')
            
            if print_logs:
                print('-' * 60 + '\n')
            
            # next iteration
            completion = self.llm.chat(history.chat, tools=self.tools)
            if completion is None:
                # fail to continue
                break
            history.add(completion)
            if print_logs:
                print('-' * 20 + f'Iteration Turn #{num_turns}' + '-' * 20)
                print(completion.choices[0].message.content)
                print('\n')
            num_turns += 1
        
        if print_logs:
            print('-' * 60 + '\n\n')
        
        if self.log_dir:
            with open(os.path.join(self.log_dir, f'{input_id}.log'), 'w') as file:
                json.dump(history, file, indent=4, default=custom_serializer)
        
        return history


