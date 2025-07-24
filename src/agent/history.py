''' Message history recording module for agentic research '''

from openai.types.chat.chat_completion import ChatCompletion
import re
import copy


class AgenticMessageHistory:

    def __init__(self, system_prompt: str=None, user_prompt: str=None, messages: list=None):
        self._messages = []

        if system_prompt:
            self._messages.append({
                'role': 'system',
                'content': system_prompt,
            })
        
        if user_prompt:
            self._messages.append({
                'role': 'user',
                'content': user_prompt,
            })
        
        if messages:
            self._messages = copy.deepcopy(messages)
    
    def add(self, completion: ChatCompletion | dict):
        if isinstance(completion, dict):
            # tool call result
            self._messages.append(copy.deepcopy(completion))
        else:
            # chat completion
            self._messages.append({
                'role': 'assistant',
                'content': completion.choices[0].message.content, # usually thinking content if tool_calls exist
                'tool_calls': completion.choices[0].message.tool_calls, # None if no tool_calls
                'prompt_tokens': completion.usage.prompt_tokens, # number of input tokens
                'completion_tokens': completion.usage.completion_tokens, # number of new generated tokens
                'total_tokens': completion.usage.total_tokens, # prompt_tokens + completion_tokens
            })
    
    def clear(self):
        self._messages.clear()
    
    def parse_thinking(self) -> "AgenticMessageHistory":
        parsed_messages = []
        for message in self._messages:
            message_cp = copy.deepcopy(message)
            if message_cp['role'] == 'assistant':
                matches = re.search(r'<think>(.*?)</think>', message_cp['content'], re.DOTALL)
                if not matches:
                    message_cp['thinking_content'] = None
                else:
                    message_cp['thinking_content'] = matches.group(1)
                    message_cp['content'] = message_cp['content'][:matches.start()] + message_cp['content'][matches.end():]
            parsed_messages.append(message_cp)
        
        return AgenticMessageHistory(messages=parsed_messages)
    
    @property
    def report(self):
        # return the last message, which may contain the report
        pattern = r"<report>(.*?)</report>"
        matches = re.findall(pattern, self._messages[-1]['content'], re.DOTALL)
        if matches:
            return matches[-1]  # return the last match, avoiding tags in thinking content
        else:
            return None
    
    @property
    def chat(self):
        res = []
        for m in self._messages:
            if m['role'] == 'assistant':
                res.append({
                    'role': m['role'],
                    'content': m['content'],
                    'tool_calls': m['tool_calls'],
                })
            elif m['role'] == 'tool':
                res.append({
                    'role': m['role'],
                    'content': m['content'],
                    'tool_call_id': m['tool_call_id'],
                    'name': m['name'],
                })
            else:
                res.append(copy.deepcopy(m))
        return res
    
    @property
    def eval(self):
        return copy.deepcopy(self._messages)
