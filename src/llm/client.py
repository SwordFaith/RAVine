''' Agent's LLM kernel (OpenAI mode) '''

from dotenv import load_dotenv
from openai import OpenAI
import os
import openai


class LLMChatClient:
    ''' this can used to be a LLM client for agentic search or LLM-as-Judge '''

    def __init__(self,
                 model_name: str,
                 enable_reasoning: bool=False):

        load_dotenv(dotenv_path='.env')

        # key and base url for openai vllm server
        self.openai_api_key = os.getenv('OPENAI_VLLM_SERVER_API_KEY')
        self.openai_api_base = os.getenv('OPENAI_VLLM_SERVER_API_BASE')
        
        # you can replace the above with hardcoded values for testing
        # self.openai_api_key = 'token-abc123'
        # self.openai_api_base = 'http://localhost:8000/v1'
        
        self.client = OpenAI(
            api_key=self.openai_api_key,
            base_url=self.openai_api_base,
        )
        self.model_name = model_name
        self.enable_reasoning = enable_reasoning

    def chat(self, messages, tools=None):
        trial_count = 50
        while trial_count > 0:
            try:
                if tools:
                    completion = self.client.chat.completions.create(messages=messages,
                                                                     model=self.model_name,
                                                                     tools=tools)
                else:
                    completion = self.client.chat.completions.create(messages=messages,
                                                                     model=self.model_name)
                break
            except openai.BadRequestError as e:
                if 'maximum context length' in str(e):
                    print(f'{self.__class__.__name__}: {str(e)}')
                    return None
            except Exception as e:
                print(f'{self.__class__.__name__}: {str(e)}')
                trial_count -= 1
                if trial_count == 0:
                    print(f"Failed to get completion after 50 attempts.")
                    return None
        
        return completion
