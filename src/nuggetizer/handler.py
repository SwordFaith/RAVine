''' LLM handler called by nuggetizer (referenced from https://github.com/castorini/nuggetizer) '''

from .utils import get_openai_api_url_key
from openai import OpenAI

class LLMHandler:
    def __init__(self,
                 model: str,
                 context_size: int = 8192):
        self.model = model
        self.context_size = context_size

        self.api_url, self.api_key = get_openai_api_url_key()
        self.client = self._initialize_client()
    
    def _initialize_client(self):
        return OpenAI(base_url=self.api_url, api_key=self.api_key)
    
    def run(self,
            messages: list[dict[str, str]], 
            temperature: float = 0) -> str:

        while True:
            if "o1" in self.model:
                # System message is not supported for o1 models
                new_messages = messages[1:]
                new_messages[0]["content"] = messages[0]["content"] + "\n" + messages[1]["content"]
                messages = new_messages[:]
                temperature = 1.0
            try:
                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=2048,
                    timeout=30
                )
                response = completion.choices[0].message.content

                return response
            except Exception as e:
                print(f"Error: {str(e)}")

