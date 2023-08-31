from concurrent.futures import ThreadPoolExecutor
from typing import List


from opencompass.registry import MODELS
from opencompass.utils.prompt import PromptList

from .base_api import BaseAPIModel

@MODELS.register_module()
class Mock(BaseAPIModel):
    is_api: bool = True

    def __init__(
        self,
        path: str = "Mock",
        max_out_len: int = 2048, 
        max_seq_len: int = 2048, 
        batch_size: int = 8
    ):  # noqa
        super().__init__(path=path)

        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            "internlm/internlm-chat-7b", 
            trust_remote_code=True)
        self.org_ctr = 0
    def generate(
        self,
        inputs,
        max_out_len: int = 2048,
        temperature: float = 0.8,
    ) -> List[str]:
        with ThreadPoolExecutor() as executor:
            results = list(
                executor.map(self._generate, inputs,
                             [temperature] * len(inputs)))
        return results

    def _generate(self, input: str or PromptList, temperature: float) -> str:
        print("======= mocking =======")
        print("input: ", input)
        print("temp:  ", temperature)
        assert isinstance(input, (str, PromptList))

        if isinstance(input, str):
            messages = [{'role': 'user', 'text': input}]
        else:
            messages = []
            for item in input:
                msg = {'text': item['prompt']}
                if item['role'] == 'HUMAN':
                    msg['role'] = 'user'
                elif item['role'] == 'BOT':
                    msg['role'] = 'assistant'
                elif item['role'] == 'SYSTEM':
                    msg['role'] = 'system'
                elif item['role'] == 'user':
                    msg['role'] = 'user'
                elif item['role'] == 'assistant':
                    msg['role'] = 'assistant'
                messages.append(msg)
        return ["Mock Result"]
            

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.
        Args:
            prompt (str): Input string.
        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer(prompt))