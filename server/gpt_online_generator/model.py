import logging
import os

import openai


class Model:
    def __init__(self, *args, **kwargs) -> None:
        _api_key = os.getenv("OPENAI_API_KEY")
        logging.debug("Initializing gpt client")
        self.gpt_client = openai.OpenAI(api_key=_api_key)

    def generate(self, code: str) -> str:
        logging.debug("Generating docstring for code")
        completion = self.gpt_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant to create google-style docstrings for python functions.",
                },
                {
                    "role": "user",
                    "content": f'Create google-style dosctring for function: \n "{code}" \n Return only docstring without quotes.',
                },
            ],
            n=1,
        )
        return completion.choices[0].message.content.strip('"')
