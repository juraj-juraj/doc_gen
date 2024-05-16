import logging
import os
import pathlib

from dotenv import load_dotenv
from predibase import Predibase
from pydantic import BaseModel

CONFIG_FILE = "config.json"


class _GeneralConfig(BaseModel):
    model: str


class Config(BaseModel):
    general: _GeneralConfig


class Model:
    def __init__(self, *args, **kwargs) -> None:
        module_dir = pathlib.Path(__file__).parent
        load_dotenv(dotenv_path=module_dir / ".env", override=True, verbose=True)
        json_configuration = (module_dir / CONFIG_FILE).read_text(encoding="utf-8")
        self.config = Config.model_validate_json(json_configuration)
        _api_key = os.getenv("PREDIBASE_API_KEY")
        logging.debug(f"Initializing predibase model: {self.config.general.model}")
        self._client = Predibase(api_token=_api_key).deployments.client(self.config.general.model)

    def generate(self, code: str) -> str:
        logging.debug("Generating docstring for code")
        prompt = f'Write an appropriate docstring for the following Python function. Return only generated docstring without quotes.\n#Function:\n "{code}" \nGenerated docstring:'
        return self._client.generate(prompt, max_new_tokens=300).generated_text
