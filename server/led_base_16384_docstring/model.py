import logging
import pathlib

import transformers
from pydantic import BaseModel

CONFIG_FILE = "config.json"


class _GeneraConfig(BaseModel):
    model: str
    source_prefix: str
    commit: str


class Config(BaseModel):
    general: _GeneraConfig


class Model:
    def __init__(self, *args, **kwargs) -> None:
        module_dir = pathlib.Path(__file__).parent
        json_configuration = (module_dir / CONFIG_FILE).read_text(encoding="utf-8")
        self.config = Config.model_validate_json(json_configuration)
        logging.info(f"Loaded configuration file: {self.config}")
        logging.info(f"Device: {kwargs['device']}")

        self._model = transformers.AutoModelForSeq2SeqLM.from_pretrained(self.config.general.model).to(
            device=kwargs["device"]
        )
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.general.model)

    def generate(self, prompt: str) -> str:
        logging.debug("Generating docstring")
        prompt = self.config.general.source_prefix + prompt

        input_tokens = self._tokenizer(prompt, return_tensors="pt").to(device=self._model.device)

        results = self._model.generate(
            input_ids=input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"]
        )

        output_text = self._tokenizer.decode(results[0], skip_special_tokens=True)
        return output_text
