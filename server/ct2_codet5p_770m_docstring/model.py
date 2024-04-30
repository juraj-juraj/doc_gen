import logging
import pathlib

import ctranslate2
import transformers
from pydantic import BaseModel

CONFIG_FILE = "config.json"


class _GeneraConfig(BaseModel):
    model: str
    source_prefix: str

class _CudaSettings(BaseModel):
    quantization: str


class _CpuSettings(BaseModel):
    quantization: str


class Config(BaseModel):
    general: _GeneraConfig
    cuda: _CudaSettings
    cpu: _CpuSettings


class Model:
    def __init__(self, *args, **kwargs) -> None:
        module_dir = pathlib.Path(__file__).parent
        json_configuration = (module_dir / CONFIG_FILE).read_text(encoding="utf-8")
        self.config = Config.model_validate_json(json_configuration)
        logging.info(f"Loaded configuration file: {self.config}")
        logging.info(f"Device: {kwargs['device']}")

        compute_type = self.config.cuda.quantization if kwargs["device"] == "cuda" else self.config.cpu.quantization
        logging.debug(f"Using compute type: {compute_type}")
        self.model = ctranslate2.Translator(
            str(module_dir / "data"), device=kwargs["device"], compute_type=compute_type,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.config.general.model)

    def generate(self, prompt: str) -> str:
        logging.debug("Generating docstring")
        prompt = self.config.general.source_prefix + prompt

        input_tokens = self.tokenizer.convert_ids_to_tokens(self.tokenizer.encode(prompt))

        results = self.model.translate_batch([input_tokens])

        output_tokens = results[0].hypotheses[0]
        output_text = self.tokenizer.decode(
            self.tokenizer.convert_tokens_to_ids(output_tokens), skip_special_tokens=True
        )

        return output_text
