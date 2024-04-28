import importlib.util
import logging
import pathlib
import sys
from typing import Protocol, Type


class ModelI(Protocol):

    def __init__(self, *args, **kwargs) -> None: ...
    def generate(self, code: str) -> str:
        """Generates a docstring for the given code.

        Args:
            code (str): Single python function definition

        Returns:
            str: Generated docstring for the given code
        """
        ...


def load_model(model_path: pathlib.Path) -> Type[ModelI]:
    str_model_path = "./" + str(model_path)
    module_name = "model.py"
    try:
        logging.info(f"Loading Model '{str_model_path}'")
        spec = importlib.util.spec_from_file_location(module_name, str(model_path.absolute()) + "/" + module_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module.Model

    except ImportError as e:
        logging.error(e)
        raise RuntimeError(f"Failed to load model '{str_model_path}'")
