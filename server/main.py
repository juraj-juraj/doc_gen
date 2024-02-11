import logging
import pathlib
from contextlib import asynccontextmanager
from typing import Literal

import model_loader
from docstring_transformer import annotate_code
from fastapi import FastAPI
from pydantic import BaseModel

CONFIG_FILE = "server_config.json"

docstring_model = None


class AppConfig(BaseModel):
    model_dir: pathlib.Path
    device: Literal["cpu", "cuda"]
    log_level: Literal["info", "debug", "warning", "error", "critical"]


class AnnotateRequest(BaseModel):
    code: str
    overwrite_docstrings: bool = False


@asynccontextmanager
async def load_model(app: FastAPI):
    global docstring_model
    module_dir = pathlib.Path(__file__).parent
    json_configuration = (module_dir / CONFIG_FILE).read_text(encoding="utf-8")
    config = AppConfig.model_validate_json(json_configuration)
    logging.basicConfig(level=logging.getLevelName(config.log_level.upper()))

    logging.info(f"Loaded configuration file: {config}")
    model_cls = model_loader.load_model(config.model_dir)
    docstring_model = model_cls(device=config.device)

    try:
        yield
    finally:
        docstring_model = None


app = FastAPI(lifespan=load_model)


@app.post("/annotate_code/")
async def annotate_code(request: AnnotateRequest):
    global docstring_model
    return annotate_code(request.code, docstring_model, request.overwrite_docstrings)
