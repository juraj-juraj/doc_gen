from __future__ import annotations

import asyncio
import enum
import logging
import pathlib
import queue
import threading
import typing
from contextlib import asynccontextmanager
from typing import Literal

import docstring_transformer
import model_loader
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

CONFIG_FILE = "server_config.json"
worker_pool: PersistentWorkerPool | None = None
CONFIGURATION: AppConfig | None = None

class AppConfig(BaseModel):
    model_dir: pathlib.Path
    device: Literal["cpu", "cuda"] = "cpu"
    workers: int = 1
    queue_size: int = 1
    log_level: Literal["info", "debug", "warning", "error", "critical"] = "warning"


class AnnotateRequest(BaseModel):
    code: str
    overwrite_docstrings: bool = False
    
class TaskType(enum.Enum):
    annotate_code = "annotate_code"
    generate_docstring = "generate_docstring"

class GeneralTask(BaseModel):
    type: TaskType
    data: AnnotateRequest

class AnnotateWorker(threading.Thread):
    def __init__(
        self,
        model_cls: typing.Type[model_loader.ModelI],
        task_queue: queue.Queue,
        device: Literal["cpu", "cuda"] = "cpu",
        name: str | None = None,
    ) -> None:
        logging.info(f"Initiating worker {name}")
        super().__init__(name=name, daemon=True)
        self.model = model_cls(device=device)
        self.task_queue = task_queue

    def run(self):
        while True:
            task, future_result = self.task_queue.get()
            logging.info(f"Worker: {self.name} annotating task")
            try:
                match task.type:
                    case TaskType.generate_docstring:
                        completed_task = docstring_transformer.generate_docstring(task.data.code, docstring_generator=self.model)
                    case TaskType.annotate_code:
                        completed_task = docstring_transformer.annotate_code(task.data.code, docstring_generator=self.model, overwrite_docstrings=task.data.overwrite_docstrings)
                future_result.set_result(completed_task)
            except Exception as e:
                logging.error(f"Worker {self.name} raised exception: {e}")
                future_result.set_exception(e)
            finally:
                logging.info(f"Worker: {self.name} done annotating task")


class WorkerPoolException(Exception): ...


class PersistentWorkerPool:
    def __init__(
        self,
        worker_initializer_cls: typing.Type[model_loader.ModelI],
        worker_args: dict[str, typing.Any] = {},
        no_workers: int = 1,
        queue_size: int = 1,
    ):
        logging.info("Creating worker pool")

        self.task_queue: queue.Queue[tuple] = queue.Queue(maxsize=queue_size)
        self.workers = [
            AnnotateWorker(worker_initializer_cls, self.task_queue, name = f"worker_{i}", **worker_args)
            for i in range(no_workers)
        ]
        [worker.start() for worker in self.workers]

    def submit_task(self, task: GeneralTask) -> asyncio.Future:
        logging.info("Creating task to queue")
        future_result: asyncio.Future = asyncio.Future()
        try:
            self.task_queue.put_nowait((task, future_result))
            logging.info(f"Actual queue size: {self.task_queue.qsize()}")
            return future_result
        except queue.Full:
            logging.warning("Resource exhausted, try again later")
            raise WorkerPoolException("Resource exhausted, try again later")



@asynccontextmanager
async def init_workers(app: FastAPI):
    global worker_pool
    global CONFIGURATION
    module_dir = pathlib.Path(__file__).parent
    json_configuration = (module_dir / CONFIG_FILE).read_text(encoding="utf-8")
    CONFIGURATION = AppConfig.model_validate_json(json_configuration)
    logging.basicConfig(
        level=logging.getLevelName(CONFIGURATION.log_level.upper()),
        format="%(asctime)s.%(msecs)03d %(levelname)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(f"Loaded configuration file: {CONFIGURATION}")
    model_cls = model_loader.load_model(CONFIGURATION.model_dir)
    worker_pool = PersistentWorkerPool(model_cls, {"device": CONFIGURATION.device}, CONFIGURATION.workers, CONFIGURATION.queue_size)
    try:
        yield
    finally:
        worker_pool = None


app = FastAPI(lifespan=init_workers)


@app.post("/annotate_code/")
async def annotate_code(request: AnnotateRequest) -> dict[str, str]:
    global worker_pool
    logging.info("Annotate code request")
    try:
        task = worker_pool.submit_task(task=GeneralTask(type=TaskType.annotate_code, data=request))

    except WorkerPoolException as exception:
        raise HTTPException(status_code=429, detail=str(exception))

    result = await asyncio.wrap_future(task)
    return {"result": result}


@app.post("/generate_docstring/")
async def generate_docstring(request: AnnotateRequest) -> dict[str, str]:
    global worker_pool
    logging.info("Generate docstring request")
    try:
        task = worker_pool.submit_task(task=GeneralTask(type=TaskType.generate_docstring, data=request))
    except WorkerPoolException as exception:
        raise HTTPException(status_code=429, detail=str(exception))

    result = await asyncio.wrap_future(task)
    return {"result": result}

@app.get("/info/")
async def info() -> AppConfig:
    global CONFIGURATION
    if not CONFIGURATION:
        raise HTTPException(status_code=404, detail="Configuration not loaded")

    return CONFIGURATION
