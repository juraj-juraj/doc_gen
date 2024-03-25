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
MAX_QUEUE_LEN = 3
worker_pool = None


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
    data: typing.Any

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
                        completed_task = docstring_transformer.generate_docstring(task.data, docstring_generator=self.model)
                    case TaskType.annotate_code:
                        completed_task = docstring_transformer.annotate_code(task.data, docstring_generator=self.model)
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
        worker_args: dict | None = None,
        no_workers: int | None = 1,
        queue_size: int | None = 1,
    ):
        logging.info("Creating worker pool")

        self.task_queue = queue.Queue(maxsize=queue_size)
        self.workers = [
            AnnotateWorker(worker_initializer_cls, self.task_queue, **worker_args | {"name": f"worker_{i}"})
            for i in range(no_workers)
        ]
        [worker.start() for worker in self.workers]

    def submit_task(self, task: GeneralTask) -> asyncio.Future | None:
        logging.info("Creating task to queue")
        future_result = asyncio.Future()
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
    module_dir = pathlib.Path(__file__).parent
    json_configuration = (module_dir / CONFIG_FILE).read_text(encoding="utf-8")
    config = AppConfig.model_validate_json(json_configuration)
    logging.basicConfig(
        level=logging.getLevelName(config.log_level.upper()),
        format="%(asctime)s.%(msecs)03d %(levelname)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info(f"Loaded configuration file: {config}")
    model_cls = model_loader.load_model(config.model_dir)
    worker_pool = PersistentWorkerPool(model_cls, {"device": config.device}, config.workers, config.queue_size)
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
        task = worker_pool.submit_task(task=GeneralTask(type=TaskType.annotate_code, data=request.code))

    except WorkerPoolException as exception:
        raise HTTPException(status_code=429, detail=str(exception))

    result = await asyncio.wrap_future(task)
    return {"result": result}


@app.post("/generate_docstring/")
async def generate_docstring(request: AnnotateRequest) -> dict[str, str]:
    global worker_pool
    logging.info("Generate docstring request")
    try:
        task = worker_pool.submit_task(task=GeneralTask(type=TaskType.generate_docstring, data=request.code))
    except WorkerPoolException as exception:
        raise HTTPException(status_code=429, detail=str(exception))

    result = await asyncio.wrap_future(task)
    return {"result": result}


@app.get("/test_endpoint/")
async def test_endpoint() -> dict[str, str]:
    return {"status": "ok"}
