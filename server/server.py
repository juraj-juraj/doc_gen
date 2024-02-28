import logging
import pathlib
import queue
import threading
import typing
from contextlib import asynccontextmanager
from typing import Any, Literal

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


class ValueEvent(threading.Event):
    def __init__(self) -> None:
        super().__init__()
        self._value = None

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Any):
        self._value = value


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
            task, event_on_done = self.task_queue.get()
            logging.info(f"Worker: {self.name} annotating task")
            try:
                for _ in range(20):
                    completed_task = docstring_transformer.annotate_code(task, docstring_generator=self.model)
                event_on_done.value = completed_task
            except Exception as e:
                logging.error(f"Worker {self.name} raised exception: {e}")
            finally:
                event_on_done.set()


class FutureResult:
    def __init__(self, on_done_event: ValueEvent):
        self._on_done_event = on_done_event

    def result(self) -> typing.Any | None:
        logging.debug("Waiting on task...")
        self._on_done_event.wait()
        return self._on_done_event.value


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

    def submit_task(self, task: any) -> FutureResult | None:
        logging.info("Creating task to queue")
        v_event = ValueEvent()
        try:
            self.task_queue.put_nowait((task, v_event))
            logging.info(f"Actual queue size: {self.task_queue.qsize}")
            return FutureResult(v_event)
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
    logging.info("Received request")
    try:
        task = worker_pool.submit_task(request.code)
    except Exception as exception:
        HTTPException(status_code=429, detail=exception)

    result = task.result()
    if result is None:
        HTTPException(status_code=429, detail="Task could not be processed successfully")
    return {"result": result}


@app.get("/test_endpoint/")
async def test_endpoint() -> dict[str, str]:
    return {"status": "ok"}
