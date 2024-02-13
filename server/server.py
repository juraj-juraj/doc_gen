import logging
import pathlib
import queue
import threading
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Literal

import docstring_transformer
import model_loader
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

CONFIG_FILE = "server_config.json"
MAX_QUEUE_LEN = 3

docstring_model = None
task_queue = queue.Queue(maxsize=MAX_QUEUE_LEN)
task_buffer = {}
task_buffer_lock = threading.Lock()


class AppConfig(BaseModel):
    model_dir: pathlib.Path
    device: Literal["cpu", "cuda"] = "cpu"
    log_level: Literal["info", "debug", "warning", "error", "critical"] = "warning"


class AnnotateRequest(BaseModel):
    code: str
    overwrite_docstrings: bool = False


class TaskProgress(Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class TaskStatus(BaseModel):
    status: TaskProgress
    result: str | None


@asynccontextmanager
async def load_model(app: FastAPI):
    global docstring_model
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
    docstring_model = model_cls(device=config.device)

    try:
        yield
    finally:
        docstring_model = None


def annotate_task(request: AnnotateRequest, task_id: str):
    global finished_tasks
    global docstring_model
    logging.debug(f"Annotating task: {task_id}")
    try:
        with task_buffer_lock:
            task_buffer[task_id].status = TaskProgress.processing

        annotated_code = docstring_transformer.annotate_code(
            request.code, docstring_model, request.overwrite_docstrings
        )

        with task_buffer_lock:
            task_buffer[task_id] = TaskStatus(status=TaskProgress.completed, result=annotated_code)

    except Exception as e:
        logging.error(f"Failed to annotate task: {task_id}")
        logging.error(e)
        with task_buffer_lock:
            task_buffer[task_id].status = TaskProgress.failed


def worker(queue: queue.Queue):
    while True:
        task, task_id = queue.get()
        logging.debug(f"Processing task: {task_id}")
        annotate_task(task, task_id)
        logging.debug(f"Finished task: {task_id}")
        task_queue.task_done()


app = FastAPI(lifespan=load_model)
worker_thread = threading.Thread(target=worker, args=(task_queue,))
worker_thread.daemon = True  # Set the thread as a daemon so it exits when the main program exits
worker_thread.start()


@app.post("/annotate_task/")
async def create_annotate_task(request: AnnotateRequest):
    logging.debug("Got request to annotate code")
    global task_queue
    if task_queue.full():
        return HTTPException(status_code=429, detail="Resource exhausted, try again later.")
    task_id = uuid.uuid4()

    with task_buffer_lock:
        task_buffer[task_id] = TaskStatus(status=TaskProgress.pending, result="")
    task_queue.put((request, task_id))

    return {"task_id": task_id}


@app.get("/task_status/{task_id}")
async def get_task_status(task_id: uuid.UUID):
    logging.debug(f"Checking task task status: {task_id}")
    with task_buffer_lock:
        if task_id in task_buffer:
            response = {"status": task_buffer[task_id].status}
            return response
    return HTTPException(status_code=404, detail="Task not found")


@app.get("/task_result/{task_id}")
async def get_task_result(task_id: uuid.UUID):
    logging.debug(f"Checking task task result: {task_id}")
    with task_buffer_lock:
        if task_id in task_buffer:
            task = task_buffer[task_id]
            if task.status == TaskProgress.completed:
                del task_buffer[task_id]
                return task
            return HTTPException(status_code=412, detail=f"Task is: {task.status.value}")
    return HTTPException(status_code=404, detail="Task not found")
