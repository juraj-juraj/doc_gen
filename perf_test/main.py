import argparse
import ast
import asyncio
import pathlib
import time

import httpx


def parse_arguments():
    parser = argparse.ArgumentParser(description="Measure performance of docgen_server")
    parser.add_argument("-H", "--host", type=str, help="Address of host server")
    parser.add_argument("-d", "--data", type=pathlib.Path, help="Python functions to be annotated")
    parser.add_argument("-n", "--nodes", type=int, default=1, help="No. of parallel annotating nodes")

    return parser.parse_args()


def load_data(path: pathlib.Path) -> list[str]:
    raw_text = path.read_text(encoding="utf-8")

    tree = ast.parse(raw_text)
    fces = [ast.unparse(fce) for fce in tree.body if isinstance(fce, ast.FunctionDef)]

    return fces


def generate_docstring_url(host: str):
    if not host.startswith("http"):
        host += "http://"

    host = host.rstrip("/")
    return host + "/generate_docstring/"


async def annotate_consumer(queue: asyncio.Queue, stop_event: asyncio.Event, url: str):
    async with httpx.AsyncClient() as client:
        while True:
            if queue.empty() and stop_event.is_set():
                break
            item = await queue.get()
            response = await client.post(url, json={"code": item, "overwrite_docstrings": True})


async def data_producer(queue: asyncio.Queue, stop_event: asyncio.Event, fces: list[str]):
    for fce in fces:
        await queue.put(fce)
    stop_event.set()


async def run_annotation(fces: list[str], url: str, nodes: int = 1):
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    stop_event = asyncio.Event()
    producer_task = asyncio.create_task(data_producer(queue, stop_event, fces))

    consumers = [asyncio.create_task(annotate_consumer(queue, stop_event, url)) for _ in range(nodes)]

    await asyncio.wait([producer_task, *consumers])


def main():
    args = parse_arguments()
    functions = load_data(args.data)
    start_time = time.time()
    docstring_url = generate_docstring_url(args.host)


if __name__ == "__main__":
    main()
