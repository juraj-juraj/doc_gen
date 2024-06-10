import argparse
import ast
import asyncio
import json
import pathlib
import time
from dataclasses import dataclass

import backoff
import httpx


@dataclass
class FunctionsDataContainer:
    word_count: int
    loc_count: int
    data: list[str]


class TooManyRequestsException(Exception): ...


def parse_arguments():
    parser = argparse.ArgumentParser(description="Measure performance of docgen_server")
    parser.add_argument("-H", "--host", type=str, help="Address of host server")
    parser.add_argument("-d", "--data", type=pathlib.Path, help="Python functions to be annotated")

    return parser.parse_args()


def load_data(path: pathlib.Path) -> FunctionsDataContainer:
    raw_text = path.read_text(encoding="utf-8")
    word_count = len(raw_text.split())
    loc_count = len(raw_text.split("\n"))

    tree = ast.parse(raw_text)
    fces = [ast.unparse(fce) for fce in tree.body if isinstance(fce, ast.FunctionDef)]

    return FunctionsDataContainer(word_count=word_count, loc_count=loc_count, data=fces)


def add_http_protocol(fce):
    def inner_fce(host: str):
        if not host.startswith("http"):
            host = "http://" + host
        return fce(host)

    return inner_fce


@add_http_protocol
def generate_docstring_url(host: str):
    host = host.rstrip("/")
    return host + "/generate_docstring/"


@add_http_protocol
def generate_info_url(host: str):
    host = host.rstrip("/")
    return host + "/info/"


@backoff.on_exception(backoff.expo, exception=TooManyRequestsException, max_tries=3)
async def annotate_consumer(queue: asyncio.Queue, stop_event: asyncio.Event, url: str):
    async with httpx.AsyncClient() as client:
        while True:
            if queue.empty() and stop_event.is_set():
                break
            item = await queue.get()
            response = await client.post(
                url, json={"code": item, "overwrite_docstrings": True}, timeout=httpx.Timeout(timeout=300)
            )
            if response.status_code == 429:
                raise TooManyRequestsException()
            assert response.status_code == 200


async def data_producer(queue: asyncio.Queue, stop_event: asyncio.Event, fces: FunctionsDataContainer):
    for fce in fces.data:
        await queue.put(fce)
    stop_event.set()


async def run_annotation(fces: FunctionsDataContainer, url: str, nodes: int = 1):
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    stop_event = asyncio.Event()
    producer_task = asyncio.create_task(data_producer(queue, stop_event, fces))

    consumers = [asyncio.create_task(annotate_consumer(queue, stop_event, url)) for _ in range(nodes)]

    await asyncio.wait([producer_task, *consumers])


def get_server_info(host: str):
    info_url = generate_info_url(host)
    response = httpx.get(info_url)
    assert response.status_code == 200
    return json.loads(response.text)


def print_stats(total_time: float, fces: FunctionsDataContainer, server_configuration: dict):
    print(f"Total time: {total_time}")
    print(f"Total loc: {fces.loc_count}")
    print(f"Total words: {fces.word_count}")
    print(f"Total functions: {len(fces.data)}")
    print(f"Lines per second: {fces.loc_count / total_time}")
    print(f"Words per second: {fces.word_count / total_time}")
    print(f"Average latency per function: {total_time / len(fces.data)}")
    print("Server configuration")
    [print(f"    {key}: {value}") for key, value in server_configuration.items()]


def main():
    args = parse_arguments()
    functions = load_data(args.data)
    docstring_url = generate_docstring_url(args.host)

    server_info = get_server_info(args.host)

    start_time = time.time()
    asyncio.run(run_annotation(functions, docstring_url, server_info["workers"]))
    total_time = time.time() - start_time
    print_stats(total_time, functions, server_info)


if __name__ == "__main__":
    main()
