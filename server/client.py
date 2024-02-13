import argparse
import json
import logging
import pathlib
import time

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-H", "--host", type=str, help="The URL to make a request to")
    parser.add_argument("filename", type=pathlib.Path, help="Code to be annotated")
    parser.add_argument("-o", "--output", type=pathlib.Path, default=pathlib.Path("/dev/stdout"), help="Output file")
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="warning",
        choices=["info", "debug", "warning", "error", "critical"],
        help="log level",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.getLevelName(args.log_level.upper()),
        format="%(asctime)s.%(msecs)03d %(levelname)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    input_code = args.filename.read_text(encoding="utf-8")

    logging.info(f"Sending code to {args.host} for annotation")
    response = httpx.post(f"{args.host}/annotate_task/", json={"code": input_code, "overwrite_docstrings": False})
    task_id = json.loads(response.text)["task_id"]
    logging.info(f"Received task id: {task_id}")

    logging.info("Waiting for task to finish")
    while True:
        response = json.loads(httpx.get(f"{args.host}/task_status/{task_id}").text)
        if response["status"] == "completed":
            break
        time.sleep(1)

    response = json.loads(httpx.get(f"{args.host}/task_result/{task_id}").text)

    args.output.write_text(response["result"], encoding="utf-8")


if __name__ == "__main__":
    main()
