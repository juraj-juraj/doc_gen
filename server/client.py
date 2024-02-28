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
    response = httpx.post(
        f"{args.host}/annotate_code/",
        json={"code": input_code, "overwrite_docstrings": False},
        timeout=httpx.Timeout(timeout=300),
    )
    if response.status_code != 200:
        logging.error(f"Server returned error code {response.status_code} with message: {response.text}")
        return

    result = json.loads(response.text)["result"]
    args.output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
