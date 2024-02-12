import argparse
import json
import logging
import pathlib

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
    # response = httpx.post(
    #     f"{args.host}/annotate_code/", json={"code": input_code, "overwrite_docstrings": False}, timeout=30
    # )

    response_wait = httpx.get(f"{args.host}/long_wait/", timeout=30)
    logging.info("Received response")
    # args.output.write_text(json.loads(response.text)["code"], encoding="utf-8")


if __name__ == "__main__":
    main()
