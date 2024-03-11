import argparse
import json
import logging
import pathlib

from corpus_processor import load_corpus
from metrics import evaluator_builder


def parse_configuration(config_path: str) -> dict:
    raw_file = pathlib.Path(config_path).read_text(encoding="utf-8")
    return json.loads(raw_file)


def main():
    parser = argparse.ArgumentParser(description="Script to quantitatively evaluate docstring generator")
    parser.add_argument("-H", "--host", type=str, help="Host serving api for docstring generation")
    parser.add_argument("-d", "--corpus", type=pathlib.Path, help="Reference docstring generating problems")
    parser.add_argument("-c", "--config", type=parse_configuration, help="Configuration of evaluator")
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="error",
        choices=["info", "debug", "warning", "error", "critical"],
        help="log level",
    )

    try:
        args = parser.parse_args()
        logging.basicConfig(level=logging.getLevelName(args.log_level.upper()))

        evaluator = evaluator_builder(args.config)
        corpus = load_corpus(args.corpus)

    except Exception as e:
        logging.error(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
