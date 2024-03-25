import argparse
import json
import logging
import pathlib

from code_anotation import annotate_corpus
from corpus_processor import load_corpus, save_annotations
from metrics import evaluator_builder


def parse_configuration(config_path: str) -> dict:
    raw_file = pathlib.Path(config_path).read_text(encoding="utf-8")
    return json.loads(raw_file)


def main():
    parser = argparse.ArgumentParser(description="Script to quantitatively evaluate docstring generator")
    parser.add_argument("-H", "--host", required=True, type=str, help="Host serving api for docstring generation")
    parser.add_argument(
        "-d", "--corpus", required=True, type=pathlib.Path, help="Reference docstring generating problems"
    )
    parser.add_argument("-c", "--config", required=True, type=parse_configuration, help="Configuration of evaluator")
    parser.add_argument("-w", "--workers", default=1, type=int, help="Number of workers")
    parser.add_argument(
        "-o", "--output", type=argparse.FileType(mode="w", encoding="utf-8"), default="-", help="Report output file"
    )
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="error",
        choices=["info", "debug", "warning", "error", "critical"],
        help="log level",
    )

    #    try:
    args = parser.parse_args()
    logging.basicConfig(level=logging.getLevelName(args.log_level.upper()))

    evaluator = evaluator_builder(args.config)
    corpus = load_corpus(corpus_path=args.corpus)
    corpus = annotate_corpus(host=args.host, data=corpus, n_workers=args.workers)
    score = evaluator.evaluate(preds=corpus["predictions"], refs=corpus["docstrings"], samples=corpus["functions"])
    score.create_md_report(evaluation_config=args.config, output=args.output)
    save_annotations(data=corpus, output=args.corpus.with_suffix(".annotations.py"))
    # except Exception as e:
    #     logging.error(f"Error occurred: {e}")


if __name__ == "__main__":
    main()
