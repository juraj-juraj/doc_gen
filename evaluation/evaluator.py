import argparse
import json
import logging
import pathlib
from code_anotation import annotate_corpus_http, annotate_corpus_model
from corpus_processor import load_corpus, save_annotations
from metrics import evaluator_builder


def parse_configuration(config_path: str) -> dict:
    """
    Parses the configuration file located at the specified path and returns a dictionary containing the parsed data.

    Args:
        config_path (str): The path to the configuration file.

    Returns:
        dict: A dictionary containing the parsed data from the configuration file.
    """
    raw_file = pathlib.Path(config_path).read_text(encoding="utf-8")
    return json.loads(raw_file)


def main():
    """
    Script to quantitatively evaluate docstring generator.

    Args:
        main(): The main function to execute.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(
        description="Script to quantitatively evaluate docstring generator"
    )
    annotator_group = parser.add_mutually_exclusive_group(required=True)
    annotator_group.add_argument(
        "-H", "--host", type=str, help="Host serving api for docstring generation"
    )
    annotator_group.add_argument(
        "-m",
        "--model",
        type=pathlib.Path,
        help="Path to model for docstring generation, if provided, host will be ignored",
    )
    parser.add_argument(
        "-d",
        "--corpus",
        required=True,
        type=pathlib.Path,
        help="Reference docstring generating problems",
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        type=parse_configuration,
        help="Configuration of evaluator",
    )
    parser.add_argument(
        "-w", "--workers", default=1, type=int, help="Number of workers"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=argparse.FileType(mode="w", encoding="utf-8"),
        default="-",
        help="Report output file",
    )
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="error",
        choices=["info", "debug", "warning", "error", "critical"],
        help="log level",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.getLevelName(args.log_level.upper()))
    evaluator = evaluator_builder(args.config)
    corpus = load_corpus(corpus_path=args.corpus)
    if args.host:
        logging.info(f"Annotating corpus using http at {args.host}")
        corpus = annotate_corpus_http(
            host=args.host, data=corpus, n_workers=args.workers
        )
    elif args.model:
        logging.info(f"Annotating corpus using model at {args.model}")
        corpus = annotate_corpus_model(model=args.model, data=corpus)
    score = evaluator.evaluate(
        preds=corpus["predictions"],
        refs=corpus["docstrings"],
        samples=corpus["functions"],
    )
    score.create_md_report(evaluation_config=args.config, output=args.output)
    save_annotations(data=corpus, output=args.corpus.with_suffix(".annotations.py"))


if __name__ == "__main__":
    main()
