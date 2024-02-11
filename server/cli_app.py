import argparse
import importlib as il
import logging
import pathlib
from typing import Dict

from docstring_transformer import annotate_code
from model_loader import load_model


def parse_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Path to the file to write the modified code to")
    parser.add_argument("--overwrite_docstrings", action="store_true", help="Overwrite existing docstrings")
    parser.add_argument(
        "--device", type=str, choices=["cuda", "cpu"], default="cpu", help="device to use for inference"
    )
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="info",
        choices=["info", "debug", "warning", "error", "critical"],
        help="log level",
    )
    parser.add_argument("--model", type=pathlib.Path, help="Model to use for generating docstrings")
    parser.add_argument("filename", type=pathlib.Path, help="Path to the file to add docstrings to")
    return parser.parse_args()


def main():
    args = parse_arguments()
    logging.basicConfig(level=logging.getLevelName(args.log_level.upper()))

    model_cls = load_model(args.model)
    model = model_cls(device=args.device)
    raw_code = args.filename.read_text(encoding="utf-8")

    generated_code = annotate_code(raw_code, model, args.overwrite_docstrings)
    if args.output:
        args.output.write_text(generated_code, encoding="utf-8")
    else:
        print(generated_code)


if __name__ == "__main__":
    main()
