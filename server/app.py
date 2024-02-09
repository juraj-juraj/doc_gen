import argparse
import ast
import pathlib
from typing import Dict

from docstring_generators import constant_docstring_generator
from docstring_transformer import annotate_code


def parse_arguments() -> Dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=pathlib.Path, help="Path to the file to write the modified code to")
    parser.add_argument("--overwrite_docstrings", action="store_true", help="Overwrite existing docstrings")
    parser.add_argument("filename", type=pathlib.Path, help="Path to the file to add docstrings to")
    return parser.parse_args()


def main():
    args = parse_arguments()

    raw_code = args.filename.read_text(encoding="utf-8")

    generated_code = annotate_code(raw_code, constant_docstring_generator, args.overwrite_docstrings)
    if args.output:
        args.output.write_text(generated_code, encoding="utf-8")
    else:
        print(generated_code)


if __name__ == "__main__":
    main()
