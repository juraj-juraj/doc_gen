import argparse
import importlib as il
import logging
import os

import model
import torch


def parse_arguments() -> dict:
    parser = argparse.ArgumentParser(prog="program to tryout models")
    parser.add_argument("-m", "--model", type=str, help="model to tryout")
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="info",
        choices=["info", "debug", "warning", "error", "critical"],
        help="log level",
    )
    parser.add_argument("--device", type=str, default="cpu", help="device to use for inference")
    parser.add_argument("prompt", type=str, help="prompt to generate from")
    return parser.parse_args()


def import_model(module_name: str) -> model.Model:
    try:
        logging.info(f"Importing module '{module_name}'")
        module = il.import_module(module_name)
        return module.Model
    except ImportError:
        raise RuntimeError(f"Failed to import module '{module_name}'")


def main():
    arguments = parse_arguments()
    logging.basicConfig(level=logging.getLevelName(arguments.log_level.upper()))
    try:
        if not os.path.exists(f"{arguments.model}.py"):
            raise RuntimeError("Model does not exist")
        model_cls = import_model(arguments.model)

        # device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model_cls(arguments.device)
        print(model.generate(arguments.prompt, 100))

    except Exception as e:
        logging.error(e)
        exit(1)


if __name__ == "__main__":
    main()
