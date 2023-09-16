import argparse
import pickle
import re

import pandas as pd


def parse_arguments() -> dict:
    """Parser arguments from command line

    Returns them as a dictionary.
    Input file is type pathlib.Path, output file is just string.

    Returns:
        dict: parsed command line arguments
    """
    parser = argparse.ArgumentParser(prog="dataset_processing ")
    parser.add_argument("-f", "--file", type=str, help="pickle dataset to process")
    parser.add_argument(
        "-o", "--output", type=str, default="stdout", help="File to save processed dateset"
    )
    return parser.parse_args()


def remove_docstring(function_string: str) -> str:
    return re.sub(r'("""|\'\'\')(.*?)(\1)', "", function_string, flags=re.DOTALL)


def dumps_series(serie: pd.Series) -> str:
    for fce in serie:
        print(f"\n-----------\n {fce}")


def main(arguments: dict):
    raw_data = pd.read_pickle(arguments.file)
    df = pd.DataFrame(raw_data)
    df = df[["docstring", "function"]].copy()

    df["function"] = df["function"].apply(remove_docstring)
    df = df[df["docstring"].str.len() > 0]

    if arguments.output != "stdout":
        with open(arguments.output, "wb") as f:
            pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print(df.to_csv(index=True))


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
