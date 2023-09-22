#################
# Dataset processing script
# Process data to huggingface dataset
# Remove docstring from code examples and remove those without documentation
# Dataset downloaded from https://www.kaggle.com/datasets/omduggineni/codesearchnet?resource=download
###########

import argparse
import re

import pandas as pd
from datasets import Dataset, DatasetDict
from tqdm import tqdm


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
        "--output_file",
        type=str,
        default=None,
        help="File to save processed dateset",
    )
    parser.add_argument(
        "--repository", type=str, default=None, help="Repository to push dataset to"
    )
    return parser.parse_args()


def remove_docstring(function_string: str) -> str:
    """Function to strip python function of docstring

    Input:
        function_string: Function with docstring
    Returns:
        str: Function without docstring
    """
    return re.sub(r'("""|\'\'\')(.*?)(\1)', "", function_string, flags=re.DOTALL)


def main(arguments: dict):
    bar = tqdm(range(6))
    raw_data = pd.read_pickle(arguments.file)
    bar.update(1)
    df = pd.DataFrame(raw_data)
    bar.update(1)
    df = df[["docstring", "function"]].copy()

    df["function"] = df["function"].apply(remove_docstring)
    df = df[df["docstring"].str.len() > 0]
    bar.update(1)
    ds_test = Dataset.from_pandas(df.loc[0:1000])
    ds_validation = Dataset.from_pandas(df.loc[1000:2000])
    ds_train = Dataset.from_pandas(df.loc[2000:])
    bar.update(1)
    dataset_dict = DatasetDict(
        {
            "train": ds_train,
            "validation": ds_validation,
            "test": ds_test,
        }
    )
    bar.update(1)
    if arguments.output_file:
        dataset_dict.save_to_disk(arguments.output_file)
    if arguments.repository:
        dataset_dict.push_to_hub(arguments.repository)
    bar.update(1)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
