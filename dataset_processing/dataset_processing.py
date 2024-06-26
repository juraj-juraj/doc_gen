import argparse
import logging
import pathlib
import re
import sys

import pandas as pd
from datasets import Dataset, DatasetDict
from grammar_filter import GrammarFilter
from tqdm import tqdm


def setup_logging(log_level: int = logging.WARNING):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )


def parse_arguments():
    """Parse arguments from command line

    Returns them as a dictionary.

    Returns:
        dict: parsed command line arguments
    """
    parser = argparse.ArgumentParser(prog="dataset_processing ")
    parser.add_argument("file", type=str, help="pickle dataset to process")
    parser.add_argument("--output_file", type=str, default=None, help="File to save processed dateset")
    parser.add_argument(
        "--log_level",
        type=logging.getLevelName,
        default="WARNING",
        help="Set logging level",
    )
    parser.add_argument("--grammar", type=pathlib.Path, default=None, help="Grammar of desired comments")
    parser.add_argument("--repository", type=str, default=None, help="Repository to push dataset to")
    return parser.parse_args()


def remove_docstring(function_string: str) -> str:
    """Function to strip python function of docstring

    Input:
        function_string: Function with docstring
    Returns:
        str: Function without docstring
    """
    return re.sub("(\"\"\"|\\'\\'\\')(.*?)(\\1)", "", function_string, flags=re.DOTALL)


def filter_lengths(dataset: pd.DataFrame, lower_bound: int = 50, high_bound: int = 500) -> pd.DataFrame:
    """
    Filters a dataset based on the length of strings within a specified range.

    Args:
        dataset (pd.DataFrame): The input dataset to be filtered.
        lower_bound (int, optional): The lower bound for the length of strings in the dataset. Defaults to 50.
        high_bound (int, optional): The upper bound for the length of strings in the dataset. Defaults to 500.

    Returns:
        pd.DataFrame: The dataset with strings longer than the specified lower and shorter than the specified upper bounds.
    """
    longer_than_lower = dataset["docstring"].str.len() > lower_bound
    shorter_than_higher = dataset["docstring"].str.len() < high_bound
    return dataset[longer_than_lower & shorter_than_higher]


def filter_by_grammar(dataset: pd.DataFrame, grammar_file: pathlib.Path) -> pd.DataFrame:
    """
    Filters a dataset based on a given grammar file.

    Args:
        dataset (pd.DataFrame): The dataset to be filtered.
        grammar_file (pathlib.Path): The path to the grammar file to be used for filtering.

    Returns:
        pd.DataFrame: The dataset filtered based on the grammar file.
    """
    grammar = grammar_file.read_text(encoding="utf-8")
    grammar_filter = GrammarFilter(grammar)
    return dataset.iloc[[grammar_filter(docstring) for docstring in dataset.docstring]]


def filter_by_grammar_neg(dataset: pd.DataFrame, grammar_file: pathlib.Path) -> pd.DataFrame:
    """
    Filters a dataset based on a given grammar file and returns a new dataset with only the docstring that do not match the grammar.

    Args:
        dataset (pd.DataFrame): The input dataset to be filtered.
        grammar_file (pathlib.Path): The path to the grammar file to be used for filtering.

    Returns:
        pd.DataFrame: A new dataset with only the docstring that do not match the grammar.
    """
    grammar = grammar_file.read_text(encoding="utf-8")
    grammar_filter = GrammarFilter(grammar)
    return dataset.iloc[[not grammar_filter(docstring) for docstring in dataset.docstring]]


def main(arguments):
    setup_logging(arguments.log_level)
    bar = tqdm(range(6))
    logging.debug("Reading dataset")
    raw_data = pd.read_pickle(arguments.file)
    bar.update(1)
    df = pd.DataFrame(raw_data)
    bar.update(1)
    df = df[["docstring", "function"]].copy()
    logging.debug("Removing docstrings")
    df["function"] = df["function"].apply(remove_docstring)
    df = df[df["docstring"].str.len() > 0]
    bar.update(1)
    df = filter_lengths(df)
    if arguments.grammar:
        logging.debug("Filtering by grammar")
        df = filter_by_grammar(df, arguments.grammar)
    logging.info(f"Entries of dataset: {len(df)}")
    ds_test = Dataset.from_pandas(df.iloc[0:1000])
    ds_validation = Dataset.from_pandas(df.iloc[1000:2000])
    ds_train = Dataset.from_pandas(df.iloc[2000:])
    bar.update(1)
    dataset_dict = DatasetDict({"train": ds_train, "validation": ds_validation, "test": ds_test})
    bar.update(1)
    if arguments.output_file:
        logging.info("Saving to disk")
        dataset_dict.save_to_disk(arguments.output_file)
    if arguments.repository:
        logging.info("Saving to hub")
        dataset_dict.push_to_hub(arguments.repository)
    bar.update(1)


def get_bad_annotated_functions(arguments):
    setup_logging(arguments.log_level)
    bar = tqdm(range(5))
    logging.debug("Reading dataset")
    raw_data = pd.read_pickle(arguments.file)
    bar.update(1)
    df = pd.DataFrame(raw_data)
    bar.update(1)
    df = df[["docstring", "function"]].copy()
    logging.debug("Removing docstrings")
    df["function"] = df["function"].apply(remove_docstring)
    df = df[df["docstring"].str.len() > 0]
    bar.update(1)
    df = filter_lengths(df, lower_bound=100)
    if arguments.grammar:
        logging.debug("Filtering by grammar negation")
        df = filter_by_grammar_neg(df, arguments.grammar)
    logging.info(f"Entries of dataset: {len(df)}")
    bar.update(1)
    df.reset_index(inplace=True)
    if arguments.output_file:
        logging.info("Saving to disk")
        df["function"].to_pickle(arguments.output_file)
    bar.update(1)


if __name__ == "__main__":
    arguments = parse_arguments()
    main(arguments)
