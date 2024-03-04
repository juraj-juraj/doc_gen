import argparse
import pathlib

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Join selected datasets in pkl format to one pkl")
    parser.add_argument("files", type=str, nargs="+", help="List of files to concatenate")
    parser.add_argument("-o", "--output", type=str, help="Output_file")
    args = parser.parse_args()

    datasets = [pd.read_pickle(file) for file in args.files]

    if len(datasets) == 0:
        print("No datasets found")

    whole_dataset = pd.concat(datasets)
    print(f"Concatenated dataset len: {len(whole_dataset)}")
    whole_dataset.to_pickle(args.output)


if __name__ == "__main__":
    main()
