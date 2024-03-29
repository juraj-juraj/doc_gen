# Take gpt generated pkl dataset, approx 66k samples
# Take human generated dataset approx 30k samples
# create splits:
#   Train: 66k gpt + 15k human
#   Validation: 7k human
#   Test: 8k human

import pathlib

import datasets
import pandas as pd

GENERATED_PKL = pathlib.Path("../data/gpt_generated_google_style.pkl")
HUMAN_MADE_DATASET = pathlib.Path("../data/googlestyle_dataset_processed_2.ds")
NEW_DATASET_NAME = pathlib.Path("../data/google_style_gpt_human_mix_2.ds")


def main():
    generated_df: pd.DataFrame = pd.read_pickle(GENERATED_PKL)
    human_ds = datasets.load_from_disk(HUMAN_MADE_DATASET)

    human_df: pd.DataFrame = pd.concat(
        [human_ds["train"].to_pandas(), human_ds["test"].to_pandas(), human_ds["validation"].to_pandas()]
    )
    human_df = human_df.reset_index()

    train_split_df = (
        pd.concat([generated_df, human_df.iloc[0:15000]]).sample(frac=1).reset_index()[["function", "docstring"]]
    )
    validation_split_df = human_df.iloc[15000:22000][["function", "docstring"]]
    test_split_df = human_df.iloc[22000:][["function", "docstring"]]

    ds_test = datasets.Dataset.from_pandas(test_split_df)
    ds_validation = datasets.Dataset.from_pandas(validation_split_df)
    ds_train = datasets.Dataset.from_pandas(train_split_df)
    dataset_dict = datasets.DatasetDict(
        {
            "train": ds_train,
            "validation": ds_validation,
            "test": ds_test,
        }
    )

    dataset_dict.save_to_disk(NEW_DATASET_NAME.name)


if __name__ == "__main__":
    main()
