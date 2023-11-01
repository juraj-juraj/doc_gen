import time
from collections import Counter
from dataclasses import dataclass, field

import git
import pandas as pd
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


def get_commit_hash() -> str:
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


@dataclass
class TrainerStatCollector(TrainerCallback):
    train_paramers: dict
    start_time: time.struct_time = field(default_factory=time.localtime)
    commit_hash: str = field(default_factory=get_commit_hash)
    text_fields: dict[str, str] = field(default_factory=dict)
    train_metrics: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["loss", "learning_rate", "epoch"], dtype=float)
    )

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        print("---------------- evaluate")
        print(f"kwargs keys: {kwargs.keys()}")
        print(f"metrics keys: {kwargs['metrics'].keys()}")

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs["logs"]
        if Counter(list(self.train_metrics.columns)) == Counter(logs.keys()):
            # got log with keys loss, learning rate, epoch. We are in middle of training
            self.train_metrics = pd.concat(
                [self.train_metrics, pd.DataFrame(logs, index=[len(self.train_metrics)])], ignore_index=True
            )

        # print("---------------- on log")
        # print(f"kwargs keys: {kwargs.keys()}")
        # print(f"logs keys: {kwargs['logs'].keys()}")

    def get_text_field(self, key):
        return self.text_fields.get(key, None)

    def add_text_field(self, heading, body):
        self.text_fields[heading] = body

    def create_summary(self):
        ...
