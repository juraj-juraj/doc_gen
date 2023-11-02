import json
import os
import pathlib
import time
from collections import Counter
from dataclasses import dataclass, field

import git
import matplotlib as plt
import pandas as pd
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


def get_commit_hash() -> str:
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


class StatCollectorException(Exception):
    """Custom exception which is raised in case of error in trainer stat collector"""

    ...


@dataclass
class TrainerStatCollector(TrainerCallback):
    train_paramers: dict
    start_time: time.struct_time = field(default_factory=time.localtime)
    commit_hash: str = field(default_factory=get_commit_hash)
    text_fields: dict[str, str] = field(default_factory=dict)
    train_metrics: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["loss", "learning_rate", "epoch"], dtype=float)
    )
    report_dir: str = field(default=None)

    def init_report_directory(self) -> str:
        self.report_dir = time.strftime("report-%x:%X", self.start_time)
        if os.path.exists(self.report_dir):
            raise StatCollectorException(f"Directory for reports already exists: {self.report_dir}")
        os.mkdir(self.report_dir)
        return self.report_dir

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """
        Event called after an evaluation phase.
        """
        print("---------------- evaluate")
        print(f"kwargs keys: {kwargs.keys()}")
        print(f"metrics keys: {kwargs['metrics'].keys()}")
        self.add_text_field("Evaluate metrics", kwargs["metrics"])

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

    def get_text_field(self, key) -> str:
        return self.text_fields.get(key, None)

    def add_text_field(self, heading, body):
        self.text_fields[heading] = body

    def _create_loss_graph(self):
        if self.report_dir == None:
            raise StatCollectorException("To create grap setup report directory beforehands")

        fig, ax = plt.subplots()
        ax.plot(self.train_metrics["epoch"], self.train_metrics["loss"])
        ax.set(xlabel="Epoch", ylabel="Loss")
        ax.grid()
        fig.savefig(f"{self.report_dir}/loss_progress.png")

    def create_summary(self):
        if self.report_dir == None:
            self.init_report_directory()
        report = (
            f"# {time.strftime('%x %X', self.start_time)} Report {self.train_paramers['model_name_or_path']}\n"
            f"Git commit: {self.commit_hash}\n"
            "## Train parameters \n"
            "```json\n"
            f"{json.dumps(self.train_paramers, indent=2)}"
            "```\n"
            "## Training loss progress\n"
            '![Training loss progress](loss_progress.png "Training loss progress)\n'
        )
        for heading, body in self.text_fields.items():
            paragraph = f"## {heading} \n {body}\n"
            report += paragraph
        report_file = pathlib.Path(self.report_dir) / "report.md"
        report_file.write_text(report)
