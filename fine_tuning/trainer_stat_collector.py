import json
import logging
import os
import pathlib
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import git
import matplotlib.pyplot as plt
import pandas as pd
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments


def get_commit_hash() -> str:
    repo = git.Repo(search_parent_directories=True)
    return repo.head.object.hexsha


def dict_2_md_json(input: dict) -> str:
    return f"```json\n{json.dumps(input, indent=2)}\n```\n"


def format_example_prediction(function, docstring):
    return f"### Prediction\n```python\n {function}\n```\nDocstring:\n```python\n {docstring}\n```\n"


class StatCollectorException(Exception):
    """Custom exception which is raised in case of error in trainer stat collector"""

    ...


class StatCollectorI(Protocol):
    def init_report_directory(self): ...

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs): ...

    def get_text_field(self, key) -> str: ...

    def add_text_field(self, heading: str, body: str): ...

    def create_summary(self): ...


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
    experiment_description: str = field(default=None)

    def init_report_directory(self) -> str:
        self.report_dir = time.strftime("report-%y_%m_%d-%H_%M_%S", self.start_time)
        if os.path.exists(self.report_dir):
            raise StatCollectorException(f"Directory for reports already exists: {self.report_dir}")
        os.mkdir(self.report_dir)
        return self.report_dir

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        print("---------------- evaluate")
        print(f"kwargs keys: {kwargs.keys()}")
        print(f"metrics keys: {kwargs['metrics'].keys()}")
        self.add_text_field("Evaluate metrics", kwargs["metrics"])

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        logs = kwargs["logs"]
        if "loss" in logs and "epoch" in logs and "learning_rate" in logs:
            self.train_metrics = pd.concat(
                [
                    self.train_metrics,
                    pd.DataFrame(
                        {"loss": logs["loss"], "learning_rate": logs["learning_rate"], "epoch": logs["epoch"]},
                        index=[len(self.train_metrics)],
                    ),
                ],
                ignore_index=True,
            )

    def get_text_field(self, key: str) -> str:
        return self.text_fields.get(key, None)

    def add_text_field(self, heading: str, body: str):
        self.text_fields[heading] = body

    def _create_loss_graph(self):
        logging.info("Creating loss graph")
        logging.info(f"Values: {self.train_metrics['loss']}")
        if self.train_metrics.empty:
            logging.warning("No loss values to create graph")
        if self.report_dir == None:
            raise StatCollectorException("To create graph setup report directory beforehand")

        fig, ax = plt.subplots()
        ax.plot(self.train_metrics["epoch"], self.train_metrics["loss"])
        ax.set(xlabel="Epoch", ylabel="Loss")
        ax.grid()
        fig.savefig(f"{self.report_dir}/loss_progress.png")

    def create_summary(self):
        if self.report_dir == None:
            self.init_report_directory()
        self._create_loss_graph()
        report = (
            f"# {time.strftime('%x %X', self.start_time)} Report {self.train_paramers['model_name_or_path']}\n"
            f"Git commit: {self.commit_hash}\n"
            f"{self.experiment_description}\n"
            "## Train parameters \n"
            "```json\n"
            f"{json.dumps(self.train_paramers, indent=2)}"
            "\n```\n"
            "## Training loss progress\n"
            '![Training loss progress](loss_progress.png "Training loss progress")\n'
        )
        for heading, body in self.text_fields.items():
            paragraph = f"## {heading} \n {body}\n"
            report += paragraph
        report_file = pathlib.Path(self.report_dir) / "report.md"
        report_file.write_text(report)
