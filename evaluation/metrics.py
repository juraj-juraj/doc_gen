from __future__ import annotations

import functools
import json
import logging
import operator
import sys
import time
from typing import Literal, Protocol, TextIO

import numpy as np
from pydantic import BaseModel, Field
from typing_extensions import runtime_checkable


@runtime_checkable
class MetricEvaluatorI(Protocol):
    def evaluate(self, preds: list[str], refs: list[list[str]], samples: list[str] | None) -> dict: ...

    def get_name(self) -> str: ...

    def get_configuration(self) -> dict: ...


class EvaluateScoreException(Exception): ...


class EvaluateScore(BaseModel):
    """Evaluate scores encapsulation
    Add scores from different metrics, compress them and create report

    Args:
        compress_scores

    Raises:
        EvaluateScoreException: raises exception if same metric result is added twice
    """

    metric_results: dict = Field(default_factory=dict)
    compress_scores: Literal["average", "variance", "product", "sum"] = Field(default="average")

    def add_result(self, metric_name: str, stats: dict):
        if metric_name in self.metric_results:
            raise EvaluateScoreException(f"{metric_name} is already registered, this would overwrite its value")
        self.metric_results[metric_name] = stats

    def _compute_results(self):
        scores = np.array([metric["score"] for metric in self.metric_results.values()])
        product = functools.reduce(operator.mul, scores)
        return {"average": scores.mean(), "variance": scores.var(), "product": product, "sum": scores.sum()}

    def get_final_score(self) -> dict:
        return self._compute_results()[self.compress_scores]

    def create_md_report(self, evaluation_config: dict, output: TextIO | sys.stdout = sys.stdout) -> None:
        report_time = time.strftime("%x %X", time.localtime())
        score_results = self._compute_results()
        m_indent = 2
        report = (
            "# Evaluation report \n"
            f"Evaluated on: {report_time}\n"
            "## Config\n"
            "```json\n"
            f"{json.dumps(evaluation_config, indent=m_indent)}\n"
            "```\n"
            "## Overall score\n"
            "```json\n"
            f"{json.dumps(score_results, indent=m_indent)}\n"
            "```\n"
            "## Per metric score\n"
            "```json"
            f"{json.dumps(self.metric_results, indent=m_indent)}\n"
            "```\n"
        )
        output.write(report)


class EvaluateScoreBuilder(BaseModel):
    configuration: dict

    def build(self) -> EvaluateScore:
        return EvaluateScore(**self.configuration)


class Evaluator(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    metrics: list[MetricEvaluatorI]
    evaluate_score_builder: EvaluateScoreBuilder

    def evaluate(self, preds: list[str], refs: list[list[str]], samples: list[str] | None) -> EvaluateScore:
        score = self.evaluate_score_builder.build()
        for metric in self.metrics:
            score.add_result(metric.get_name, metric.evaluate(preds, refs, samples))
        return score


class LengthEvaluator(BaseModel):
    """Evaluate lengths of given predictions
    Prediction should not be longer than upper treshold * longest reference and shorter than lower treshold * shortest reference

    Args:
        length_penalty (float) : length penalty which is given if the length or prediction id out of bounds
        low_length_treshold (float) : Multiplier for lower length treshold of shortest reference for that prediction
        upper_length_treshold (float) : Multiplier for upper length treshold of longest reference for that prediction
    """

    name: str = Field(default="length_evaluator")
    length_penalty: float = Field(ge=0, le=1)
    low_length_treshold: float = Field(ge=0, le=1)
    upper_length_treshold: float = Field(ge=1)

    def evaluate(self, preds: list[str], refs: list[list[str]], samples: list[str] | None = None) -> dict:
        logging.info(f"Evaluating with evaluator: {self}")
        len_list_strs = lambda obj_list: [len(obj) for obj in obj_list]
        min_max_from_list = lambda obj_list: [min(obj_list), max(obj_list)]
        ref_lengths = [min_max_from_list(len_list_strs(ref_list)) for ref_list in refs]
        prediction_lens = [len(prediction) for prediction in preds]

        penalties = np.array(
            [
                0 if bound[0] < prediction_len < bound[1] else self.length_penalty
                for bound, prediction_len in zip(ref_lengths, prediction_lens)
            ]
        )

        return {"score": penalties.mean(), "average": penalties.mean(), "variance": penalties.var()}

    def get_name(self) -> str:
        return self.name

    def get_configuration(self) -> dict:
        self.__dict__.items()


_METRICS_REGISTRY: dict[str, type[MetricEvaluatorI]] = {"LengthEvaluator": LengthEvaluator}


def evaluator_builder(configuration: dict) -> Evaluator:
    logging.debug(f"Building score builder: {configuration['score_settings']}")
    score_builder = EvaluateScoreBuilder(configuration=configuration["score_settings"])
    metrics = []
    for evaluator_entry in configuration["evaluators"]:
        logging.info(f"Building evaluator: {evaluator_entry['class']}")
        eval_cls = _METRICS_REGISTRY[evaluator_entry["class"]]
        metrics.append(eval_cls(**evaluator_entry["params"]))
    return Evaluator(metrics=metrics, evaluate_score_builder=score_builder)
