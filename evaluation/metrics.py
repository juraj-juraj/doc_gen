from __future__ import annotations

import functools
import json
import logging
import operator
import sys
import time
from typing import Literal, Protocol, TextIO

import evaluate
import numpy as np
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer, util
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
            "```json\n"
            f"{json.dumps(self.metric_results, indent=1)}\n"
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
            score.add_result(metric.get_name(), metric.evaluate(preds, refs, samples))
        return score


class LengthEvaluator(BaseModel):
    """Evaluate lengths of given predictions
    Prediction should not be longer than upper treshold * longest reference and shorter than lower treshold * shortest reference

    Args:
        name (str) : Then name of evaluator. Defaults to name of class
        n_workers (int) : Number of concurrent workers evaluating predictions. Not implemented yet.
        length_penalty (float) : Length penalty which is given if the length or prediction id out of bounds
        low_length_treshold (float) : Multiplier for lower length treshold of shortest reference for that prediction
        upper_length_treshold (float) : Multiplier for upper length treshold of longest reference for that prediction
    """

    name: str = Field(default="length_evaluator")
    length_penalty: float = Field(ge=0, le=1)
    low_length_treshold: float = Field(ge=0, le=1)
    upper_length_treshold: float = Field(ge=1)
    n_workers: int | None

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


class SacreBleuEvaluator(BaseModel):
    """Evaluate predictions and references using bleu scoring
    More info here: https://huggingface.co/spaces/evaluate-metric/sacrebleu

    Args:
        name (str) : Name of evaluator, default to name of class.
        n_workers: Number of concurrent worker evaluating set.
        smooth_method (str): The smoothing method to use, defaults to 'exp'. Possible values are:
            'none': no smoothing
            'floor': increment zero counts
            'add-k': increment num/denom by k for n>1
            'exp': exponential decay
        smooth_value (float): The smoothing value. Only valid when smooth_method='floor' (in which case smooth_value defaults to 0.1) or smooth_method='add-k' (in which case smooth_value defaults to 1).
        tokenize (str): Tokenization method to use for BLEU. If not provided, defaults to 'zh' for Chinese, 'ja-mecab' for Japanese and '13a' (mteval) otherwise. Possible values are:
            'none': No tokenization.
            'zh': Chinese tokenization.
            '13a': mimics the mteval-v13a script from Moses.
            'intl': International tokenization, mimics the mteval-v14 script from Moses
            'char': Language-agnostic character-level tokenization.
            'ja-mecab': Japanese tokenization. Uses the MeCab tokenizer.
        lowercase (bool): If True, lowercases the input, enabling case-insensitivity. Defaults to False.
        force (bool): If True, insists that your tokenized input is actually detokenized. Defaults to False.
        use_effective_order (bool): If True, stops including n-gram orders for which precision is 0. This should be True, if sentence-level BLEU will be computed. Defaults to False.

    """

    name: str = Field(default="sacrebleu_evaluator")
    n_workers: int | None = Field(default=1, frozen=True)
    smooth_method: Literal["none", "floor", "add-k", "exp"] | None = Field(default="none")
    smooth_value: float | None = Field(default=1)
    tokenize: Literal["none", "zh", "13a", "intl", "char", "ja-mecab"] | None = Field(default="none")
    lowercase: bool | None = Field(default=False)
    force: bool | None = Field(default=False)
    use_effective_order: bool | None = Field(default=False)

    def evaluate(self, preds: list[str], refs: list[list[str]], samples: list[str] | None) -> dict:
        evaluator = evaluate.load("sacrebleu", num_process=self.n_workers)
        result = evaluator.compute(
            predictions=preds,
            references=refs,
            smooth_method=self.smooth_method,
            smooth_value=self.smooth_value,
            tokenize=self.tokenize,
            lowercase=self.lowercase,
            force=self.force,
            use_effective_order=self.use_effective_order,
        )
        result["score"] = result["score"] / 100
        return result

    def get_name(self) -> str:
        return self.name

    def get_configuration(self) -> dict:
        return self.__dict__.items()


class RougeEvaluator(BaseModel):
    """Rouge metric evaluator
    More info here: https://huggingface.co/spaces/evaluate-metric/rouge

    Args:
        name (str) : Name of evaluator, default to name of class.
        n_workers: Number of concurrent worker evaluating set.
        use_stemmer (boolean): If True, uses Porter stemmer to strip word suffixes. Defaults to False.
    """

    name: str = Field(default="rouge_evaluator")
    n_workers: int | None = Field(default=1, frozen=True)
    use_stemmer: bool | None = Field(default=False)

    def evaluate(self, preds: list[str], refs: list[list[str]], samples: list[str] | None) -> dict:
        evaluator = evaluate.load("rouge", num_process=self.n_workers)
        result = evaluator.compute(predictions=preds, references=refs, use_stemmer=self.use_stemmer)
        result["score"] = np.mean([result["rouge1"], result["rouge2"], result["rougeL"]])
        return result

    def get_name(self) -> str:
        return self.name

    def get_configuration(self) -> dict:
        return self.__dict__.items()


class MeteorEvaluator(BaseModel):
    """Meteor metric evaluator
    More info here: https://huggingface.co/spaces/evaluate-metric/meteor

    Args:
        name (str) : Name of evaluator, default to name of class.
        n_workers (int) : Number of concurrent worker evaluating set.
        alpha (float) : Parameter for controlling relative weights of precision and recall. The default value is 0.9.
        beta (float) : Parameter for controlling shape of penalty as a function of fragmentation. The default value is 3.
        gamma (float) : The relative weight assigned to fragmentation penalty. The default is 0.5.
    """

    name: str = Field(default="meteor_evaluator")
    n_workers: int | None = Field(default=1, frozen=True)
    alpha: float = Field(default=0.9)
    beta: float = Field(default=3)
    gamma: float = Field(default=0.5)

    def evaluate(self, preds: list[str], refs: list[list[str]], samples: list[str] | None) -> dict:
        evaluator = evaluate.load("meteor", num_process=self.n_workers)
        result = evaluator.compute(predictions=preds, references=refs)
        result["score"] = result["meteor"]
        return result

    def get_name(self) -> str:
        return self.name

    def get_configuration(self) -> dict:
        return self.__dict__.items()


class EmbeddingSimilarityEvaluator(BaseModel):
    """_summary_

    Args:
        name (str): Name of evaluator, default to name of class.
        n_workers (int): Number of concurrent worker evaluating set.
        model_name (str): Name of sentence transformer model
        device (str): Device to use for sentence transformer model
    """

    name: str = Field(default="embedding_similarity_evaluator")
    n_workers: int | None = Field(default=1, frozen=True)
    model: str
    device: Literal["cpu", "cuda"] = Field(default="cpu")

    def evaluate(self, preds: list[str], refs: list[list[str]], samples: list[str] | None) -> dict:
        model = SentenceTransformer(self.model, device=self.device)
        preds_embeddings = model.encode(preds, device=self.device)
        refs_embeddings = model.encode(sum(refs, []), device=self.device)

        # reshaping and repeat to match shape of references
        preds_embeddings = np.repeat(preds_embeddings[:, np.newaxis], len(refs[0]), axis=1)
        preds_embeddings = preds_embeddings.reshape(-1, preds_embeddings.shape[-1])

        cos_similarities = util.pairwise_cos_sim(preds_embeddings, refs_embeddings)
        score = {
            "score": float(cos_similarities.mean()),
            "average": float(cos_similarities.mean()),
            "variance": float(cos_similarities.var()),
        }
        return score

    def get_name(self) -> str:
        return self.name

    def get_configuration(self) -> dict:
        return self.__dict__.items()


_METRICS_REGISTRY: dict[str, type[MetricEvaluatorI]] = {
    "LengthEvaluator": LengthEvaluator,
    "SacreBleuEvaluator": SacreBleuEvaluator,
    "RougeEvaluator": RougeEvaluator,
    "MeteorEvaluator": MeteorEvaluator,
    "EmbeddingSimilarityEvaluator": EmbeddingSimilarityEvaluator,
}


def evaluator_builder(configuration: dict, n_workers: int = 1) -> Evaluator:
    logging.debug(f"Building score builder: {configuration['score_settings']}")
    score_builder = EvaluateScoreBuilder(configuration=configuration["score_settings"])
    metrics = []
    for evaluator_entry in configuration["evaluators"]:
        logging.info(f"Building evaluator: {evaluator_entry['class']}")
        eval_cls = _METRICS_REGISTRY[evaluator_entry["class"]]
        metrics.append(eval_cls(n_workers=n_workers, **evaluator_entry["params"]))
    return Evaluator(metrics=metrics, evaluate_score_builder=score_builder)
