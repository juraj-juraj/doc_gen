###
# This module encapsulates fine tuning process
# This script is inspired and contains parts of code from here https://github.com/huggingface/transformers/blob/main/examples/pytorch/translation/run_translation.py
# I (Juraj Novosad), am not sole creator of this script


import logging
import os
import pathlib
from typing import Callable, Literal

import evaluate
import numpy as np
import sentence_transformers
from accelerate import Accelerator
from datasets import Dataset, DatasetDict
from fine_tuning_utils import postprocess_text
from model_args import DataTrainingArguments
from trainer_stat_collector import (
    StatCollectorI,
    dict_2_md_json,
    format_example_prediction,
)
from transformers import (
    DataCollatorForSeq2Seq,
    EvalPrediction,
    PreTrainedTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    default_data_collator,
)
from transformers.tokenization_utils_base import BatchEncoding
from transformers.trainer_utils import get_last_checkpoint


class TrainerContainerException(ValueError):
    "Special type of exception thrown by trainer Container"

    ...


def sacrebleu_metrics(tokenizer: PreTrainedTokenizer) -> Callable:
    metric = evaluate.load("sacrebleu")

    def wrapper(eval_preds: EvalPrediction) -> dict:
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {"bleu": result["score"]}

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    return wrapper


def embedding_similarity_metric(
    tokenizer: PreTrainedTokenizer, model: str = "all-MiniLM-L6-v2", device: str = "cuda"
) -> Callable:
    metric = sentence_transformers.SentenceTransformer(model, device=device)

    def wrapper(eval_preds: EvalPrediction):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        preds_embeddings = metric.encode(decoded_preds, device=device)
        refs_embeddings = metric.encode(decoded_labels, device=device, show_progress_bar=True)
        score = sentence_transformers.util.pairwise_cos_sim(preds_embeddings, refs_embeddings).mean()
        return {"score": score}

    return wrapper


METRICS_MAP = {"sacrebleu": sacrebleu_metrics, "embedding_similarity": embedding_similarity_metric}


class TrainerContainer:
    def __init__(
        self,
        model,
        tokenizer: PreTrainedTokenizer,
        training_args: Seq2SeqTrainingArguments,
        data_args: DataTrainingArguments,
        stat_collector: StatCollectorI,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.training_args = training_args
        self.data_args = data_args
        self.stat_collector = stat_collector
        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch

        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))

        logging.info(f"Model.decoder.start token: {model.config.decoder_start_token_id}")
        if model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

        self.prefix = data_args.source_prefix or ""

        # Temporarily set max_target_length for training.
        self.max_target_length = data_args.max_target_length
        self.padding = "max_length" if data_args.pad_to_max_length else False

        if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
            logging.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f" `{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more"
                " memory"
            )

    def prepare_trainer(self, raw_datasets: DatasetDict):
        self.train_dataset = self._prepare_dataset(raw_datasets, "train")
        self.eval_dataset = self._prepare_dataset(raw_datasets, "validation")

        self.max_length = self.training_args.generation_max_length or self.data_args.val_max_target_length
        metrics_fce = METRICS_MAP[self.data_args.metric_function](self.tokenizer)

        label_pad_token_id = -100 if self.data_args.ignore_pad_token_for_loss else self.tokenizer.pad_token_id
        if self.data_args.pad_to_max_length:
            data_collator = default_data_collator
        else:
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                padding=self.padding,
                max_length=self.data_args.max_source_length,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8 if self.training_args.fp16 else None,
            )
        accelerator = Accelerator()
        self.trainer = accelerator.prepare(
            Seq2SeqTrainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset if self.training_args.do_eval else None,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=metrics_fce if self.training_args.predict_with_generate else None,
                callbacks=[self.stat_collector],
            )
        )

    def do_train(self) -> None:
        logging.info("*** Train ***")

        checkpoint = self._get_last_checkpoint()
        logging.info(f"Starting from checkpoint {checkpoint}")
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint)
        self.trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = self.data_args.max_train_samples or len(self.train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(self.train_dataset))

        self.stat_collector.add_text_field("Train metrics", dict_2_md_json(self.trainer.metrics_format(metrics)))

        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()

    def do_eval(self) -> None:
        logging.info("*** Evaluate ***")

        metrics = self.trainer.evaluate(
            max_length=self.max_length, num_beams=self.training_args.generation_num_beams, metric_key_prefix="eval"
        )
        max_eval_samples = self.data_args.max_eval_samples or len(self.eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(self.eval_dataset))

        self.stat_collector.add_text_field("Evaluate metrics", dict_2_md_json(self.trainer.metrics_format(metrics)))
        self.trainer.log_metrics("eval", metrics)
        self.trainer.save_metrics("eval", metrics)

    def do_predict(self, dataset: DatasetDict) -> None:
        logging.info("*** Predict ***")
        predict_dataset = self._prepare_dataset(dataset, "test")

        predict_results = self.trainer.predict(
            test_dataset=predict_dataset,
            metric_key_prefix="predict",
            max_length=self.max_length,
            num_beams=self.training_args.generation_num_beams,
        )
        metrics = predict_results.metrics
        max_predict_samples = self.data_args.max_predict_samples or len(predict_dataset)

        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        self.stat_collector.add_text_field("Predict Metrics", dict_2_md_json(self.trainer.metrics_format(metrics)))
        self.trainer.log_metrics("predict", metrics)
        self.trainer.save_metrics("predict", metrics)

        if self.training_args.predict_with_generate:
            predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, self.tokenizer.pad_token_id)
            predictions = self.tokenizer.batch_decode(
                predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = pathlib.Path(self.stat_collector.report_dir) / "generated_predictions.txt"
            output_prediction_file.write_text("\n##################\n".join(predictions), encoding="utf-8")
            example_predictions = "5 Functions with generated docstrings: \n"
            for index in range(5):
                example_predictions += format_example_prediction(
                    function=dataset["test"]["function"][index], docstring=predictions[index]
                )
            self.stat_collector.add_text_field("Example predictions", example_predictions)

    def get_trainer(self) -> Trainer:
        return self.trainer

    def _get_last_checkpoint(self) -> os.PathLike | None:
        # TODO rewrite it
        last_checkpoint = None
        if (
            os.path.isdir(self.training_args.output_dir)
            and self.training_args.do_train
            and not self.training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(self.training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(self.training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({self.training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and self.training_args.resume_from_checkpoint is None:
                logging.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        checkpoint = None
        if self.training_args.resume_from_checkpoint is not None:
            checkpoint = self.training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        return checkpoint

    def _prepare_dataset(self, raw_datasets: DatasetDict, split: Literal["train", "validation", "test"]) -> Dataset:
        logging.debug(f"Preparing split: {split}")
        if split not in raw_datasets:
            raise TrainerContainerException(f"{split} section is  not in raw dataset")

        max_predict_samples = {
            "train": self.data_args.max_train_samples,
            "validation": self.data_args.max_eval_samples,
            "test": self.data_args.max_predict_samples,
        }
        split_dataset = raw_datasets[split]
        if max_predict_samples[split] is not None:
            max_predict_samples = min(len(split_dataset), max_predict_samples[split])
            split_dataset = split_dataset.select(range(max_predict_samples))
        with self.training_args.main_process_first(desc="prediction dataset map pre-processing"):
            split_dataset = split_dataset.map(
                self._preprocess_function,
                batched=True,
                num_proc=self.data_args.preprocessing_num_workers,
                remove_columns=raw_datasets[split].column_names,
                load_from_cache_file=not self.data_args.overwrite_cache,
                desc=f"Running tokenizer on {split} dataset",
            )
        return split_dataset

    def _preprocess_function(self, examples: Dataset) -> BatchEncoding:
        inputs = examples["function"]
        targets = examples["docstring"]
        inputs = [
            self.prefix + inp for inp in inputs
        ]  # prefix should be something like "generate docstring to this python function: "

        model_inputs = self.tokenizer(
            inputs,
            max_length=self.data_args.max_source_length,
            padding=self.padding,
            truncation=True,
        )

        # Tokenize targets with the `text_target` keyword argument, for labels
        labels = self.tokenizer(
            text_target=targets,
            max_length=self.max_target_length,
            padding=self.padding,
            truncation=True,
        )

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
