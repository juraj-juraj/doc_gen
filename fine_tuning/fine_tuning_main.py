import argparse
import json
import logging
import os
import pathlib
import sys

import datasets
import transformers
from model_args import DataTrainingArguments, ModelArguments
from trainer_container import (
    TrainerContainer,
    TrainerContainerException,
    embedding_metric,
    sacrebleu_metrics,
)
from trainer_stat_collector import StatCollectorException, TrainerStatCollector
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)


def setup_logging(log_level: int, report_dir: pathlib.Path, should_log: bool) -> None:
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
        level=log_level,
    )
    logging.getLogger().addHandler(logging.FileHandler(filename=report_dir / "training.log"))

    if should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    # logging.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    transformers.utils.logging.get_logger().addHandler(logging.FileHandler(filename=report_dir / "training.log"))


def define_model_card(model_args: ModelArguments, data_args: DataTrainingArguments) -> dict[str, str]:
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation", "language": ["en"]}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        kwargs["dataset"] = data_args.dataset_name

    return kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("configuration", type=str, help="Json configuration for training")
    args = parser.parse_args()

    hf_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    model_args, data_args, training_args = hf_parser.parse_json_file(json_file=os.path.abspath(args.configuration))

    json_configuration = pathlib.Path(args.configuration).read_text(encoding="utf-8")
    json_configuration = json.loads(json_configuration)
    stat_collector = TrainerStatCollector(
        train_paramers=json_configuration, experiment_description="Using tokenizer from codet5p small"
    )
    log_level = training_args.get_process_log_level()
    report_dir = pathlib.Path(stat_collector.init_report_directory())

    setup_logging(log_level, report_dir, training_args.should_log)

    logging.info(f"Training/evaluation parameters {training_args}")

    if not any([training_args.do_train, training_args.do_eval, training_args.do_predict]):
        logging.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    transformers.set_seed(training_args.seed)

    # Downloading and loading a dataset from the hub.
    raw_datasets = datasets.load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
    )

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    tokenizer.pad_token_id = (
        -100 if data_args.pad_to_max_length == True and data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    trainer_container = TrainerContainer(model, tokenizer, training_args, data_args, stat_collector)
    trainer_container.prepare_trainer(raw_datasets, )

    if training_args.do_train:
        trainer_container.do_train()
    if training_args.do_eval:
        trainer_container.do_eval()
    if training_args.do_predict:
        trainer_container.do_predict(raw_datasets)

    stat_collector.create_summary()
    model_card = define_model_card(model_args, data_args)
    trainer_container.get_trainer().create_model_card(**model_card)

    if training_args.push_to_hub:
        trainer_container.get_trainer().push_to_hub(**model_card)


if __name__ == "__main__":
    try:
        main()
    except TrainerContainerException as e:
        logging.error(f"TrainerContainer threw: {e}")
    except StatCollectorException as e:
        logging.error(f"TrainerStatCollector threw: {e}")
    except Exception as e:
        logging.error(f"Unexpected exception: {e}")
