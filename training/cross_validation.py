#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import warnings
from dataclasses import dataclass, field
from typing import List, Optional
import json
import datasets
import evaluate
import numpy as np
from datasets import Value, load_dataset

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from custom_model import CodeBERTSCForSequenceClassification, CodeBERTCommentAttForSequenceClassification
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from sklearn.metrics import accuracy_score, f1_score
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from training_utils import parse_config, print_summary
import yaml
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.


logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file."
                'If not specified, will use the "sentence" column for single/multi-label classifcation task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file."
                'If not specified, will use the "label" column for single/multi-label classifcation task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_query_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    sc_comment: bool = field(
        default=False,
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    cross_validation: bool = field(
        default=False, metadata={"help": "Whether to use cross validation."}
    )
    extra_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the extra training data."}
    )
    ignore_label: Optional[str] = field(
        default=None
    )
    bin_classification: bool = field(
        default=False, metadata={"help": "Whether to use cross validation."}
    )
    remove_special_tokens: bool = field(
        default=False,
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_short_name: str = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    lora_config_path: str = field(
        default=None,
    )
    load_in_8bit: bool = field(
        default=False,
    )
    is_llm: bool = field(
        default=False,
    )
    is_enc_dec: bool = field(
        default=False,
    )
    low_cpu_mem_usage: bool = field(
        default=False,
    )
    device_map: str = field(
        default=None,
    )

special_tokens = ["todo", "xxx", "fixme"]
def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a mutli-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list

def get_model_class(model_name):
    if model_name == "codebert-sc":
        return CodeBERTSCForSequenceClassification
    elif "comment-att" in model_name:
        return CodeBERTCommentAttForSequenceClassification
    else:
        return AutoModelForSequenceClassification


LLM_TEMPLATE ="""Task: Analyze the provided code the provided code to identify technical debts, which can be categorized into the following four types:\\

DESIGN: indicate misplaced code, lack of abstraction, long methods, poor implementation, workarounds or a temporary solution.
IMPLEMENTATION: incompleteness of the functionality in the method, class or program
DEFECT: bugs or errors in the code that affect functionality.
TEST: express the need for creation or improvement of the current set of tests.
If the code does not exhibit any of these types of technical debt, respond with NonTD.
 
Code to Analyze: {code}

Answer: """

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_classification", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files, or specify a dataset name
    # to load from huggingface/datasets. In ether case, you can specify a the key of the column(s) containing the text and
    # the key of the column containing the label. If multiple columns are specified for the text, they will be joined togather
    # for the actual text value.
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if data_args.cross_validation:
      
        training_files = [os.path.join(data_args.train_file, filename) for filename in os.listdir(data_args.train_file)]
        output_parent_dir = training_args.output_dir

        if data_args.extra_file is not None:
            training_files.append(data_args.extra_file)

        for test_file in training_files:
            training_args.output_dir = os.path.join(output_parent_dir, test_file.split("/")[-1].split(".")[0])

            train_files = list(set(training_files) - set([test_file]))

            # Loading a dataset from your local files.
            # CSV/JSON training and evaluation files are needed.
            data_files = {"train": train_files}
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
            else:
                data_files["validation"] = train_files

            
            # Get the test dataset: you can provide your own CSV/JSON test file
            if training_args.do_predict:
                data_files["test"] = test_file

            for key in data_files.keys():
                logger.info(f"load a local file for {key}: {data_files[key]}")

            if train_files[0].endswith(".csv"):
                # Loading a dataset from local csv files
                raw_datasets = load_dataset(
                    "csv",
                    data_files=data_files,
                    # cache_dir=model_args.cache_dir,
                )
            else:
                # Loading a dataset from local json files
                raw_datasets = load_dataset(
                    "json",
                    data_files=data_files,
                    # cache_dir=model_args.cache_dir,
                )
   
            if data_args.bin_classification:
                def change_label(example):
                    example[data_args.label_column_name] = "SATD" if example[data_args.label_column_name] != "NONSATD" and example[data_args.label_column_name] != "WITHOUT_CLASSIFICATION" else "NONSATD"
                    return example
                raw_datasets = raw_datasets.map(change_label)

            if data_args.ignore_label is not None:
                for subset in raw_datasets:
                    raw_datasets[subset] = raw_datasets[subset].filter(lambda x: x[data_args.label_column_name] != data_args.ignore_label)

            if data_args.remove_columns is not None:
                for split in raw_datasets.keys():
                    for column in data_args.remove_columns.split(","):
                        logger.info(f"removing column {column} from split {split}")
                        raw_datasets[split].remove_columns(column)

            if data_args.label_column_name is not None and data_args.label_column_name != "label":
                for key in raw_datasets.keys():
                    raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

            # Trying to have good defaults here, don't hesitate to tweak to your needs.
            is_multi_label = False
            if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
                is_multi_label = True
                logger.info("Label type is list, doing multi-label classification")
            # Trying to find the number of labels in a multi-label classification task
            # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
            # So we build the label list from the union of labels in train/val/test.
            label_list = get_label_list(raw_datasets, split="train")
            for split in ["validation", "test"]:
                if split in raw_datasets:
                    val_or_test_labels = get_label_list(raw_datasets, split=split)
                    diff = set(val_or_test_labels).difference(set(label_list))
                    if len(diff) > 0:
                        # add the labels that appear in val/test but not in train, throw a warning
                        logger.warning(
                            f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                        )
                        label_list += list(diff)
            # if label is -1, we throw a warning and remove it from the label list
            for label in label_list:
                if label == -1:
                    logger.warning("Label -1 found in label list, removing it.")
                    label_list.remove(label)

            label_list.sort()
            if is_multi_label:
                num_labels = len(raw_datasets["train"]["label"][0])
            else:
                num_labels = len(label_list)
            if num_labels <= 1:
                raise ValueError("You need more than one label to do classification.")

    
            # Load pretrained model and tokenizer
            # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
            # download model & vocab.
            config = AutoConfig.from_pretrained(
                model_args.config_name if model_args.config_name else model_args.model_name_or_path,
                num_labels=num_labels,
                finetuning_task="text-classification",
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                token=model_args.token,
                trust_remote_code=model_args.trust_remote_code,
            )

            if is_multi_label:
                config.problem_type = "multi_label_classification"
                logger.info("setting problem type to multi label classification")
            else:
                config.problem_type = "single_label_classification"
                logger.info("setting problem type to single label classification")

            tokenizer = AutoTokenizer.from_pretrained(
                model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
                cache_dir=model_args.cache_dir,
                use_fast=model_args.use_fast_tokenizer,
                revision=model_args.model_revision,
                trust_remote_code=model_args.trust_remote_code,
            )

            if model_args.is_llm:
                tokenizer.padding_side = 'left'
            

            if data_args.max_seq_length > tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
            config.sc_comment = data_args.sc_comment
            config.max_query_length = data_args.max_query_length
            config.max_seq_length = max_seq_length - data_args.max_query_length - 4


            model = get_model_class(model_args.model_short_name).from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                low_cpu_mem_usage= model_args.low_cpu_mem_usage, 
                device_map= model_args.device_map, 
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                trust_remote_code=model_args.trust_remote_code,
                ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
                resume_download= True
            )

            
            
            # Define LoRA Config
            if model_args.lora_config_path is not None:
                _lora_config = yaml.load(open(model_args.lora_config_path), Loader = yaml.FullLoader)
                _lora_config = parse_config(_lora_config)
                lora_config = LoraConfig(
                    r=_lora_config.r,
                    target_modules=_lora_config.target_modules,
                    lora_alpha=_lora_config.alpha,
                    lora_dropout=_lora_config.dropout,
                    bias=_lora_config.bias,
                    task_type=TaskType.CAUSAL_LM
                )

                # overcome gradient checkpointing error
                # see https://github.com/huggingface/transformers/issues/23170
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

                # prepare int-8 model for training
                if model_args.load_in_8bit:
                    model = prepare_model_for_kbit_training(model)

                model = get_peft_model(model, lora_config)
                model.print_trainable_parameters()
            
            if not tokenizer.pad_token:
                tokenizer.pad_token_id = tokenizer.eos_token_id
                tokenizer.pad_token = tokenizer.eos_token
                # model.config.pad_token = model.config.eos_token
            
            if not model.config.pad_token_id:
                model.config.pad_token_id = tokenizer.pad_token_id

            # Padding strategy
            if data_args.pad_to_max_length:
                padding = "max_length"
            else:
                # We will pad later, dynamically at batch creation, to the max sequence length in each batch
                padding = False

            # for training ,we will update the config with label infos,
            # if do_train is not set, we will use the label infos in the config
            if training_args.do_train:  # classification, training
                label_to_id = {v: i for i, v in enumerate(label_list)}
                # update config with label infos
                if model.config.label2id != label_to_id:
                    logger.warning(
                        "The label2id key in the model config.json is not equal to the label2id key of this "
                        "run. You can ignore this if you are doing finetuning."
                    )
                model.config.label2id = label_to_id
                model.config.id2label = {id: label for label, id in config.label2id.items()}
            else:  # classification, but not training
                logger.info("using label infos in the model config")
                logger.info("label2id: {}".format(model.config.label2id))
                label_to_id = model.config.label2id


            def multi_labels_to_ids(labels: List) -> List[float]:
                if len(set(labels)) <= 2 and (0 in labels or 1 in labels):
                    ids = [1.0 * x for x in labels]
                else:
                    ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
                    for label in labels:
                        ids[label_to_id[label]] = 1.0
                return ids

            def preprocess_function(examples):
                if data_args.text_column_names is not None:
                    text_column_names = data_args.text_column_names.split(",")
                    # join together text columns into "sentence" column
                    # result = {"input_ids": [], "attention_mask": [], "pq_end_pos": []}
                    # if len(text_column_names) == 2:
                    #     for comment, code in zip(examples[text_column_names[0]], examples[text_column_names[1]]):
                    #         comment_input_ids = tokenizer.encode(comment, add_special_tokens=False)[:data_args.max_query_length]
                    #         code_input_ids = tokenizer.encode(code, add_special_tokens=False)[:max_seq_length - data_args.max_query_length - 4]                    
                    #         input_ids = [tokenizer.bos_token_id] + comment_input_ids + [tokenizer.eos_token_id, tokenizer.eos_token_id] + code_input_ids + [tokenizer.eos_token_id]
                    #         attention_mask = [1] * len(input_ids)
                            
                    #         pq_end_pos = [len(comment_input_ids), len(input_ids)]
                    #         if data_args.pad_to_max_length:
                    #             pq_end_pos += [0] * (max_seq_length - len(pq_end_pos))
                    #         if len(input_ids) < max_seq_length and data_args.pad_to_max_length:
                    #             input_ids += [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
                    #             attention_mask += [0] * (max_seq_length - len(attention_mask))
                                
                    #         result["input_ids"].append(input_ids)
                    #         result["attention_mask"].append(attention_mask)
                    #         result["pq_end_pos"].append(pq_end_pos)
                            
                    # else:
                    examples["sentence"] = examples[text_column_names[0]]
                    if data_args.remove_special_tokens:
                        for i in range(len(examples["sentence"])):
                            examples["sentence"][i] = " ".join([x for x in examples["sentence"][i].split(" ") if x.lower() not in special_tokens])

                    if model_args.is_llm:
                        for i in range(len(examples["sentence"])):
                            examples["sentence"][i] = LLM_TEMPLATE.format(code=examples["sentence"][i])

                    for column in text_column_names[1:]:
                        for i in range(len(examples[column])):
                            if data_args.remove_special_tokens:
                                examples["sentence"][i] += data_args.text_column_delimiter + " ".join([x for x in examples[column][i].split(" ") if x.lower() not in special_tokens])
                            else:
                                examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
                    
                    result = tokenizer(examples["sentence"], padding=padding, max_length=min(max_seq_length+ data_args.max_query_length,  tokenizer.model_max_length), truncation=True)            
                # Tokenize the texts
                else:
                    result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
                if label_to_id is not None and "label" in examples:
                    if is_multi_label:
                        result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
                    else:
                        result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]

                return result
            
            if "comment-att" in model_args.model_short_name:
                def preprocess_function(examples):

                    text_column_names = data_args.text_column_names.split(",")
                    # join together text columns into "sentence" column
                    result = {"comment_input_ids": [], "comment_attention_mask": [], "code_input_ids": [], "code_attention_mask": [],}
                    
                    tokenized_comment = tokenizer(examples[text_column_names[0]], padding=padding, max_length=max_seq_length, truncation=True)
                    result["comment_input_ids"] = tokenized_comment["input_ids"]
                    result["comment_attention_mask"] = tokenized_comment["attention_mask"]

                    tokenized_code = tokenizer(examples[text_column_names[1]], padding=padding, max_length=data_args.max_query_length, truncation=True)
                    result["code_input_ids"] = tokenized_code["input_ids"]
                    result["code_attention_mask"] = tokenized_code["attention_mask"]          
   
                    if label_to_id is not None and "label" in examples:
                        if is_multi_label:
                            result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
                        else:
                            result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
                    return result

            if is_multi_label:
                print(raw_datasets)
                for subset in raw_datasets:
                    raw_datasets[subset] = raw_datasets[subset].filter(lambda x: len(tokenizer.encode(x["cleancode"])) <= data_args.max_seq_length)
                print("After filtering")
                print(raw_datasets)
            
            # Running the preprocessing pipeline on all the datasets
            with training_args.main_process_first(desc="dataset map pre-processing"):
                raw_datasets = raw_datasets.map(
                    preprocess_function,
                    batched=True,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )



            if training_args.do_train:
                if "train" not in raw_datasets:
                    raise ValueError("--do_train requires a train dataset.")
                train_dataset = raw_datasets["train"]
                if data_args.shuffle_train_dataset:
                    logger.info("Shuffling the training dataset")
                    train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
                if data_args.max_train_samples is not None:
                    max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                    train_dataset = train_dataset.select(range(max_train_samples))

            if training_args.do_eval:
                if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
                    if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                        raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
                    else:
                        logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                        eval_dataset = raw_datasets["test"]
                else:
                    eval_dataset = raw_datasets["validation"]

                if data_args.max_eval_samples is not None:
                    max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                    eval_dataset = eval_dataset.select(range(max_eval_samples))
                

            if training_args.do_predict or data_args.test_file is not None:
                if "test" not in raw_datasets:
                    raise ValueError("--do_predict requires a test dataset")
                predict_dataset = raw_datasets["test"]
                # remove label column if it exists
                if data_args.max_predict_samples is not None:
                    max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                    predict_dataset = predict_dataset.select(range(max_predict_samples))

            # Log a few random samples from the training set:
            if training_args.do_train:
                for index in random.sample(range(len(train_dataset)), 3):
                    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
                    for key in train_dataset[index].keys():
                        if "input_ids" in key:
                            logger.info(key + ": " + tokenizer.decode(train_dataset[index][key]))

            if data_args.metric_name is not None:
                metric = (
                    evaluate.load(data_args.metric_name, config_name="multilabel")
                    if is_multi_label
                    else evaluate.load(data_args.metric_name)
                )
                logger.info(f"Using metric {data_args.metric_name} for evaluation.")
            else: 
                if is_multi_label:
                    metric = evaluate.load("f1", config_name="multilabel")
                    logger.info(
                        "Using multilabel F1 for multi-label classification task, you can use --metric_name to overwrite."
                    )
                else:
                    metric = evaluate.load("accuracy")
                    logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")


            def compute_metrics(p: EvalPrediction):
                preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions


                if is_multi_label:
                    preds = np.array([np.where(p > 0, 1, 0) for p in preds])  # convert logits to multi-hot encoding
                    # Micro F1 is commonly used in multi-label classification
                    result = metric.compute(predictions=preds, references=p.label_ids, average="micro")
                else:
                    preds = np.argmax(preds, axis=1)
                    if "f1" in data_args.metric_name:
                        result = metric.compute(predictions=preds, references=p.label_ids, average="macro")
                    else:
                        result = metric.compute(predictions=preds, references=p.label_ids)
                if len(result) > 1:
                    result["combined_score"] = np.mean(list(result.values())).item()
                return result

            # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
            # we already did the padding.
            if data_args.pad_to_max_length:
                data_collator = default_data_collator
            elif training_args.fp16:
                data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
            else:
                data_collator = None

            # Initialize our Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_dataset if training_args.do_eval else None,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )

            # Training
            if training_args.do_train:
                checkpoint = None
                if training_args.resume_from_checkpoint is not None:
                    checkpoint = training_args.resume_from_checkpoint
                elif last_checkpoint is not None:
                    checkpoint = last_checkpoint
                train_result = trainer.train(resume_from_checkpoint=checkpoint)
                metrics = train_result.metrics
                max_train_samples = (
                    data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
                )
                metrics["train_samples"] = min(max_train_samples, len(train_dataset))
                trainer.save_model()  # Saves the tokenizer too for easy upload
                trainer.log_metrics("train", metrics)
                trainer.save_metrics("train", metrics)
                trainer.save_state()

            # Evaluation
            if training_args.do_eval:
                logger.info("*** Evaluate ***")
                metrics = trainer.evaluate(eval_dataset=eval_dataset)
                max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
                metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
                trainer.log_metrics("eval", metrics)
                trainer.save_metrics("eval", metrics)

            if training_args.do_predict:
                logger.info("*** Predict ***")
                # Removing the `label` columns if exists because it might contains -1 and Trainer won't like that.
                if "label" in predict_dataset.features:
                    labels = predict_dataset["label"]
                    predict_dataset = predict_dataset.remove_columns("label")
                predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
                if is_multi_label:
                    # Convert logits to multi-hot encoding. We compare the logits to 0 instead of 0.5, because the sigmoid is not applied.
                    # You can also pass `preprocess_logits_for_metrics=lambda logits, labels: nn.functional.sigmoid(logits)` to the Trainer
                    # and set p > 0.5 below (less efficient in this case)
                    if model_args.is_enc_dec:
                        predictions = predictions[0]
                    predictions = np.array([np.where(p > 0, 1, 0) for p in predictions])
                else:
                    predictions = np.argmax(predictions, axis=1)

                output_predict_file = os.path.join(training_args.output_dir, test_file.split("/")[-1].split(".")[0] + "_predict_results.txt")
                if trainer.is_world_process_zero():
                    with open(output_predict_file, "w") as writer:
                        logger.info("***** Predict results *****")
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(predictions):
                            if is_multi_label:
                                # recover from multi-hot encoding
                                item = " ".join([str(int(i)) for i in item])
                                writer.write(f"{index}\t{item}\n")
                            else:
                                item = label_list[item]
                                writer.write(f"{index}\t{item}\n")
                logger.info("Predict results saved at {}".format(output_predict_file))
                
                if not is_multi_label:
                    score_summary = {"accuracy": accuracy_score(labels, predictions),
                                    "macro_f1": f1_score(labels, predictions, average='macro')}
                    with open(os.path.join(training_args.output_dir, test_file.split("/")[-1].split(".")[0] + "_predict_score.json"), "w") as f:
                        json.dump(score_summary, f, indent=4)
        print("FINISHED!!!")
    else:
        training_files = [data_args.train_file]
        output_parent_dir = training_args.output_dir

        if data_args.extra_file is not None:
            training_files.append(data_args.extra_file)


        training_args.output_dir = os.path.join(output_parent_dir, data_args.train_file.split("/")[-1].split(".")[0])


        # Loading a dataset from your local files.
        # CSV/JSON training and evaluation files are needed.
        data_files = {"train": training_files}
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        else:
            data_files["validation"] = training_files

        
        # Get the test dataset: you can provide your own CSV/JSON test file
        if training_args.do_predict and data_args.test_file:
            data_files["test"] = data_args.test_file

        for key in data_files.keys():
            logger.info(f"load a local file for {key}: {data_files[key]}")

        if training_files[0].endswith(".csv"):
            # Loading a dataset from local csv files
            raw_datasets = load_dataset(
                "csv",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
            )
        else:
            # Loading a dataset from local json files
            raw_datasets = load_dataset(
                "json",
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                )

        if data_args.remove_columns is not None:
            for split in raw_datasets.keys():
                for column in data_args.remove_columns.split(","):
                    logger.info(f"removing column {column} from split {split}")
                    raw_datasets[split].remove_columns(column)

        if data_args.label_column_name is not None and data_args.label_column_name != "label":
            for key in raw_datasets.keys():
                raw_datasets[key] = raw_datasets[key].rename_column(data_args.label_column_name, "label")

        # Trying to have good defaults here, don't hesitate to tweak to your needs.

        if raw_datasets["train"].features["label"].dtype == "list":  # multi-label classification
            is_multi_label = True
            logger.info("Label type is list, doing multi-label classification")
        # Trying to find the number of labels in a multi-label classification task
        # We have to deal with common cases that labels appear in the training set but not in the validation/test set.
        # So we build the label list from the union of labels in train/val/test.
        label_list = get_label_list(raw_datasets, split="train")
        for split in ["validation", "test"]:
            if split in raw_datasets:
                val_or_test_labels = get_label_list(raw_datasets, split=split)
                diff = set(val_or_test_labels).difference(set(label_list))
                if len(diff) > 0:
                    # add the labels that appear in val/test but not in train, throw a warning
                    logger.warning(
                        f"Labels {diff} in {split} set but not in training set, adding them to the label list"
                    )
                    label_list += list(diff)
        # if label is -1, we throw a warning and remove it from the label list
        for label in label_list:
            if label == -1:
                logger.warning("Label -1 found in label list, removing it.")
                label_list.remove(label)

        label_list.sort()
        if is_multi_label:
            num_labels = len(raw_datasets["train"]["label"][0])
        else:
            num_labels = len(label_list)
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

        # Load pretrained model and tokenizer
        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task="text-classification",
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

        config.problem_type = "single_label_classification"
        logger.info("setting problem type to single label classification")

        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
        )

        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        # for training ,we will update the config with label infos,
        # if do_train is not set, we will use the label infos in the config
        if training_args.do_train:  # classification, training
            label_to_id = {v: i for i, v in enumerate(label_list)}
            # update config with label infos
            if model.config.label2id != label_to_id:
                logger.warning(
                    "The label2id key in the model config.json is not equal to the label2id key of this "
                    "run. You can ignore this if you are doing finetuning."
                )
            model.config.label2id = label_to_id
            model.config.id2label = {id: label for label, id in config.label2id.items()}
        else:  # classification, but not training
            logger.info("using label infos in the model config")
            logger.info("label2id: {}".format(model.config.label2id))
            label_to_id = model.config.label2id

        if data_args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

        def multi_labels_to_ids(labels: List[str]) -> List[float]:
            ids = [0.0] * len(label_to_id)  # BCELoss requires float as target type
            for label in labels:
                ids[label_to_id[label]] = 1.0
            return ids

        def preprocess_function(examples):
            if data_args.text_column_names is not None:
                text_column_names = data_args.text_column_names.split(",")
                # join together text columns into "sentence" column
                examples["sentence"] = examples[text_column_names[0]]
                for column in text_column_names[1:]:
                    for i in range(len(examples[column])):
                        examples["sentence"][i] += data_args.text_column_delimiter + examples[column][i]
            # Tokenize the texts
            result = tokenizer(examples["sentence"], padding=padding, max_length=max_seq_length, truncation=True)
            if label_to_id is not None and "label" in examples:
                result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
            return result

        # Running the preprocessing pipeline on all the datasets
        with training_args.main_process_first(desc="dataset map pre-processing"):
            raw_datasets = raw_datasets.map(
                preprocess_function,
                batched=True,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
            
        if is_multi_label:
            print(raw_datasets)
            for subset in raw_datasets:
                raw_datasets[subset] = raw_datasets[subset].filter(lambda x: len(tokenizer.encode(x["cleancode"])) <= data_args.max_seq_length)
            print("After filtering")
            print(raw_datasets)

        if training_args.do_train:
            if "train" not in raw_datasets:
                raise ValueError("--do_train requires a train dataset.")
            train_dataset = raw_datasets["train"]
            if data_args.shuffle_train_dataset:
                logger.info("Shuffling the training dataset")
                train_dataset = train_dataset.shuffle(seed=data_args.shuffle_seed)
            if data_args.max_train_samples is not None:
                max_train_samples = min(len(train_dataset), data_args.max_train_samples)
                train_dataset = train_dataset.select(range(max_train_samples))

        if training_args.do_eval:
            if "validation" not in raw_datasets and "validation_matched" not in raw_datasets:
                if "test" not in raw_datasets and "test_matched" not in raw_datasets:
                    raise ValueError("--do_eval requires a validation or test dataset if validation is not defined.")
                else:
                    logger.warning("Validation dataset not found. Falling back to test dataset for validation.")
                    eval_dataset = raw_datasets["test"]
            else:
                eval_dataset = raw_datasets["validation"]

            if data_args.max_eval_samples is not None:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))

        if training_args.do_predict or data_args.test_file is not None:
            if "test" not in raw_datasets:
                split_dataset = train_dataset.train_test_split(test_size=0.1)
                train_dataset, predict_dataset = split_dataset["train"], split_dataset["test"]
                predict_dataset.to_json(os.path.join(training_args.output_dir, "grouth_truth.json"))
                exit()
            else:
                predict_dataset = raw_datasets["test"]

        # Log a few random samples from the training set:
        if training_args.do_train:
            for index in random.sample(range(len(train_dataset)), 3):
                # logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
                print(tokenizer.decode(train_dataset[index]["input_ids"]))


        if data_args.metric_name is not None:
            metric = (
                evaluate.load(data_args.metric_name)
            )
            logger.info(f"Using metric {data_args.metric_name} for evaluation.")
        else: 
            metric = evaluate.load("accuracy")
            logger.info("Using accuracy as classification score, you can use --metric_name to overwrite.")

        def compute_metrics(p: EvalPrediction):
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            preds = np.argmax(preds, axis=1)
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result

        # Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
        # we already did the padding.
        if data_args.pad_to_max_length:
            data_collator = default_data_collator
        elif training_args.fp16:
            data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
        else:
            data_collator = None

        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
        )

        # Training
        if training_args.do_train:
            checkpoint = None
            if training_args.resume_from_checkpoint is not None:
                checkpoint = training_args.resume_from_checkpoint
            elif last_checkpoint is not None:
                checkpoint = last_checkpoint
            train_result = trainer.train(resume_from_checkpoint=checkpoint)
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))
            trainer.save_model()  # Saves the tokenizer too for easy upload
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()

        # Evaluation
        if training_args.do_eval:
            logger.info("*** Evaluate ***")
            metrics = trainer.evaluate(eval_dataset=eval_dataset)
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)

        if training_args.do_predict:
            logger.info("*** Predict ***")
            # Removing the `label` columns if exists because it might contains -1 and Trainer won't like that.
            if "label" in predict_dataset.features:
                labels = list(predict_dataset["label"])
                predict_dataset = predict_dataset.remove_columns("label")
            predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
            predictions = list(np.argmax(predictions, axis=1))
            output_predict_file = os.path.join(training_args.output_dir, data_args.train_file.split("/")[-1].split(".")[0] + "_predict_results.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info("***** Predict results *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")
            logger.info("Predict results saved at {}".format(output_predict_file))

            score_summary = {"accuracy": accuracy_score(labels, predictions),
                            "macro_f1": f1_score(labels, predictions, average='macro')}
            with open(os.path.join(training_args.output_dir, data_args.train_file.split("/")[-1].split(".")[0] + "_predict_score.json"), "w") as f:
                json.dump(score_summary, f, indent=4)

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()