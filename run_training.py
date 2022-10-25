"""
Installed prerequisites:
!pip install transformers
!pip install datasets
!pip install sentencepiece
!pip install sacrebleu
!pip install bert_score
!pip install -U nltk
"""
import json
import os

import time

from typing import TextIO

import torch
import nltk
import argparse
import numpy as np
from nltk.translate import meteor_score
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset, load_metric, Dataset  # , tqdm
from datasets.dataset_dict import DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, BatchEncoding, PreTrainedTokenizer, AutoModel, PreTrainedModel,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, T5Tokenizer, AutoTokenizer)
from easse.sari import corpus_sari, compute_macro_sari

from helpers import SariSeq2SeqTrainer, MultiRefEvaluator

from transformers.file_utils import (
    is_torch_tpu_available,
)

from transformers.trainer_utils import (
    set_seed,
)
from transformers.utils import logging

if is_torch_tpu_available():
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

# from pynvml import *

# TODO: Fix comment outs
# TODO: Add functionality descriptions


def set_up_global_vars():
    pass


def add_prefix(example):
    """
    Add prefix of simplify: to start of input
    :param example:
    :return:
    """
    example['source'] = 'simplify: ' + example['source']
    return example


def add_t5_mask(example):
    """
    Add particular mask for T5 model
    :param example:
    :return:
    """
    example['source'] = f"<extra_id_0> {example['source']} <extra_id_1>"
    if isinstance(example['target'], str):
        example['target'] = f"<extra_id_0> {' '.join([f'<{a}>' for a in example['actions']])} <extra_id_1> {example['target']} <extra_id_2>"
    elif isinstance(example['target'], list):
        example['target'] = [f"<extra_id_0> {' '.join([f'<{a}>' for a in example['actions']])} <extra_id_1> {t} <extra_id_2>" for t in example['target']]

    return example


def add_bart_mask(example):
    """
    Add masking for BART models
    :param example:
    :return:
    """
    example['source'] = f"<mask> {example['source']}"
    if isinstance(example['target'], str):
        example['target'] = f"{' '.join([f'<{a}>' for a in example['actions']])} {example['target']}"
    elif isinstance(example['target'], list):
        example['target'] = [f"{' '.join([f'<{a}>' for a in example['actions']])} {t}" for t in example['target']]

    return example


def preprocess_function(examples):
    """
    Take samples from a hf.Dataset and convert to tokenized inputs for model.
    :param examples:
    :return:
    """
    model_inputs = tokenizer(examples['source'], padding="max_length", truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        refs = [t for t in examples['target']]
        if isinstance(refs[0], str):  # if has single target sentence
            labels = tokenizer(examples['target'], padding="max_length", truncation=True)
            if 't5' in tokenizer.name_or_path:  # T5 uses -100 as padding token. BART uses 1 as padding token
                labels["input_ids"] = [np.where(np.array(label) != 0, np.array(label), -100).tolist() for label in labels["input_ids"]]
        elif isinstance(refs[0], list):  # if has multiple target senteces (ASSET, TURKCORPUS)
            tokenized = [[tokenizer(r, padding="max_length", truncation=True) for r in ref_group] for ref_group in refs]
            # [item for sublist in t for item in sublist]
            labels = BatchEncoding()
            res = []
            for ref_group in tokenized:
                subres = []
                for ref in ref_group:
                    subres.append(ref["input_ids"])
                res.append(subres)
            labels["input_ids"] = res

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_datasets(args):
    if args.training_dataset == "manual":
        dataset_script_name = "create_datasets"
    else:
        dataset_script_name = "wiki-auto_with_actions"
    if args.all_action:
        if args.transformers_cache is not None:
            dataset = load_dataset(f"create_hf_datasets/{dataset_script_name}.py", cache_dir=args.transformers_cache)
        else:
            dataset = load_dataset(f"create_hf_datasets/{dataset_script_name}.py")
        if args.eval_dataset == 'asset':
            if args.transformers_cache is not None:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py", cache_dir=args.transformers_cache)
            else:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py")
        elif args.eval_dataset == 'turk':
            asset = None
    elif args.no_action or args.t5_classification or args.bart_classification:  # to perform the t5/bart classification we need examples where the actions are not added. makes the map function easier.
        if args.transformers_cache is not None:
            dataset = load_dataset(f"create_hf_datasets/{dataset_script_name}.py", "all_json_act-none",
                                   cache_dir=args.transformers_cache)
        else:
            dataset = load_dataset(f"create_hf_datasets/{dataset_script_name}.py", "all_json_act-none")
        if args.eval_dataset == 'asset':
            if args.transformers_cache is not None:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py", "all_json_act-none",
                                     cache_dir=args.transformers_cache)
            else:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py", "all_json_act-none")
        elif args.eval_dataset == 'turk':
            asset = None
    elif args.single_action:
        if args.transformers_cache is not None:
            dataset = load_dataset(f"create_hf_datasets/{dataset_script_name}.py", "all_json_act-single",
                                   cache_dir=args.transformers_cache)
        else:
            dataset = load_dataset(f"create_hf_datasets/{dataset_script_name}.py", "all_json_act-single")
        if args.eval_dataset == 'asset':
            if args.transformers_cache is not None:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py", "all_json_act-single",
                                     cache_dir=args.transformers_cache)
            else:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py", "all_json_act-single")
        elif args.eval_dataset == 'turk':
            asset = None
    else:
        raise ValueError("No argument type provided")

    if _ADD_T5_HEADER:
        dataset = dataset.map(add_prefix)
        asset = asset.map(add_prefix)

    return dataset, asset


def compute_metrics(sources, predictions, references):
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(references != -100, references, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_sources = tokenizer.batch_decode(sources, skip_special_tokens=True)

    result = {}

    # Remove the task header ("simplify:") from the sources
    if _ADD_T5_HEADER:
        decoded_sources = [source[len("simplify:"):].strip() for source in decoded_sources]

    if _COMPUTE_BERTSCORE:
        # bertscore expects regular sentences
        bertscore_result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang='en')
        # print(bertscore_result)
        bertscore_result = {f"bertscore_{k}": np.mean(v) for k, v in bertscore_result.items() if k != 'hashcode'}

    # BLEU expects each prediction to be a tokenized sentence and each reference to be a list of tokenized sentences
    tokenized_preds = [nltk.word_tokenize(pred.strip()) for pred in decoded_preds]
    tokenized_labels = [nltk.word_tokenize(label.strip()) for label in decoded_labels]
    tokenized_sources = [nltk.word_tokenize(source.strip()) for source in decoded_sources]

    bleu_result = bleu.compute(predictions=tokenized_preds, references=[[label] for label in tokenized_labels])
    # for k,v in bleu_result.items():
    #   print(f"{k}: {v}")
    for i, precision in enumerate(bleu_result['precisions']):
        bleu_result[f'{i + 1}-gram_precision'] = precision
    bleu_result.pop("precisions")
    # for k in bleu_result.keys():
    #     if k != 'bleu':
    #       bleu_result[f"bleu_{k}"] = bleu_result.pop(k)
    # bleu_result = {f"bleu_{k}": v for k, v in bleu_result.items()}

    # Sari expects a space delimited tokenised sentence, and the references to be a list of space delimited tokenized senteces
    # METEOR expects a space delimited tokenised sentence
    decoded_preds = [" ".join(pred) for pred in tokenized_preds]
    decoded_labels = [" ".join(label) for label in tokenized_labels]
    decoded_sources = [" ".join(source) for source in tokenized_sources]
    meteor_result = meteor.compute(predictions=decoded_preds, references=decoded_labels)
    sair_result = sari.compute(sources=decoded_sources, predictions=decoded_preds,
                               references=[[label] for label in decoded_labels])
    if _COMPUTE_BERTSCORE:
        result = {**sair_result, **meteor_result, **bleu_result, **bertscore_result}
    else:
        result = {**sair_result, **meteor_result, **bleu_result}
    # print(result)
    return {k: round(v, 4) for k, v in result.items()}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def create_model_name(args: argparse.Namespace):
    keys_for_name = ["training_dataset", "model_name",
                     #"compute_bertscore",
                     "add_t5_header",
                     "max_gen_len", "gen_beams_num",
                     #"train_batch_size", "eval_batch_size", "grad_accum_steps",
                     #"warm_up_steps",
                     "train_epochs", "fp16",
                     "adafactor",
                     ]
    result = ""
    for k in keys_for_name:
        if k == "model_name" and "bart" in args.__dict__[k]:
            result = f"{result}&{k}={args.__dict__[k].split('/')[-1]}"
        else:
            result = f"{result}&{k}={args.__dict__[k]}"

    result = f"{result}&time={_DATE}"

    return result


# def get_cuda_memory():
#     # t = torch.cuda.get_device_properties(0).total_memory
#     # r = torch.cuda.memory_reserved()
#     # a = torch.cuda.memory_allocated()
#     # f = r-a  # free inside reserved
#     # return {"total": t, "reserved": r, "allocated": a, "free": f}
#     # return torch.cuda.memory_stats_as_nested_dict()
#     h = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(h)
#     return {'total': info.total, 'free': info.free, 'used': info.used}


def assert_legal_args(args):
    if 't5' in args.model_name:
        assert not args.bart_classification
    if 'bart' in args.model_name:
        assert not args.t5_classification
        assert not args.add_t5_header


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train a T5 model on Text Simplification with Actions")
    parser.add_argument('--eval_dataset', type=str, default='asset')
    parser.add_argument('--training_dataset', type=str, required=True, choices=['manual', 'wiki-auto'])
    parser.add_argument('--model_name', required=True, choices=['t5-small', 't5-base', 't5-large', 't5-3b', 't5-11b',
                                                                'facebook/bart-base', 'facebook/bart-large'])
    parser.add_argument('--compute_bertscore', action='store_true')
    parser.add_argument('--add_t5_header', action='store_true')
    parser.add_argument('--transformers_cache', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--results_folder', type=str, default='.')
    parser.add_argument('--model_save_folder', type=str, default='.')
    parser.add_argument('--get_full_result', action='store_true')
    parser.add_argument('--debug', action='store_true')
    dataset_type_group = parser.add_mutually_exclusive_group(required=True)
    dataset_type_group.add_argument('--all_action', action='store_true')
    dataset_type_group.add_argument('--single_action', action='store_true')
    dataset_type_group.add_argument('--no_action', action='store_true')
    dataset_type_group.add_argument('--t5_classification', action='store_true')
    dataset_type_group.add_argument('--bart_classification', action='store_true')
    model_args_group = parser.add_argument_group()
    model_args_group.add_argument('--max_gen_len', type=int, default=200)
    model_args_group.add_argument('--model_version', type=int, required=True)
    model_args_group.add_argument('--gen_early_stop', action='store_true')
    model_args_group.add_argument('--gen_beams_num', type=int, default=1)
    training_args_group = parser.add_argument_group()
    training_args_group.add_argument('--train_batch_size', type=int, required=True)
    training_args_group.add_argument('--eval_batch_size', type=int, required=True)
    training_args_group.add_argument('--grad_accum_steps', type=int, required=True)
    training_args_group.add_argument('--lr', type=float, required=True)
    training_args_group.add_argument('--save_limit', type=int, default=2)
    training_args_group.add_argument('--save_steps', type=int, required=True)
    training_args_group.add_argument('--warm_up_steps', type=int, required=True)
    training_args_group.add_argument('--logging_steps', type=int, required=True)
    training_args_group.add_argument('--train_epochs', type=int, required=True)
    training_args_group.add_argument('--fp16', action='store_true')
    training_args_group.add_argument('--adafactor', action='store_true')
    training_args_group.add_argument('--eval_strategy', choices=["no", "steps", "epoch"], default="epoch")
    training_args_group.add_argument('--trunc_dataset', type=int)

    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # nvmlInit()

    args = parser.parse_args()
    assert_legal_args(args)
    # TODO: add assertions to argument formats (i.e. paths)
    # TO_DO: add loggers for step along the way
    # TO_DO: finalize all changes on the server to the git
    # TO_DO: add new naming conventions
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if is_torch_tpu_available():
        device = xm.xla_device()

    logger = logging.get_logger(__name__)
    if args.debug:
        logging.set_verbosity_debug()

    logger.warning(args)
    logger.warning("Set Global Vars")
    _COMPUTE_BERTSCORE = args.compute_bertscore
    _ADD_T5_HEADER = args.add_t5_header

    results_folder = args.results_folder

    set_seed(args.seed)

    _DATE = int(time.time())

    max_len = args.max_gen_len
    version = args.model_version

    model_config_dict = {
        "early_stopping": args.gen_early_stop,
        "num_beams": args.gen_beams_num,
        "max_length": max_len,
    }

    # logger.warning(f"GPU Memory State: {get_cuda_memory()}")
    logger.warning(f"Torch version: {torch.__version__}")
    logger.warning(f"Torch CUDA version: {torch.version.cuda}")

    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    grad_accum_steps = args.grad_accum_steps
    lr = args.lr
    save_limit = args.save_limit
    save_steps = args.save_steps
    warm_up_steps = args.warm_up_steps
    logging_steps = args.logging_steps
    train_epochs = args.train_epochs
    eval_strategy = args.eval_strategy
    fp16 = args.fp16
    adafactor = args.adafactor
    model_name = args.model_name.split("/")[-1]
    if args.all_action:
        save_dir = 'all_actions'
    elif args.no_action:
        save_dir = 'baseline'
    elif args.single_action:
        save_dir = 'single_action'
    elif args.t5_classification:
        save_dir = 't5_classification'
    elif args.bart_classification:
        save_dir = 'bart_classification'
    else:
        save_dir = ''
        raise ValueError("No argument type provided")
    model_save_name = f"{save_dir}{create_model_name(args)}"
    save_dir = f"{save_dir}{create_model_name(args)}"
    if args.model_save_folder != '.':
        save_dir = f"{args.model_save_folder}/{save_dir}"
    # save_dir = f"{save_dir}_maxlen_{max_len}_v{version}"

    nltk.download('punkt')

    logger.warning("Loading Metrics")
    if args.transformers_cache is not None:
        sari = load_metric("sari", cach_dir=args.transformers_cache)
        if _COMPUTE_BERTSCORE:
            bertscore = load_metric("bertscore", cach_dir=args.transformers_cache)

        meteor = load_metric("meteor", cach_dir=args.transformers_cache)
        bleu = load_metric("bleu", cach_dir=args.transformers_cache)
    else:
        sari = load_metric("sari")
        if _COMPUTE_BERTSCORE:
            bertscore = load_metric("bertscore")

        meteor = load_metric("meteor")
        bleu = load_metric("bleu")

    # logger.warning(f"GPU Memory State: {get_cuda_memory()}")

    logger.warning("Loading Datasets")
    datasets, asset = load_datasets(args)
    if args.trunc_dataset is not None:
        for k in datasets.keys():
            datasets[k] = datasets[k].select(range(args.trunc_dataset))
    # logger.warning(f"GPU Memory State: {get_cuda_memory()}")

    logger.warning("Loading Tokenizer")
    if args.transformers_cache is not None:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, cach_dir=args.transformers_cache)
    else:
        # tokenizer = AutoTokenizer.from_pretrained('/Users/eytan.chamovitz/Downloads/t5-small-for-debug')
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    special_toks = ["<PROX>", "<REPHRASE>", "<DEL>", "<ADD>", "<EXAMPLE>",
                    "<EXPLAIN>", "<EXPLICIT>", "<REORDER>", "<SPLIT>"]
    special_tokens_dict = {"additional_special_tokens": special_toks}
    tokenizer.add_special_tokens(special_tokens_dict)
    # logger.warning(f"GPU Memory State: {get_cuda_memory()}")

    if args.t5_classification:
        logger.warning("Processing Dataset for T5 Classification")
        datasets = datasets.map(add_t5_mask)
        asset = asset.map(add_t5_mask)
        # logger.warning(f"GPU Memory State: {get_cuda_memory()}")

    if args.bart_classification:
        logger.warning("Processing Dataset for BART Classification")
        datasets = datasets.map(add_bart_mask)
        asset = asset.map(add_bart_mask)

    logger.warning("Tokenizing Datasets")
    tokenized_datasets = datasets.map(preprocess_function, batched=True)
    asset_tokenized = asset.map(preprocess_function, batched=True)
    # logger.warning(f"GPU Memory State: {get_cuda_memory()}")

    # if 'bart' in model_name:
    #     model_config_dict["vocab_size"] = len(tokenizer.get_vocab())

    logger.warning("Loading Model")
    if args.transformers_cache is not None:
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, **model_config_dict,
                                                      cache_dir=args.transformers_cache)
    else:
        # model = AutoModelForSeq2SeqLM.from_pretrained('/Users/eytan.chamovitz/Downloads/t5-small-for-debug', **model_config_dict)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, **model_config_dict)
    # logger.warning(f"GPU Memory State: {get_cuda_memory()}")

    if 'bart' in model.name_or_path:
        model.resize_token_embeddings(len(tokenizer))
    logger.warning("Loading training args")
    training_args = Seq2SeqTrainingArguments(
        save_dir,
        evaluation_strategy=eval_strategy,
        learning_rate=lr,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        # weight_decay=wd,
        save_total_limit=save_limit,
        save_steps=save_steps,
        num_train_epochs=train_epochs,
        predict_with_generate=True,
        fp16=fp16,
        adafactor=adafactor,
        logging_steps=logging_steps,
        report_to="all",
        # tpu_num_cores=8,
        # push_to_hub=True,
        # push_to_hub_model_id=f"{model_name}-finetuned-xsum",
    )
    # logger.warning(f"GPU Memory State: {get_cuda_memory()}")

    logger.warning("Loading Collator")
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    # logger.warning(f"GPU Memory State: {get_cuda_memory()}")
    """## Define trainers"""

    logger.warning(f"Loading Trainer")
    trainer = SariSeq2SeqTrainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    # logger.warning(f"GPU Memory State: {get_cuda_memory()}")
    _COMPUTE_BERTSCORE = False
    logger.warning("Training")
    trainer.train()

    """## Save model"""
    # logger.warning(f"GPU Memory State: {get_cuda_memory()}")
    logger.warning("Saving Model")
    trainer.save_model()

    """## Predict"""

    logger.warning("Evaluating Model on test dataset")
    _COMPUTE_BERTSCORE = args.compute_bertscore
    try:
        preds = trainer.predict(tokenized_datasets['test'])
        logger.warning(f"Saving preds to {results_folder}/{model_save_name}.test_preds.json")
        with open(f"{results_folder}/{model_save_name}.test_preds.json", "w") as j:
            json.dump(preds._asdict(), j, cls=NumpyEncoder)

        if logger.getEffectiveLevel() < 30:
            # Show first 10 examples
            examples = tokenizer.batch_decode(preds.predictions[:10], skip_special_tokens=True)
            print(f"Sources:\n{tokenized_datasets['test'][:10]['source']}\nPredictions:\n{examples}")
    except KeyError:
        logger.error("No test split in dataset - continuing")

    logger.warning("Evaluating on ASSET")
    asset_evaluator = MultiRefEvaluator(asset_tokenized, tokenizer, device=device,
                                        batch_size=eval_batch_size, add_t5_header=_ADD_T5_HEADER)
    baseline_sari_score, _ = asset_evaluator.eval(model, split_aligns=True, split_actions=True, get_baseline=True)
    sari_score, result = asset_evaluator.eval(model, split_aligns=True, split_actions=True)

    pct_identical = asset_evaluator.get_pct_identical(result)

    logger.warning(f"Results for {save_dir}")
    logger.warning(f"% Identical: {pct_identical}\nResults DF:\n{asset_evaluator.get_results_dataframe_string(sari_score)}")
    if args.get_full_result:
        with open(f"{results_folder}/{model_save_name}.full_preds.txt", "w") as f:
            f.write(asset_evaluator.get_results_dataframe_string(sari_score))
            asset_evaluator.print_asset_results(f, save_dir, sari_score, pct_identical, result)
