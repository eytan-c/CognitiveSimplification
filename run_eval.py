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
import pathlib

from typing import TextIO

import torch
import nltk
import argparse
import time
import random
import numpy as np
from nltk.translate import meteor_score
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset, load_metric, Dataset
from datasets.dataset_dict import DatasetDict
from transformers import (
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, BatchEncoding, PreTrainedTokenizer, AutoModel, PreTrainedModel,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, T5Tokenizer, AutoTokenizer)
from easse.sari import corpus_sari, compute_macro_sari

from run_training import add_prefix, add_t5_mask, add_bart_mask
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


def parse_model_name(model_name: str) -> dict:
    split = model_name.split('&')
    kvs = [tuple(s.split('=')) for s in split]
    for i in range(len(kvs)):
        if len(kvs[i]) == 1:
            kvs[i] = ('model_type', kvs[i][0])
    return {k: v for (k, v) in kvs}


def create_cognitive_like_asset(example):
    example["target"] = [example["target"]]
    return example


def load_asset(model_type, eval_dataset, add_t5_header, cache_dir=None):
    corpus = "all"
    act_type = ""
    dataset_loading_script = ""
    if model_type == "all_actions":
        act_type = "all"
    elif model_type == "baseline" or model_type == "t5_classification" or model_type == "bart_classification" or model_type.startswith("GEM"):
        act_type = "none"
    elif model_type == "single_action":
        act_type = "single"
    else:
        raise ValueError("No argument type provided")

    if eval_dataset == 'asset':
        dataset_loading_script = "asset_with_actions.py"
    elif eval_dataset == 'cognitive':
        dataset_loading_script = "create_datasets.py"
        corpus = "cognitive"
    elif eval_dataset == 'turk':
        dataset_loading_script = ""
    else:
        dataset_loading_script = ""

    if len(dataset_loading_script) > 0:
        asset = load_dataset(f"create_hf_datasets/{dataset_loading_script}", name=f"{corpus}_json_act-{act_type}",
                             cache_dir=cache_dir)
    else:
        asset = None

    if eval_dataset == 'cognitive':
        asset = asset.map(create_cognitive_like_asset)

    if add_t5_header:
        asset = asset.map(add_prefix)
    return asset


def load_asset2(model_type, eval_dataset, add_t5_header):
    if model_type == "all_actions":
        if eval_dataset == 'asset':
            if args.transformers_cache is not None:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py", cache_dir=args.transformers_cache)
            else:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py")
        elif eval_dataset == 'cognitive':
            if args.transformers_cache is not None:
                asset = load_dataset("create_hf_datasets/create_datasets.py", "cognitive_json_act-all", cache_dir=args.transformers_cache)
            else:
                asset = load_dataset("create_hf_datasets/create_datasets.py", "cognitive_json_act-all")
        elif eval_dataset == 'turk':
            asset = None
        else:
            asset = None
    elif model_type == "baseline" or model_type == "t5_classification":
        if eval_dataset == 'asset':
            if args.transformers_cache is not None:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py", "all_json_act-none",
                                     cache_dir=args.transformers_cache)
            else:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py", "all_json_act-none")
        elif eval_dataset == 'cognitive':
            if args.transformers_cache is not None:
                asset = load_dataset("create_hf_datasets/create_datasets.py", "cognitive_json_act-none", cache_dir=args.transformers_cache)
            else:
                asset = load_dataset("create_hf_datasets/create_datasets.py", "cognitive_json_act-none")
        elif eval_dataset == 'turk':
            asset = None
        else:
            asset = None
    elif model_type == "single_action":
        if eval_dataset == 'asset':
            if args.transformers_cache is not None:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py", "all_json_act-single",
                                     cache_dir=args.transformers_cache)
            else:
                asset = load_dataset("create_hf_datasets/asset_with_actions.py", "all_json_act-single")
        elif eval_dataset == 'cognitive':
            if args.transformers_cache is not None:
                asset = load_dataset("create_hf_datasets/create_datasets.py", "cognitive_json_act-single", cache_dir=args.transformers_cache)
            else:
                asset = load_dataset("create_hf_datasets/create_datasets.py", "cognitive_json_act-single")
        elif eval_dataset == 'turk':
            asset = None
        else:
            asset = None
    else:
        raise ValueError("No argument type provided")

    if eval_dataset == 'cognitive':
        asset = asset.map(create_cognitive_like_asset)

    if add_t5_header:
        asset = asset.map(add_prefix)

    return asset


def get_example_random_act(example):
    if len(example["actions"]) > 0:
        example['source'] = f'<{random.choice(example["actions"])}> ' + example['source']
    return example


def get_example_random_two_act(example):
    order = ["PROX", "REPHRASE", "DEL", "ADD", "EXAMPLE", "EXPLAIN", "EXPLICIT", "REORDER", "SPLIT"]
    if len(example["actions"]) > 1:
        actions = random.sample(example["actions"], 2)
    elif len(example["actions"]) == 1:
        actions = [example["actions"][0], random.choice([act for act in order if act != example["actions"][0]])]
    else:
        return example
    actions_idx = [order.index(a) for a in actions]
    sorted_actions = [x for _, x in sorted(zip(actions_idx, actions))]
    example['source'] = f'<{sorted_actions[0]}> <{sorted_actions[1]}> ' + example['source']

    return example


"""
{
"source": f"{data['source']}",
"target": data['target'],
"actions": data['actions'],
"corpus": data['corpus'],
"entry_type": data['entry_type']
}
"""


def create_single_action_oracle(batch):
    return {
        "source": [f"<{a}> {source}" for i, source in enumerate(batch["source"]) for a in batch["actions"][i]],
        "target": [batch["target"][i] for i, source in enumerate(batch["source"]) for a in batch["actions"][i]],
        "actions": [batch["actions"][i] for i, source in enumerate(batch["source"]) for a in batch["actions"][i]],
        "corpus": [batch["corpus"][i] for i, source in enumerate(batch["source"]) for a in batch["actions"][i]],
        "entry_type": [batch["entry_type"][i] for i, source in enumerate(batch["source"]) for a in batch["actions"][i]],
    }


def create_double_action_oracle(batch):
    return {
        "source": [f"<{a}> <{b}> {source}" for i, source in enumerate(batch["source"]) for j, a in enumerate(batch["actions"][i]) for k, b in enumerate(batch["actions"][i]) if k > j],
        "target": [batch["target"][i] for i, source in enumerate(batch["source"]) for j, a in enumerate(batch["actions"][i]) for k, b in enumerate(batch["actions"][i]) if k > j],
        "actions": [batch["actions"][i] for i, source in enumerate(batch["source"]) for j, a in enumerate(batch["actions"][i]) for k, b in enumerate(batch["actions"][i]) if k > j],
        "corpus": [batch["corpus"][i] for i, source in enumerate(batch["source"]) for j, a in enumerate(batch["actions"][i]) for k, b in enumerate(batch["actions"][i]) if k > j],
        "entry_type": [batch["entry_type"][i] for i, source in enumerate(batch["source"]) for j, a in enumerate(batch["actions"][i]) for k, b in enumerate(batch["actions"][i]) if k > j],
    }


def create_single_action_det(batch):
    actions = ["PROX", "REPHRASE", "DEL", "ADD", "EXAMPLE", "EXPLAIN", "EXPLICIT", "REORDER", "SPLIT"]
    return {
        "source": [f"<{a}> {source}" for i, source in enumerate(batch["source"]) for a in actions],
        "target": [batch["target"][i] for i, source in enumerate(batch["source"]) for a in actions],
        "actions": [batch["actions"][i] for i, source in enumerate(batch["source"]) for a in actions],
        "corpus": [batch["corpus"][i] for i, source in enumerate(batch["source"]) for a in actions],
        "entry_type": [batch["entry_type"][i] for i, source in enumerate(batch["source"]) for a in actions],
    }


def create_double_action_det(batch):
    actions = ["PROX", "REPHRASE", "DEL", "ADD", "EXAMPLE", "EXPLAIN", "EXPLICIT", "REORDER", "SPLIT"]
    return {
        "source": [f"<{a}> <{b}> {source}" for i, source in enumerate(batch["source"]) for j, a in enumerate(actions) for k, b in enumerate(actions) if k > j],
        "target": [batch["target"][i] for i, source in enumerate(batch["source"]) for j, a in enumerate(actions) for k, b in enumerate(actions) if k > j],
        "actions": [batch["actions"][i] for i, source in enumerate(batch["source"]) for j, a in enumerate(actions) for k, b in enumerate(actions) if k > j],
        "corpus": [batch["corpus"][i] for i, source in enumerate(batch["source"]) for j, a in enumerate(actions) for k, b in enumerate(actions) if k > j],
        "entry_type": [batch["entry_type"][i] for i, source in enumerate(batch["source"]) for j, a in enumerate(actions) for k, b in enumerate(actions) if k > j],
    }


def evalution(model_path: str, args, metrics={}):
    set_seed(args.seed)  # set the seed for all the models the same
    model_name = model_path.split('/')[-1]
    logger.warning("Parsing model name")
    model_args = parse_model_name(model_name)
    _ADD_T5_HEADER = eval(model_args["add_t5_header"])
    _ABLATION = 0
    _GET_CLS = False
    _SARI_PATH = f"{args.save_sari_path}/{model_path.split('/')[-1]}.json" if len(args.save_sari_path) > 0 else None

    logger.warning(f"Loading Model Tokenizer {model_name}")
    if args.transformers_cache is not None:
        tokenizer = AutoTokenizer.from_pretrained(model_path,
                                                  cache_dir=args.transformers_cache)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)

    logger.warning(f"Loading Model {model_name}")
    if args.transformers_cache is not None:
        model_to_eval = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_config_dict,
                                                              cache_dir=args.transformers_cache)
    else:
        model_to_eval = AutoModelForSeq2SeqLM.from_pretrained(model_path, **model_config_dict)
    model_to_eval.to(device)

    def preprocess_function(examples):
        model_inputs = tokenizer(examples['source'], padding="max_length", truncation=True, max_length=512)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            refs = [t for t in examples['target']]
            if isinstance(refs[0], str):  # if has single target sentence
                labels = tokenizer(examples['target'], padding="max_length", truncation=True)
                if 't5' in tokenizer.name_or_path:  # T5 uses -100 as padding token. BART uses 1 as padding token
                    labels["input_ids"] = [np.where(np.array(label) != 0, np.array(label), -100).tolist() for label in
                                           labels["input_ids"]]
            elif isinstance(refs[0], list):  # if has multiple target senteces (ASSET, TURKCORPUS)
                tokenized = [[tokenizer(r, padding="max_length", truncation=True) for r in ref_group] for ref_group in
                             refs]
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

    logger.warning(f"Loading {args.eval_dataset} dataset")
    if args.no_add_info:
        eval_type = "no_add_info"
        eval_dataset = load_asset('baseline', args.eval_dataset, _ADD_T5_HEADER, cache_dir=args.transformers_cache)
    elif args.single_det:
        _ABLATION = 1
        eval_type = "single_det"
        eval_dataset = load_asset('baseline', args.eval_dataset, add_t5_header=False, cache_dir=args.transformers_cache)
        eval_dataset = DatasetDict({key: eval_dataset[key].map(create_single_action_det, batched=True,
                                                               remove_columns=eval_dataset[key].column_names)
                                    for key in eval_dataset.keys()
                                    }
                                   )
        if _ADD_T5_HEADER:
            eval_dataset = eval_dataset.map(add_prefix)
    elif args.single_oracle_rand:
        _ABLATION = 1
        eval_type = "single_oracle_rand"
        eval_dataset = load_asset('baseline', args.eval_dataset, add_t5_header=False, cache_dir=args.transformers_cache)
        eval_dataset = eval_dataset.map(get_example_random_act)
        if _ADD_T5_HEADER:
            eval_dataset = eval_dataset.map(add_prefix)
    elif args.single_oracle_det:
        _ABLATION = 1
        eval_type = "single_oracle_det"
        eval_dataset = load_asset('baseline', args.eval_dataset, add_t5_header=False, cache_dir=args.transformers_cache)
        eval_dataset = DatasetDict({key: eval_dataset[key].map(create_single_action_oracle, batched=True,
                                                               remove_columns=eval_dataset[key].column_names)
                                    for key in eval_dataset.keys()
                                    }
                                   )
        if _ADD_T5_HEADER:
            eval_dataset = eval_dataset.map(add_prefix)
    elif args.double_det:
        _ABLATION = 2
        eval_type = "double_det"
        eval_dataset = load_asset('baseline', args.eval_dataset, add_t5_header=False, cache_dir=args.transformers_cache)
        eval_dataset = DatasetDict({key: eval_dataset[key].map(create_double_action_det, batched=True,
                                                               remove_columns=eval_dataset[key].column_names)
                                    for key in eval_dataset.keys()
                                    }
                                   )
        if _ADD_T5_HEADER:
            eval_dataset = eval_dataset.map(add_prefix)
    elif args.double_oracle_rand:
        _ABLATION = 2
        eval_type = "double_oracle_rand"
        eval_dataset = load_asset('baseline', args.eval_dataset, add_t5_header=False, cache_dir=args.transformers_cache)
        eval_dataset = eval_dataset.map(get_example_random_two_act)
        if _ADD_T5_HEADER:
            eval_dataset = eval_dataset.map(add_prefix)
    elif args.double_oracle_det:
        _ABLATION = 2
        eval_type = "double_oracle_det"
        eval_dataset = load_asset('baseline', args.eval_dataset, add_t5_header=False, cache_dir=args.transformers_cache)
        eval_dataset = DatasetDict({key: eval_dataset[key].map(create_double_action_oracle, batched=True,
                                                               remove_columns=eval_dataset[key].column_names)
                                    for key in eval_dataset.keys()
                                    }
                                   )
        if _ADD_T5_HEADER:
            eval_dataset = eval_dataset.map(add_prefix)
    else:
        eval_type = "default"
        eval_dataset = load_asset(model_args["model_type"], args.eval_dataset, _ADD_T5_HEADER, cache_dir=args.transformers_cache)
        if model_args["model_type"] == "t5_classification":
            eval_dataset = eval_dataset.map(add_t5_mask)
            _GET_CLS = True
        if model_args["model_type"] == "bart_classification":
            eval_dataset = eval_dataset.map(add_bart_mask)
            _GET_CLS = True

    if args.trunc_dataset is not None:
        for k in eval_dataset.keys():
            eval_dataset[k] = eval_dataset[k].select(range(args.trunc_dataset))

    logger.warning(f"Tokenizing dataset")
    eval_ds_tokenized = eval_dataset.map(preprocess_function, batched=True)

    logger.warning(f"Evaluating on {args.eval_dataset}")
    if args.eval_dataset == "cognitive":
        num_annots = 1
    elif args.eval_dataset == "asset":
        num_annots = 10
    else:
        num_annots = 10
    evaluator = MultiRefEvaluator(eval_ds_tokenized, tokenizer, device=device, metrics=metrics, num_annots=num_annots,
                                  batch_size=eval_batch_size, add_t5_header=_ADD_T5_HEADER)
    baseline_sari_score, _ = evaluator.eval(model_to_eval, split_aligns=True, split_actions=True, get_baseline=True)
    sari_score, result = evaluator.eval(model_to_eval, split_aligns=True, split_actions=True, ablation=_ABLATION,
                                        return_classification=_GET_CLS, save_sari_args=_SARI_PATH)

    pct_identical = evaluator.get_pct_identical(result)

    logging.set_verbosity_info()
    logger.warning(f"#### Results for {model_name} ####")
    logger.warning(f"% Identical: {pct_identical}")
    logger.warning(f"Results DF:\n{evaluator.get_results_dataframe_string(sari_score)}")
    if args.get_full_result:
        with open(f"{results_folder}/{model_name}&eval_type={eval_type}&time={_DATE}.full_preds.txt", "w") as f:
            f.write(evaluator.get_results_dataframe_string(sari_score))
            evaluator.print_asset_results(f, model_name, sari_score, pct_identical, result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run evaluation of T5 model trained for cognitive simplification")
    parser.add_argument('--eval_dataset', type=str, default='asset', help="To run on the FestAbility Transcripts dataset, change this value to `cognitive`.")
    parser.add_argument('--transformers_cache', type=str)
    parser.add_argument('--results_folder', type=str, default='.')
    parser.add_argument('--get_full_result', action='store_true')
    parser.add_argument('--compute_bertscore', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    logging_severity_group = parser.add_mutually_exclusive_group()
    logging_severity_group.add_argument('--debug', action='store_true')
    logging_severity_group.add_argument('--info', action='store_true')
    model_loading_group = parser.add_mutually_exclusive_group(required=True)
    model_loading_group.add_argument('--model_load_folder', type=str)
    model_loading_group.add_argument('--model_to_eval', type=str)
    eval_args_group = parser.add_argument_group()
    eval_args_group.add_argument('--max_gen_len', type=int, default=200)
    eval_args_group.add_argument('--gen_early_stop', action='store_true')
    eval_args_group.add_argument('--gen_beams_num', type=int, default=4)
    eval_args_group.add_argument('--eval_batch_size', type=int, required=True)
    eval_args_group.add_argument('--trunc_dataset', type=int)
    eval_args_group.add_argument('--save_sari_path', type=str, default="")
    special_eval_group = eval_args_group.add_mutually_exclusive_group()
    special_eval_group.add_argument('--no_add_info', action='store_true', help="Flag to tell evaluator to run a baseline evaluation (no adding of actions)")
    special_eval_group.add_argument('--single_oracle_rand', action='store_true', help="Flag to tell the evaluator to add just 1 action randomly from the chosen actions")
    special_eval_group.add_argument('--single_oracle_det', action='store_true', help="Flag to tell the evaluator to add just 1 action deterministically from the chosen actions")
    special_eval_group.add_argument('--double_oracle_rand', action='store_true', help="Flag to tell the evaluator to add just 2 actions randomly from the chosen actions")
    special_eval_group.add_argument('--double_oracle_det', action='store_true', help="Flag to tell the evaluator to add just 2 actions deterministically from the chosen actions")
    special_eval_group.add_argument('--single_det', action='store_true', help="Flag to tell the evaluator to add just 1 action deterministically from the chosen actions")
    special_eval_group.add_argument('--double_det', action='store_true', help="Flag to tell the evaluator to add just 2 actions randomly from the chosen actions")

    args = parser.parse_args()

    logger = logging.get_logger(__name__)
    if args.debug:
        logging.set_verbosity_debug()
    elif args.info:
        logging.set_verbosity_info()

    logger.warning(args)
    logger.warning("Set Global Vars")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if is_torch_tpu_available():
        device = xm.xla_device()

    _COMPUTE_BERTSCORE = args.compute_bertscore
    eval_batch_size = args.eval_batch_size

    results_folder = args.results_folder

    set_seed(args.seed)

    _DATE = int(time.time())

    model_config_dict = {
        "early_stopping": args.gen_early_stop,
        "num_beams": args.gen_beams_num,
        "max_length": args.max_gen_len,
    }

    logger.warning("Loading Metrics")
    nltk.download('punkt')
    metrics = {}
    if args.transformers_cache is not None:
        sari = load_metric("sari", cache_dir=args.transformers_cache)
        if _COMPUTE_BERTSCORE:
            bertscore = load_metric("bertscore", cache_dir=args.transformers_cache)
            metrics["bertscore"] = bertscore

        meteor = load_metric("meteor", cache_dir=args.transformers_cache)
        bleu = load_metric("bleu", cache_dir=args.transformers_cache)
        # rouge = load_metric("rouge", cache_dir=args.transformers_cache)
        metrics["meteor"] = meteor
        metrics["bleu"] = bleu
        # metrics["rouge"] = rouge
    else:
        sari = load_metric("sari")
        if _COMPUTE_BERTSCORE:
            bertscore = load_metric("bertscore")
            metrics["bertscore"] = bertscore

        meteor = load_metric("meteor")
        bleu = load_metric("bleu")
        # rouge = load_metric("rouge")
        metrics["meteor"] = meteor
        metrics["bleu"] = bleu
        # metrics["rouge"] = rouge

    if args.model_load_folder is not None:
        main_folder = pathlib.Path(args.model_load_folder)
        assert main_folder.is_dir()
        for model_path in main_folder.iterdir():
            if model_path.is_dir() and '&' in model_path.stem:
                evalution(model_path.as_posix(), args, metrics)
                logger.warning("Clearing GPU memory")
                torch.cuda.empty_cache()
    else:  # we have args.model_to_eval
        evalution(args.model_to_eval, args, metrics)
