import json

import torch
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.trainer_pt_utils import nested_concat
import pandas as pd
from typing import Union, Any, TextIO
from datasets import Dataset, Metric
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader, default_collate
from easse.sari import corpus_sari, get_corpus_sari_operation_scores
from transformers import (
    PreTrainedTokenizer, AutoTokenizer, AutoModel, PreTrainedModel, T5Config)
from transformers.trainer import Trainer

from datasets import load_dataset
import torch
import nltk
import copy
import numpy as np
from nltk.translate import meteor_score
from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset, load_metric
from datasets.dataset_dict import DatasetDict
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Stack
from transformers import (
    AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq,  # Seq2SeqTrainer,
    Seq2SeqTrainingArguments, T5Tokenizer, AutoTokenizer, BatchEncoding,
    PreTrainedTokenizer, AutoModel, PreTrainedModel, T5EncoderModel
)
from transformers.deepspeed import deepspeed_init
from collections import Counter

# for Sari trainer object
import collections
import inspect
import math
import os
import random
import re
import shutil
import sys
import time
import warnings
from logging import StreamHandler
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, IterableDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from transformers.file_utils import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    PushToHubMixin,
    is_apex_available,
    is_datasets_available,
    is_in_notebook,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_training_run_on_sagemaker,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    get_last_checkpoint,
    number_of_arguments,
    set_seed,
    speed_metrics,
)
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

logger = logging.get_logger(__name__)


class SariSeq2SeqTrainer(Seq2SeqTrainer):

    def evaluation_loop(  # taken from huggingface
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by :obj:`Trainer.evaluate()` and :obj:`Trainer.predict()`.

        Works both with or without labels.
        """
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # if eval is called w/o train init deepspeed here
        if self.args.deepspeed and not self.deepspeed:
            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(self, num_training_steps=0, resume_from_checkpoint=None)
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            # XXX: we don't need optim/sched for inference, but this needs to be sorted out, since
            # for example the Z3-optimizer is a must for zero3 to work even for inference - what we
            # don't need is the deepspeed basic optimizer which is self.optimizer.optimizer
            deepspeed_engine.optimizer.optimizer = None
            deepspeed_engine.lr_scheduler = None

        model = self._wrap_model(self.model, training=False)

        # if full fp16 is wanted on eval and this ``evaluation`` or ``predict`` isn't called while
        # ``train`` is running, halve it first and then put on device
        if not self.is_in_train and self.args.fp16_full_eval:
            model = model.half().to(self.args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if isinstance(dataloader.dataset, collections.abc.Sized):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if is_torch_tpu_available():
            dataloader = pl.ParallelLoader(dataloader, [self.args.device]).per_device_loader(self.args.device)

        if self.args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs is not None:
                inputs = self._pad_across_processes(inputs["input_ids"])
                inputs = self._nested_gather(inputs)
                inputs_host = inputs if inputs_host is None else nested_concat(inputs_host, inputs, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(self.args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if self.args.eval_accumulation_steps is not None and (step + 1) % self.args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )
                if inputs_host is not None:
                    inputs = nested_numpify(inputs_host)
                    all_inputs = inputs if all_inputs is None else nested_concat(all_inputs, inputs, padding_index=-100)

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host, inputs_host = None, None, None, None

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
        if inputs_host is not None:
            inputs = nested_numpify(inputs_host)
            all_inputs = inputs if all_inputs is None else nested_concat(all_inputs, inputs, padding_index=-100)

        # Number of samples
        if not isinstance(eval_dataset, IterableDataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(sources=all_inputs, predictions=all_preds, references=all_labels)
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


class MultiRefEvaluator:
    def __init__(self,
                 dataset: Dataset, tokenizer: Union[PreTrainedTokenizer, AutoTokenizer],
                 device: str, batch_size: int, num_annots=10, add_t5_header=False, metrics=dict()):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        self.batch_size = batch_size
        self.num_annots = num_annots
        self.add_t5_header = add_t5_header
        self.device = device
        self.metrics = metrics

    def get_pct_identical(self, result):
        count = 0
        total = 0
        # total = len(result.values())
        for orig, sys_sent, _, _, _ in zip(*result.values()):
            total += 1
            if self.add_t5_header:
                if orig[len('simplify:'):].strip() == sys_sent.strip():
                    count += 1
            else:
                if orig.strip() == sys_sent.strip():
                    count += 1
        return count / total

    def print_asset_results(self, file: TextIO, name: str, score, pct_ident, result):
        file.write(f"\nEvaluation for {name}\n")
        file.write(f"Sari: {score}\n")
        file.write(f"% Identitcal: {pct_ident}\n")
        for orig, sys_sent, actions, refs_sent, cls_preds in zip(*result.values()):
            o = orig[len('simplify:'):].strip() if self.add_t5_header else orig.strip()
            if o != sys_sent.strip():
                refs = [f"\n\t{r}" for r in refs_sent] if isinstance(refs_sent, list) else refs_sent
                file.write(f"Source:\n\t{o}"
                           f"\nActions:\n\t{actions}\nClassification:\n\t{cls_preds}"
                           f"\nPrediction:\n\t{sys_sent}"
                           f"\nReferences:{''.join(refs)}\n")

    def get_results_dataframe_string(self, result):
        return pd.DataFrame.from_dict(result, orient='index').to_string()

    def calculate_other_metrics(self, tokenized_preds, tokenized_labels, decoded_preds, decoded_labels):
        result = {}
        # BLEU expects each prediction to be a tokenized sentence and each reference to be a list of tokenized sentences
        if 'bleu' in self.metrics:
            bleu_result = self.metrics['bleu'].compute(predictions=tokenized_preds, references=tokenized_labels)
            for i, precision in enumerate(bleu_result['precisions']):
                bleu_result[f'{i + 1}-gram_precision'] = precision
            bleu_result.pop("precisions")
            result = {**result, **bleu_result}

        if 'meteor' in self.metrics:
            meteor_result = self.metrics['meteor'].compute(predictions=[p for p in decoded_preds for i in range(self.num_annots)],
                                                           references=[l for refs in decoded_labels for l in refs])
            result = {**result, **meteor_result}

        if 'rouge' in self.metrics:
            rouge_result = self.metrics['rouge'].compute(predictions=decoded_preds, references=decoded_labels)
            result = {**result, **rouge_result}
        return result

    def eval(self, model: Union[PreTrainedModel, AutoModel],
             split='test', dataset_size=None, get_baseline=False,
             split_actions=False, split_aligns=False, ablation=0,
             return_preds=True, return_classification=False, save_sari_args=None) -> Tuple[Dict, Dict]:
        if dataset_size is None:
            data = self.dataset[split]
        else:
            data = self.dataset[split].select(range(dataset_size))

        cls_preds_detoked = None  # for return default - will be changed when relevant

        if get_baseline:
            preds_detoked = self.tokenizer.batch_decode(data['input_ids'], skip_special_tokens=True)
            if self.add_t5_header:
                preds_detoked = [pred[len("simplify:"):].strip() for pred in preds_detoked]
        else:
            dataloader = torch.utils.data.DataLoader(data, batch_size=self.batch_size)

            preds = None
            with torch.no_grad():
                for b in tqdm(iter(dataloader)):
                    b['attention_mask'] = b['attention_mask'].to(self.device)
                    b['input_ids'] = b['input_ids'].to(self.device)
                    pred = model.generate(**b)
                    preds = nested_concat(preds, pred) if preds is not None else pred
                preds = preds.detach().cpu().numpy()
                if return_classification:
                    idxs = np.isin(preds, self.tokenizer.additional_special_tokens_ids)
                    cls_preds = [preds[i, idxs[i]].tolist() for i in range(len(idxs))]
                    cls_preds_detoked = self.tokenizer.batch_decode(cls_preds)
                    cls_preds_detoked = [self.tokenizer.tokenize(cls_p) for cls_p in cls_preds_detoked]
                preds = np.where(preds != -100, preds, self.tokenizer.pad_token_id)
                preds_detoked = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        sources = self.tokenizer.batch_decode(data['input_ids'], skip_special_tokens=True)
        targets = [self.tokenizer.batch_decode(rfs, skip_special_tokens=True) for rfs in data['labels']]
        if self.add_t5_header:
            sources = [source[len("simplify:"):].strip() for source in sources]

        tokenized_preds = [nltk.word_tokenize(pred.strip()) for pred in preds_detoked]
        tokenized_labels = [[nltk.word_tokenize(label.strip()) for label in refs] for refs in targets]
        tokenized_sources = [nltk.word_tokenize(pred.strip()) for pred in sources]
        decoded_preds = [" ".join(pred) for pred in tokenized_preds]
        decoded_labels = [[" ".join(label) for label in refs] for refs in tokenized_labels]
        decoded_sources = [" ".join(source) for source in tokenized_sources]

        other_result = {}
        if len(self.metrics) > 0:
            other_result = self.calculate_other_metrics(tokenized_preds, tokenized_labels, decoded_preds, decoded_labels)


        sari_args = {
            "orig_sents": decoded_sources,  # (len = n_samples)
            "sys_sents": decoded_preds,  # (len = n_samples)
            "refs_sents": [[target[i] for target in decoded_labels] for i in range(self.num_annots)],  # (shape = (n_references, n_samples))
        }

        if save_sari_args is not None:
            with open(save_sari_args, "w") as o:
                o.write(json.dumps(sari_args))

        def return_result(sari_args, return_preds):
            add_score, keep_score, del_score = get_corpus_sari_operation_scores(**sari_args)
            if return_preds:
                return ((add_score + keep_score + del_score) / 3,
                        add_score, keep_score, del_score,
                        {"orig_sents": decoded_sources, "sys_sents": preds_detoked, "refs_sents": targets})
            else:
                return ((add_score + keep_score + del_score) / 3,
                        add_score, keep_score, del_score,
                        {})

        sari_score, add_score, keep_score, del_score, _ = return_result(sari_args, False)

        result = {
            "whole_corpus": {"sari_score": sari_score, "add_score": add_score,
                             "keep_score": keep_score, "del_score": del_score, **other_result}
        }

        if split_actions:
            all_acts = Counter()
            for e in data['actions']:
                all_acts.update(e)
            all_acts = sorted(all_acts.keys())
            all_acts = {a: [] for a in all_acts}
            for i, e in enumerate(data['actions']):
                for act in e:
                    all_acts[act].append(i)
            for act, idxs in all_acts.items():
                filtered_targets = [targets for i, targets in enumerate(decoded_labels) if i in idxs]
                filtered_tokenized_preds = [pred for i, pred in enumerate(tokenized_preds) if i in idxs]
                filtered_tokenized_targets = [targets for i, targets in enumerate(tokenized_labels) if i in idxs]
                updated_args = {
                    "orig_sents": list(np.array(sari_args["orig_sents"])[idxs]),  # (len = n_samples)
                    "sys_sents": list(np.array(sari_args["sys_sents"])[idxs]),  # (len = n_samples)
                    "refs_sents": [[target[i] for target in filtered_targets] for i in range(self.num_annots)]  # (shape = (n_references, n_samples))
                }
                sari_score, add_score, keep_score, del_score, _ = return_result(updated_args, False)
                split_result = {}
                if len(self.metrics) > 0:
                    split_result = self.calculate_other_metrics(filtered_tokenized_preds, filtered_tokenized_targets, updated_args["sys_sents"], filtered_targets)
                result[act] = {"sari_score": sari_score, "add_score": add_score,
                               "keep_score": keep_score, "del_score": del_score, **split_result}
        if split_aligns:
            all_aligns = sorted(Counter(data['entry_type']).keys())
            all_aligns = {a: [] for a in all_aligns}
            for i, e in enumerate(data['entry_type']):
                all_aligns[e].append(i)
            for align, idxs in all_aligns.items():
                filtered_targets = [targets for i, targets in enumerate(decoded_labels) if i in idxs]
                filtered_tokenized_preds = [pred for i, pred in enumerate(tokenized_preds) if i in idxs]
                filtered_tokenized_targets = [targets for i, targets in enumerate(tokenized_labels) if i in idxs]
                updated_args = {
                    "orig_sents": list(np.array(sari_args["orig_sents"])[idxs]),  # (len = n_samples)
                    "sys_sents": list(np.array(sari_args["sys_sents"])[idxs]),  # (len = n_samples)
                    "refs_sents": [[target[i] for target in filtered_targets] for i in range(self.num_annots)]  # (shape = (n_references, n_samples))
                }
                sari_score, add_score, keep_score, del_score, _ = return_result(updated_args, False)
                split_result = {}
                if len(self.metrics) > 0:
                    split_result = self.calculate_other_metrics(filtered_tokenized_preds, filtered_tokenized_targets,
                                                                updated_args["sys_sents"], filtered_targets)
                result[align] = {"sari_score": sari_score, "add_score": add_score,
                                 "keep_score": keep_score, "del_score": del_score, **split_result}

        if ablation > 0:
            regex = re.compile(r"^<([^>]+)> [^<]") if ablation == 1 else re.compile(r"^<([^>]+)> <([^>]+)> [^<]")
            matches = [(i, regex.match(source)) for i, source in enumerate(data["source"])]
            all_groups = sorted(Counter([f'abl::{"+".join(m.groups())}' for _, m in matches if m is not None]))
            all_groups = {a: [] for a in all_groups}
            for i, m in matches:
                if m is not None:
                    all_groups[f'abl::{"+".join(m.groups())}'].append(i)
            for group, idxs in all_groups.items():
                filtered_targets = [targets for i, targets in enumerate(decoded_labels) if i in idxs]
                filtered_tokenized_preds = [pred for i, pred in enumerate(tokenized_preds) if i in idxs]
                filtered_tokenized_targets = [targets for i, targets in enumerate(tokenized_labels) if i in idxs]
                updated_args = {
                    "orig_sents": list(np.array(sari_args["orig_sents"])[idxs]),  # (len = n_samples)
                    "sys_sents": list(np.array(sari_args["sys_sents"])[idxs]),  # (len = n_samples)
                    "refs_sents": [[target[i] for target in filtered_targets] for i in range(self.num_annots)]  # (shape = (n_references, n_samples))
                }
                sari_score, add_score, keep_score, del_score, _ = return_result(updated_args, False)
                split_result = {}
                if len(self.metrics) > 0:
                    split_result = self.calculate_other_metrics(filtered_tokenized_preds, filtered_tokenized_targets,
                                                                updated_args["sys_sents"], filtered_targets)
                result[group] = {"sari_score": sari_score, "add_score": add_score,
                                 "keep_score": keep_score, "del_score": del_score, **split_result}

        if cls_preds_detoked is None:
            cls_preds_detoked = [[] for s in sources]

        if return_preds:
            return result, {"orig_sents": sources, "sys_sents": preds_detoked, "actions": data["actions"], "refs_sents": targets, "cls_preds": cls_preds_detoked}
        else:
            return result, {}


class T5CustomClassifier(torch.nn.Module):
    def __init__(self, config: dict):
        super(T5CustomClassifier, self).__init__()
        self.encoder = T5EncoderModel.from_pretrained(config['model_type'])
        self.dropout_rate = config['dropout']
        self.num_labels = config['num_labels']
        self.dropout1 = torch.nn.Dropout(self.dropout_rate)
        self.classifier1 = torch.nn.Linear(self.encoder.config.d_model, 1)
        self.dropout2 = torch.nn.Dropout(self.dropout_rate)
        self.classifier2 = torch.nn.Linear(self.encoder.config.d_model, self.num_labels)

    def forward(self, input_ids=None, attention_mask=None, head_mask=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None, return_dict=None,
                ):
        return_dict = return_dict if return_dict is not None else self.encoder.config.use_return_dict

        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                                       inputs_embeds=inputs_embeds, head_mask=head_mask,
                                       output_attentions=output_attentions, output_hidden_states=output_hidden_states,
                                       return_dict=return_dict,
                                       )

        pooled_output = encoder_outputs[0]

        pooled_output = self.dropout1(pooled_output)
        pooled_output = self.classifier1(pooled_output)
        pooled_output = self.dropout2(pooled_output.squeeze(-1))
        logits = self.classifier2(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                problem_type = "regression"
            elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                problem_type = "single_label_classification"
            else:
                problem_type = "multi_label_classification"
            if problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

            if not return_dict:
                output = (logits,) + encoder_outputs[1:]
                return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class MultiLabelTrainer(Trainer):
    """ Taken from huggingface.co examples """
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        try:
            loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                            labels.float().view(-1, self.model.config.num_labels))
        except AttributeError:
            loss = loss_fct(logits.view(-1, self.model.num_labels),
                            labels.float().view(-1, self.model.num_labels))
        return (loss, outputs) if return_outputs else loss
