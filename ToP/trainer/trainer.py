# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import math
import os
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import Trainer
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_concat, nested_numpify
from transformers.trainer_utils import (PREFIX_CHECKPOINT_DIR, EvalPrediction,
                                        EvaluationStrategy, PredictionOutput,
                                        TrainOutput)
from transformers.utils import logging
from transformers.training_args import TrainingArguments

from args import AdditionalArguments
from utils.pruning_utils import *
from utils.utils import *
from transformers.trainer_pt_utils import nested_detach
import wandb
import datetime
from pytorchltr.loss import LambdaNDCGLoss1, PairwiseHingeLoss, LambdaNDCGLoss2, PairwiseLogisticLoss, LambdaARPLoss2
from models.l0_module import MAX_SEQUENCE_LENGTH

logger = logging.get_logger(__name__)

glue_tasks = {"cola": "matthews_correlation",
              "mnli": "accuracy",
              "mrpc": "f1",
              "sst2": "accuracy",
            #   "stsb": "corr",
              "stsb": "pearson",
              "qqp": "accuracy",
              "qnli": "accuracy",
              "rte": "accuracy",
              "sst2_aug": "accuracy",
              "rte_aug": "accuracy",
              "mrpc_aug": "accuracy",
              "qnli_aug": "accuracy",
            #   "stsb_aug": "corr",
                "stsb_aug": "pearson",
              "wnli": "accuracy",
              "20news": "accuracy",
              "imdb": "accuracy",
              "yelp": "accuracy",
              }
GLUE_TASK_FOR_NO_DISTILLATION = ["stsb"]

class Eval_Counter():
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_eval_score = 0
        self.near_sparsity_eval_times = 0
        self.level_best_score = {0.85: 0, 0.8: 0, 0.7: 0,
                                 0.6: 0, 0.75: 0, 0.9: 0, 0.95: 0, 0.65: 0}

    def round_nearest(self, x, a):
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))

    def update(self, epoch, global_step, eval_score):
        best_so_far = False
        if eval_score > self.best_eval_score:
            self.epoch = epoch
            self.global_step = global_step
            self.best_eval_score = eval_score
            best_so_far = True
        return best_so_far

    def clear(self):
        self.eval_score = 0


class ToPTrainer(Trainer):
    def __init__(
            self,
            model: PreTrainedModel = None,
            args: TrainingArguments = None,
            additional_args: AdditionalArguments = None,
            data_collator: Optional[DataCollator] = None,
            train_dataset: Optional[Dataset] = None,
            eval_dataset: Optional[Dataset] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            model_init: Callable[[], PreTrainedModel] = None,
            compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
            l0_module=None,
            teacher_model=None,
            **kwargs,
    ):

        Trainer.__init__(self, model, args, data_collator, train_dataset,
                         eval_dataset, tokenizer, model_init, compute_metrics, **kwargs)

        self.additional_args = additional_args
        self.best_eval_score_so_far = None
        self.best_step = None
        self.best_epoch = None

        self.l0_module = l0_module
        # FIXME: prepruning_finetune_steps
        self.prepruning_finetune_steps = 2
        self.start_prune = False

        self.l0_optimizer = None
        self.lagrangian_optimizer = None

        self.eval_counter = Eval_Counter()
        self.start_saving_best = True if self.additional_args.pruning_type is None else False

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.args.device)

        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)
        logger.setLevel(log_level)

    def create_optimizer_and_scheduler(self, num_training_steps: int, build_l0_optimizer:bool=True):
        def log_params(param_groups, des):
            for i, grouped_parameters in enumerate(param_groups):
                logger.info(
                    f"{des}, number of params: {sum(p.nelement() for p in grouped_parameters['params'])}, weight_decay: {grouped_parameters['weight_decay']}, lr: {grouped_parameters['lr']}")

        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            freeze_keywords = ["embeddings"]

            main_model_params = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and not any(fk in n for fk in freeze_keywords)],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate
                },
            ]
            log_params(main_model_params, "main params")
            self.optimizer = AdamW(
                main_model_params,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )

            if build_l0_optimizer and self.l0_module is not None:
                l0_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" not in n],
                    "weight_decay": 0.0,
                    "lr": self.additional_args.reg_learning_rate
                }]
                log_params(l0_params, "l0 reg params")
                self.l0_optimizer = AdamW(l0_params,
                                          betas=(self.args.adam_beta1,
                                                 self.args.adam_beta2),
                                          eps=self.args.adam_epsilon, )

                lagrangian_params = [{
                    "params": [p for n, p in self.l0_module.named_parameters() if "lambda" in n],
                    "weight_decay": 0.0,
                    "lr": -self.additional_args.reg_learning_rate
                }]
                log_params(lagrangian_params, "l0 reg lagrangian params")
                self.lagrangian_optimizer = AdamW(lagrangian_params,
                                                    betas=(self.args.adam_beta1,
                                                            self.args.adam_beta2),
                                                    eps=self.args.adam_epsilon)

        if self.lr_scheduler is None:
            if self.additional_args.scheduler_type == "linear":
                self.lr_scheduler = get_linear_schedule_with_warmup(
                    self.optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=num_training_steps
                )
            else:
                self.lr_scheduler = None

    def train(self):
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(
            train_dataloader) // self.args.gradient_accumulation_steps
        num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1) #! 12272

        if self.l0_module is not None:
            lagrangian_warmup_steps = self.additional_args.lagrangian_warmup_epochs * num_update_steps_per_epoch #! 24544
            # self.prepruning_finetune_steps = self.additional_args.prepruning_finetune_epochs * num_update_steps_per_epoch
            self.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)
            logger.info(f"Prepruning finetune steps: {self.prepruning_finetune_steps}")
            logger.info(f"Lagrangian warmup steps: {lagrangian_warmup_steps}")

        if self.args.max_steps > 0:
            self.t_total = self.args.max_steps
            num_train_epochs = self.args.max_steps // num_update_steps_per_epoch + int(
                self.args.max_steps % num_update_steps_per_epoch > 0
            )
        else:
            self.t_total = int(num_update_steps_per_epoch *
                               self.args.num_train_epochs)
            num_train_epochs = self.args.num_train_epochs
            self.args.max_steps = self.t_total

        self.create_optimizer_and_scheduler(num_training_steps=self.t_total, build_l0_optimizer = self.start_prune)

        model = self.model

        total_train_batch_size = (
            self.args.train_batch_size
            * self.args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
        )

        logger.info("***** Running training *****")

        logger.info("  Num examples = %d", self.num_examples(train_dataloader))
        logger.info("  Num Epochs = %d", num_train_epochs)
        logger.info("  Instantaneous batch size per device = %d",
                    self.args.per_device_train_batch_size)
        logger.info(
            "  Total train batch size (w. parallel, distributed & accumulation) = %d", total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d",
                    self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.t_total)

        self.state.global_step = 0
        self.epoch = 0
        self.total_flos = 0

        epochs_trained = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        reg_loss = torch.tensor(0.0).to(self.args.device)
        lag_loss = torch.tensor(0.0).to(self.args.device)
        score_loss = torch.tensor(0.0).to(self.args.device)

        logging_loss_scalar = 0.0
        logging_reg_loss_scalar = 0.0
        logging_lag_loss_scalar = 0.0
        logging_score_loss_scalar = 0.0

        model.zero_grad()
        if self.l0_module is not None:
            self.l0_module.zero_grad()

        self.optimizer.zero_grad()
        if self.l0_optimizer is not None:
            self.l0_optimizer.zero_grad()
        if self.lagrangian_optimizer is not None:
            self.lagrangian_optimizer.zero_grad()

        # disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        total_epoch_num = int(np.ceil(num_train_epochs)) - epochs_trained
        total_steps = total_epoch_num * len(train_dataloader)

        if self.additional_args.eval_only:
            self.start_prune = True
            self.evaluate()
            return

        self.evaluate()
        pbar = tqdm(
            total=total_steps,
            desc="Training Process",
            disable="device-aware-bert" in self.args.output_dir,
        )

        training_start_time = time.time()
        # training
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))): #! 20 epoch
            epoch_start = time.time()

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            self.eval_counter.clear()

            for step, inputs in enumerate(train_dataloader):
                pbar.update(1)
                if self.prepruning_finetune_steps > 0 and self.state.global_step == self.prepruning_finetune_steps and self.l0_module is not None: #! before pruning, run 12272 steps
                    self.start_prune = True

                    self.optimizer = None
                    self.lr_scheduler = None
                    lr_steps = self.t_total - self.state.global_step

                    # reset the optimizer
                    self.create_optimizer_and_scheduler(lr_steps, self.start_prune)
                    logger.info(f"Starting l0 regularization! using {type(self.l0_module)}")
                    print(f"Starting l0 regularization! using {type(self.l0_module)}, temperature: {self.l0_module.temperature:.2f}, init drop rate: {self.l0_module.droprate_init} token_loga shape: {list(self.l0_module.token_loga.shape)} prune location: {self.l0_module.token_prune_loc}")
                    print("NDCG TOPK=", self.additional_args.topk)

                if self.start_prune:
                    zs = self.l0_module.forward(training=True) #! get the zs
                    self.fill_inputs_with_zs(zs, inputs) #! use the zs

                loss_terms = self.training_step(model, inputs)
                tr_loss_step = loss_terms["loss"]
                lag_loss_step = loss_terms["lagrangian_loss"]
                score_loss_step = loss_terms["attention_score_distillation_loss"]

                tr_loss += tr_loss_step
                lag_loss += lag_loss_step if lag_loss_step is not None else 0.0
                score_loss += score_loss_step if score_loss_step is not None else 0.0

                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm)

                    self.optimizer.step()

                    if self.l0_module is not None and self.l0_optimizer is not None:
                        self.l0_optimizer.step()
                        self.lagrangian_optimizer.step()

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    if self.l0_module is not None:
                        self.l0_module.constrain_parameters()

                    model.zero_grad()
                    if self.l0_module is not None:
                        self.l0_module.zero_grad()
                    self.optimizer.zero_grad()
                    if self.l0_optimizer is not None:
                        self.l0_optimizer.zero_grad()
                    if self.lagrangian_optimizer is not None:
                        self.lagrangian_optimizer.zero_grad()

                    self.state.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.state.global_step % self.args.logging_steps == 0) or (
                            self.state.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        reg_loss_scalar = reg_loss.item()
                        lag_loss_scalar = lag_loss.item()
                        score_loss_scalar = score_loss.item()

                        logs["loss"] = (
                            tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["reg_loss"] = (
                            reg_loss_scalar - logging_reg_loss_scalar) / self.args.logging_steps
                        logs["lag_loss"] = (
                            lag_loss_scalar - logging_lag_loss_scalar) / self.args.logging_steps
                        logs["score_loss"] = (
                            score_loss_scalar - logging_score_loss_scalar) / self.args.logging_steps

                        # backward compatibility for pytorch schedulers
                        if self.lr_scheduler is not None:
                            lr = self.lr_scheduler.get_last_lr()[0] if version.parse(
                                torch.__version__) >= version.parse("1.4") else self.lr_scheduler.get_lr()[0]
                        else:
                            lr = self.args.learning_rate

                        logs["learning_rate"] = lr
                        logging_loss_scalar = tr_loss_scalar
                        logging_reg_loss_scalar = reg_loss_scalar
                        logging_lag_loss_scalar = lag_loss_scalar
                        logging_score_loss_scalar = score_loss_scalar

                        self.log(logs)

                    if self.state.global_step % self.args.eval_steps == 0:
                        self.evaluate()

                if self.args.max_steps > 0 and self.state.global_step >= self.args.max_steps:
                    break

            epoch_end = time.time()
            rate = (time.time() - training_start_time) / self.state.global_step
            remaining_time = rate * (total_steps - self.state.global_step)
            remaining_time = int(remaining_time)
            time_str = str(datetime.timedelta(seconds=remaining_time))
            print(f"ETA: {time_str} | Epoch {epoch} finished. Took {round(epoch_end - epoch_start, 2)} seconds.")

            if self.args.max_steps > 0 and self.state.global_step >= self.args.max_steps:
                break

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        # wandb.log({'global_step':self.state.global_step,'training_loss':tr_loss.item() / self.state.global_step})
        return TrainOutput(self.state.global_step, tr_loss.item() / self.state.global_step, None)

    def prediction_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None) -> PredictionOutput:
        prediction_loss_only = (
            prediction_loss_only if prediction_loss_only is not None else self.args.prediction_loss_only
        )

        # disable output hidden states and attention during evaluation
        self.model.config.output_hidden_states = False
        self.model.config.output_attentions = False

        model = self.model

        # multi-gpu eval
        model = self.model

        batch_size = dataloader.batch_size
        # logger.info("***** Running %s *****", description)
        # logger.info("  Num examples = %d", self.num_examples(dataloader))
        # logger.info("  Batch size = %d", batch_size)

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        model.eval()

        if self.args.past_index >= 0:
            self._past = None

        zs = None
        if self.start_prune:
            self.l0_module.eval()
            zs = self.l0_module.forward(training=False)

        if zs is not None:
            pruned_model_size_info = self.l0_module.calculate_model_size(zs, self.model.config.finetuning_task)

        disable_tqdm = "device-aware-bert" in self.args.output_dir
        expected_sparsities = []
        expected_sequence_sparsities = []
        target_sparsities = []
        for ii, inputs in enumerate(tqdm(dataloader, desc=description, disable=disable_tqdm)):
            if zs is not None:
                if ii == 0:
                    logger.info(f"Putting zs {zs.keys()} into inputs:")
                self.fill_inputs_with_zs(zs, inputs) #! use the zs
            loss, logits, labels = self.prediction_step(
                model, inputs, prediction_loss_only)

            if zs is not None:
                lag_loss, expected_sparsity, target_sparsity, expected_sequence_sparsity = self.l0_module.lagrangian_regularization(
                    self.state.global_step - self.prepruning_finetune_steps,
                    attention_mask=inputs["attention_mask"].cuda(),
                    token_score=zs["token_z"].cuda(),
                    pruner_score=zs["pruner_z"].cuda(),
                )
                expected_sparsities.append(expected_sparsity.item())
                expected_sequence_sparsities.append(expected_sequence_sparsity.item())
                target_sparsities.append(target_sparsity)

            batch_size = inputs[list(inputs.keys())[0]].shape[0]

            if logits is not None:
                preds_host = logits if preds_host is None else nested_concat(
                    preds_host, logits)
            if labels is not None:
                labels_host = labels if labels_host is None else nested_concat(
                    labels_host, labels)
            if loss is not None:
                if type(loss) == float:
                    losses = [loss] * batch_size
                    if losses_host is None:
                        losses_host = losses
                    else:
                        losses_host.extend(losses)
                else:
                    losses = loss.repeat(batch_size)
                    losses_host = losses if losses_host is None else torch.cat(
                        (losses_host, losses), dim=0)

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation.py loop
            delattr(self, "_past")

        if losses_host is not None:
            if not torch.is_tensor(losses_host):
                losses_host = torch.tensor(losses_host)
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate(
                (all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(
                all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(
                all_labels, labels, padding_index=-100)

        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(
                predictions=all_preds, label_ids=all_labels))
            metrics = {k: round(v, 4) for k, v in metrics.items()}
        else:
            metrics = {}

        if all_losses is not None and len(all_losses) > 0:
            metrics["eval_loss"] = round(np.mean(all_losses).item(), 4)

        if zs is not None:
            expected_sparsity = np.mean(expected_sparsities)
            expected_sequence_sparsity = np.mean(expected_sequence_sparsities)
            target_sparsity = np.mean(target_sparsities)
            expected_sparsity = round(expected_sparsity.item(), 5)
            expected_sequence_sparsity = round(expected_sequence_sparsity.item(), 5)
            metrics.update(pruned_model_size_info)
            metrics["expected_sparsity"] = round(expected_sparsity, 4)
            metrics["expected_sequence_sparsity"] = round(expected_sequence_sparsity, 4)
            metrics["target_sparsity"] = round(target_sparsity, 4)

            if (not self.start_saving_best) and (expected_sparsity - self.additional_args.target_sparsity >= -self.additional_args.sparsity_epsilon):
                self.start_saving_best = True
                print(f"Starting saving the best from epoch {int(self.epoch)} and step {self.state.global_step}")

        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True

        return PredictionOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics)


    def get_z_str(self, z):
        out_str = ""
        for zz in z:
            if zz:
                out_str += "1"
            else:
                out_str += "0"
        return out_str

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Tuple[Dict[str, float], List]:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        output = self.prediction_loop(
            eval_dataloader, description="Evaluation")

        name = glue_tasks[self.model.config.finetuning_task]
        output.metrics["step"] = self.state.global_step
        loggable_output_metrics = {k: v for k, v in output.metrics.items() if not isinstance(v, list)}
        self.log(loggable_output_metrics)
        print_keys = [
            name, "eval_loss",
            'macs_sparsity', "expected_sparsity", "expected_sequence_sparsity",
            'target_sparsity', 'step', "token_prune_loc"
        ]
        print("-" * 70)
        print("time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"Evaluating: {', '.join([f'{k}: {v}' for k, v in output.metrics.items() if k in print_keys])}")
        if self.l0_module is not None:
            print(f"lambda_1: {self.l0_module.lambda_1.item():.4f}, lambda_2: {self.l0_module.lambda_2.item():.4f} lambda_3: {self.l0_module.lambda_3.item():.4f}")
        if "token_ratio_for_training" in output.metrics:
            print("train remain:", np.round(output.metrics["token_ratio_for_training"].detach().cpu().numpy(), 2))
            print("infer remain:", output.metrics["token_ratio_for_inference"])
            print("layerwise remain:", output.metrics["remaining_token_ratio"])
            for z in output.metrics["token_z"]:
                print(self.get_z_str(z))
        if self.best_eval_score_so_far is not None:
            print(f"Best eval score so far: {self.best_eval_score_so_far:.4f} @ step {self.best_step} epoch {self.best_epoch:.2f}")
            self.log({"best_eval_score_so_far": self.best_eval_score_so_far, "best_step": self.best_step, "best_epoch": self.best_epoch})

        eval_score = 0
        if isinstance(name, str):
            if name in output.metrics:
                eval_score = output.metrics[name]
        else:
            for na in name:
                if na in output.metrics:
                    eval_score = output.metrics[na]
                    break

        if self.start_saving_best:
            best_so_far = self.eval_counter.update(
                self.epoch, self.state.global_step, eval_score)
            if best_so_far:
                best_dir = os.path.join(self.args.output_dir, "best")
                if not os.path.exists(best_dir):
                    os.makedirs(best_dir)
                info_text = f"Saving the best model so far: [Epoch {int(self.epoch)} | Step: {self.state.global_step} | MACs sparsity: {output.metrics['macs_sparsity'] if 'macs_sparsity' in output.metrics else 'Full' } | Score: {round(eval_score, 5)} | Loss: {round(output.metrics['eval_loss'], 5)}]"
                self.best_eval_score_so_far = eval_score
                self.best_epoch = self.epoch
                self.best_step = self.state.global_step
                logger.info(info_text)
                print(info_text)
                # not save model for saving storage space
                if "device-aware-bert" not in self.args.output_dir:
                    if self.l0_module is not None:
                        zs = self.l0_module.forward(training=False)
                        torch.save(zs, os.path.join(best_dir, "zs.pt"))
                        torch.save(self.l0_module, os.path.join(
                            best_dir, "l0_module.pt"))
                    self.model.save_pretrained(best_dir)

        return output.metrics

    def save_model(self, output_dir: Optional[str] = None):
        if "device-aware-bert" in self.args.output_dir:
            return
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        torch.save(self.l0_module, os.path.join(output_dir, "l0_module.pt"))

        if self.l0_module is not None:
            zs = self.l0_module.forward(training=False)
            torch.save(zs, os.path.join(output_dir, "zs.pt"))

        self.model.save_pretrained(output_dir)

    def calculate_layer_distillation_loss(self, teacher_outputs, student_outputs, zs, token_masks):
        if self.additional_args.do_layer_distill: #! only do layer distill
            raise NotImplementedError
            teacher_layer_output = teacher_outputs[2][1:] #! hidden states, with a length of 12. Every has a shape of [32, 65, 768]
            student_layer_output = student_outputs[2][1:]

            layer_loss = 0
            for layer_num, (t_layer_o, s_layer_o, token_mask) in enumerate(zip(teacher_layer_output, student_layer_output, token_masks)):
                if token_mask is None:
                    continue
                binary_mask = (token_mask > 0.0).float().detach().repeat(1, 1, s_layer_o.shape[-1])
                l = (((t_layer_o - s_layer_o) ** 2) * binary_mask).sum() / binary_mask.sum()
                layer_loss += l
            return layer_loss if layer_loss > 0 else None
        else:
            return None

    def calculate_distillation_loss(self, teacher_outputs, student_outputs, zs, token_masks):
        layer_loss = self.calculate_layer_distillation_loss(teacher_outputs, student_outputs, zs, token_masks)
        distill_loss = layer_loss
        ce_distill_loss = F.kl_div(
            input=F.log_softmax(
                student_outputs[1] / self.additional_args.distill_temp, dim=-1), #! logits: [32,3]
            target=F.softmax(
                teacher_outputs[1] / self.additional_args.distill_temp, dim=-1), #! distill_temp: 2.0
            reduction="batchmean") * (self.additional_args.distill_temp ** 2)

        if distill_loss is not None:
            raise NotImplementedError
            loss = self.additional_args.distill_ce_loss_alpha * ce_distill_loss
            loss += self.additional_args.distill_loss_alpha * distill_loss
        else:
            loss = ce_distill_loss

        return distill_loss, ce_distill_loss, loss

    def shortens_inputs(self, inputs):
        max_length = inputs["attention_mask"].sum(-1).max().item()
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]

    def calculate_attention_score_distillation_loss(
        self, target_attention_score, attention_scores, attention_mask, token_masks, TOPK=20,
    ):
        # score_distillation_loss_fn = None
        # score_distillation_loss_fn = PairwiseHingeLoss()
        # score_distillation_loss_fn = LambdaNDCGLoss2()
        # score_distillation_loss_fn = PairwiseLogisticLoss()
        score_distillation_loss_fn = LambdaARPLoss2()

        max_length = attention_mask.sum(-1).max().item()
        target_attention_score = target_attention_score[..., 0][:, :max_length].detach()
        attention_mask = attention_mask[:, :max_length]
        n = attention_mask.sum(-1).detach().long()

        loss = 0
        for i in range(1, len(attention_scores) // 3):
            if isinstance(score_distillation_loss_fn, LambdaARPLoss2):
                # if token_masks[i] is not None:
                #     target = target_attention_score * token_masks[i]
                # else:
                #     target = target_attention_score
                target = target_attention_score
                target_token_index = torch.argsort(target, dim=1, descending=True)
                target_token_rank = torch.argsort(target_token_index, dim=1)
                # if token_masks[i] is not None:
                #     attention_mask = token_masks[i] * 1.0
                target_token_rank = (attention_mask.sum(-1).unsqueeze(-1) - target_token_rank - 1)
                # target_token_rank *= attention_mask
                # relevance = target_token_rank.detach().long()
                # scores = attention_scores[i][:, :max_length, 0]
                # scores *= attention_mask
                ########################## topk ARP ############################
                target_token_rank += 1
                relevance = target_token_rank.detach().long()
                scores = attention_scores[i][:, :max_length, 0]
                # scores *= attention_mask
                topk = min(TOPK, max_length)
                threshold = torch.topk(relevance, k=topk, dim=1, largest=True)[0][:, -1].unsqueeze(-1)
                selection = relevance >= threshold
                relevance = relevance[selection].reshape(-1, topk).clamp_min(0)
                scores = scores[selection].reshape(-1, topk)
                scores = torch.where(
                    relevance == 0,
                    torch.zeros_like(scores),
                    scores,
                )
                relevance = relevance - relevance.min(-1, keepdim=True)[0]
                n = torch.ones(relevance.shape[0], device=relevance.device, dtype=torch.long) * relevance.shape[1]
                mask = (relevance == 0).sum(-1) == 1
                relevance[mask] += 1
                ###############################################################
                losses = score_distillation_loss_fn(scores, relevance, n)
                losses /= n**2
                loss += losses.mean()
            elif isinstance(score_distillation_loss_fn, LambdaNDCGLoss2):
                target = target_attention_score
                target_token_index = torch.argsort(target, dim=1, descending=True)
                target_token_rank = torch.argsort(target_token_index, dim=1)
                target_token_rank = (attention_mask.sum(-1).unsqueeze(-1) - target_token_rank - 1)
                target_token_rank += 1
                relevance = target_token_rank.detach().long()
                scores = attention_scores[i][:, :max_length, 0]
                topk = min(TOPK, max_length)
                threshold = torch.topk(relevance, k=topk, dim=1, largest=True)[0][:, -1].unsqueeze(-1)
                selection = relevance >= threshold
                relevance = relevance[selection].reshape(-1, topk).clamp_min(0)
                scores = scores[selection].reshape(-1, topk)
                scores = torch.where(
                    relevance == 0,
                    torch.zeros_like(scores),
                    scores,
                )
                relevance = relevance - relevance.min(-1, keepdim=True)[0]
                n = torch.ones(relevance.shape[0], device=relevance.device, dtype=torch.long) * relevance.shape[1]
                mask = (relevance == 0).sum(-1) == 1
                relevance[mask] += 1
                losses = score_distillation_loss_fn(scores, relevance, n)
                loss += losses.mean()
            else:
                raise NotImplementedError
        return loss

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
        model.train()
        if self.l0_module is not None:
            self.l0_module.train()
        inputs = self._prepare_inputs(inputs)

        distill_loss = None
        distill_ce_loss = None

        if self.teacher_model is not None:
            with torch.no_grad():
                # only retain inputs of certain keys
                teacher_inputs_keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels",
                                        "output_attentions", "output_hidden_states", "return_dict"]
                teacher_inputs = {key: inputs[key]
                                    for key in teacher_inputs_keys if key in inputs}
                self.shortens_inputs(teacher_inputs)
                teacher_outputs = self.teacher_model(**teacher_inputs)
        
        if self.teacher_model is None:
            loss = self.compute_loss(model, inputs)
        elif self.model.config.finetuning_task not in GLUE_TASK_FOR_NO_DISTILLATION:
            self.shortens_inputs(inputs)
            student_outputs = model(**inputs) #! get the two outputs
            zs = {key: inputs[key] for key in inputs if "_z" in key} #! extract the zs
            distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                teacher_outputs, student_outputs, zs, model.bert.encoder.masks)
        else:
            self.shortens_inputs(inputs)
            student_outputs = model(**inputs) #! get the two outputs
            loss = self.compute_loss(model, inputs)
            # loss = F.kl_div(
            #     input=F.log_softmax(
            #         model.pooled_output / self.additional_args.distill_temp, dim=-1), #! logits: [32,3]
            #     target=F.softmax(
            #         self.teacher_model.pooled_output / self.additional_args.distill_temp, dim=-1), #! distill_temp: 2.0
            #     reduction="batchmean") * (self.additional_args.distill_temp ** 2)

        lagrangian_loss = None
        mask_loss = None
        attention_score_distillation_loss = None
        if self.start_prune:
            lagrangian_loss, _, _, _ = \
                self.l0_module.lagrangian_regularization(
                    self.state.global_step - self.prepruning_finetune_steps,
                    attention_mask=inputs["attention_mask"],
                )
            debug_loss = loss.item()
            debug_lagrangian_loss = lagrangian_loss.item()
            loss += lagrangian_loss
            #################### MSE/KL distillation loss ######################
            # target_attention_score = self.teacher_model.bert.encoder.last_pred_score
            # attention_scores=model.bert.encoder.pred_scores
            # # kl_loss = nn.KLDivLoss(reduction='batchmean')
            # mse_loss = nn.MSELoss(reduction='mean')
            # attention_score_distillation_loss = 0
            # for i in range(1, len(attention_scores) // 3):
            #     # attention_score_distillation_loss += kl_loss(
            #     #     F.log_softmax(attention_scores[i][..., 0], dim=-1),
            #     #     F.softmax(target_attention_score[..., 0], dim=-1),
            #     # )
            #     attention_score_distillation_loss += mse_loss(
            #         attention_scores[i][..., 0],
            #         target_attention_score[..., 0],
            #     )
            # warmup_progress = self.l0_module.get_warmup_progress(
            #     self.state.global_step - self.prepruning_finetune_steps,
            # )
            # loss += attention_score_distillation_loss * max(0.01, 1.0 - warmup_progress)
            ################################################################
            #################### rank distillation loss ####################
            target_attention_score = self.teacher_model.bert.encoder.last_pred_score
            # target_attention_score = self.teacher_model.bert.encoder.pred_scores[11]
            attention_score_distillation_loss = self.calculate_attention_score_distillation_loss(
                target_attention_score=target_attention_score,
                attention_scores=model.bert.encoder.pred_scores,
                attention_mask=inputs["attention_mask"][..., 1:],
                token_masks=model.bert.encoder.masks,
                TOPK=self.additional_args.topk,
            )
            warmup_progress = self.l0_module.get_warmup_progress(
                self.state.global_step - self.prepruning_finetune_steps,
            )
            rank_loss_lambda = self.additional_args.distill_ce_loss_alpha * max(0.01, 1.0 - warmup_progress)
            # rank_loss_lambda = self.additional_args.distill_ce_loss_alpha
            debug_attention_score_distillation_loss = attention_score_distillation_loss.item() * rank_loss_lambda
            loss += attention_score_distillation_loss * rank_loss_lambda
            if self.state.global_step % (self.args.eval_steps // 2) == 0:
                print(f"loss: {debug_loss:.6f}, lagrangian_loss: {debug_lagrangian_loss:.6f}, attention_score_distillation_loss: {debug_attention_score_distillation_loss:.6f}")
            ################################################################

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()

        return {
            "loss": loss.detach(),
            "lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
            "distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
            "distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None,
            "mask_loss": mask_loss.detach() if mask_loss is not None else None,
            "attention_score_distillation_loss": attention_score_distillation_loss.detach() if attention_score_distillation_loss is not None else None,
        }

    def fill_inputs_with_zs(self, zs, inputs):
        for key in zs:
            inputs[key] = zs[key]
