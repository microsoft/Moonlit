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
from packaging import version
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange
from transformers import Trainer,default_data_collator
from transformers.data.data_collator import DataCollator
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer import Trainer
from transformers.trainer_pt_utils import nested_concat, nested_numpify, nested_truncate, IterableDatasetShard
from transformers.trainer_utils import (EvalPrediction, EvalLoopOutput, TrainOutput,seed_worker,
                                        has_length, speed_metrics, denumpify_detensorize)
from transformers.utils import logging
from transformers.training_args import TrainingArguments
from utils.deepspeed_utils import is_deepspeed_zero3_enabled, deepspeed_init
from args import AdditionalArguments
import random
import datasets
from transformers.utils import is_datasets_available
import deepspeed
import utils.lora_utils as lora
from transformers.trainer_callback import TrainerState
import mlflow
import torch.nn as nn
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
mlflow.autolog()

logger = logging.get_logger(__name__)

TRAINER_STATE_NAME = "trainer_state.json"


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


class CEloss_Counter():
    def __init__(self):
        self.epoch = 0
        self.global_step = 0
        self.best_ce_loss = 10e4
        # self.near_sparsity_eval_times = 0
        # self.level_best_score = {0.85: 0, 0.8: 0, 0.7: 0,
        #                          0.6: 0, 0.75: 0, 0.9: 0, 0.95: 0, 0.65: 0}

    def round_nearest(self, x, a):
        return round(round(x / a) * a, -int(math.floor(math.log10(a))))

    def update(self, epoch, global_step, ce_loss):
        best_so_far = False
        if ce_loss < self.best_ce_loss:
            self.epoch = epoch
            self.global_step = global_step
            self.best_ce_loss = ce_loss
            best_so_far = True
        return best_so_far

    def clear(self):
        self.eval_score = 10e4


class CompressoTrainer(Trainer):
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
            use_lora: bool = False,
            lora_train_bias: str = "none",
            **kwargs,
    ):

        Trainer.__init__(self, model, args, data_collator, train_dataset,
                         eval_dataset, tokenizer, model_init, compute_metrics, **kwargs)

        self.additional_args = additional_args
        
        self.l0_module = l0_module
        self.prepruning_finetune_steps = 0
        self.start_prune = False
        self.zs = None
        self.l0_optimizer = None
        self.lagrangian_optimizer = None
        self.global_step = 0
        self.eval_counter = Eval_Counter()
        self.celoss_counter = CEloss_Counter()
        self.start_saving_best = False # if self.additional_args.pruning_type is None else False

        self.teacher_model = teacher_model
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.args.device)

        log_level = args.get_process_log_level()
        logging.set_verbosity(log_level)
        logger.setLevel(log_level)
        self.use_lora = use_lora
        self.lora_train_bias = lora_train_bias
        if self.use_lora:
            logger.info("LoRA enabled.")

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

        return

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
        data_collator = default_data_collator#self.data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def train(self,resume_from_checkpoint):
        teacher_model =None
        self._memory_tracker.start()
        args = self.args
        train_dataloader = self.get_train_dataloader()
        trial=None
        self._hp_search_setup(trial)
        self.is_in_train = True
        self._train_batch_size = self.args.train_batch_size

        import datetime
        now = datetime.datetime.now()
        self.args.output_dir = os.path.join(self.args.output_dir, 'Compresso-{}-s{}-lr{}-reglr{}-warmup{}/{}-{}-{}-{}-{}'.format(
            self.additional_args.task_name,
            self.additional_args.target_sparsity*100,
            self.args.learning_rate,
            self.additional_args.reg_learning_rate,
            self.additional_args.lagrangian_warmup_epochs,
            now.year, now.month, now.day, now.hour, now.minute
        ))
        logger.info(f"Output dir: {self.args.output_dir}")
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        logger.info("building folder finish")

        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
            num_train_epochs = math.ceil(args.num_train_epochs)
        
        #add deepspeed https://huggingface.co/docs/accelerate/usage_guides/deepspeed
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None
        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        model = self._wrap_model(self.model_wrapped)
        self.model_wrapped = model

        # teacher model for distillation
        if self.additional_args.do_distill:
            model_parameters = list(filter(lambda p: p.requires_grad, self.teacher_model.parameters()))
            kwargs = dict(
            model=self.teacher_model,
            model_parameters=model_parameters,
            config_params=self.args.hf_deepspeed_config.config,
            optimizer=None,
            lr_scheduler=None,
            )
            deepspeed_engine, _, _, _ = deepspeed.initialize(**kwargs)
            self.teacher_model = deepspeed_engine.module
            self.teacher_model_wrapped = deepspeed_engine
            teacher_model = self._wrap_model(self.teacher_model_wrapped)
            self.teacher_model_wrapped = teacher_model
            #teacher_model.layer_transformation_bool=True

        # the value of self.prepruning_finetune_steps is zero if finetune
        if self.additional_args.pretrained_pruned_model is None:
            self.prepruning_finetune_steps = len_dataloader * self.additional_args.prepruning_finetune_epochs
        if self.l0_module is not None:
            lagrangian_warmup_steps = self.additional_args.lagrangian_warmup_epochs * num_update_steps_per_epoch #! 24544
            self.l0_module.set_lagrangian_warmup_steps(lagrangian_warmup_steps)
            logger.info(f"Prepruning finetune steps: {self.prepruning_finetune_steps}")
            logger.info(f"Lagrangian warmup steps: {lagrangian_warmup_steps}")

        self.t_total = self.args.max_steps
        self.create_optimizer_and_scheduler(num_training_steps=self.t_total, build_l0_optimizer = self.start_prune)
            
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

        self.global_step = 0
        self.epoch = 0
        self.total_flos = 0

        epochs_trained = 0

        tr_loss = torch.tensor(0.0).to(self.args.device)
        ce_loss = torch.tensor(0.0).to(self.args.device)
        lag_loss = torch.tensor(0.0).to(self.args.device)

        logging_loss_scalar = 0.0
        logging_ce_loss_scalar = 0.0
        logging_lag_loss_scalar = 0.0

        model.zero_grad()
        if self.l0_module is not None:
            self.l0_module.zero_grad()

        self.optimizer.zero_grad()
        if self.l0_optimizer is not None:
            self.l0_optimizer.zero_grad()
        if self.lagrangian_optimizer is not None:
            self.lagrangian_optimizer.zero_grad()

        disable_tqdm = self.args.disable_tqdm or not self.is_local_process_zero()
        train_pbar = trange(epochs_trained, int(
            np.ceil(num_train_epochs)), desc="Epoch", disable=disable_tqdm)

        # self.evaluate()

        # training
        for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))): #! 20 epoch
            epoch_start = time.time()

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            epoch_iterator = train_dataloader
            print("training on dataset with pruning prompt")

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None

            epoch_pbar = tqdm(epoch_iterator, desc="Iteration",
                              disable=disable_tqdm)
            # self.eval_counter.clear()
            # self.celoss_counter.clear()
            for step, inputs in enumerate(epoch_iterator):
                if (self.prepruning_finetune_steps > 0 and self.global_step == self.prepruning_finetune_steps) or \
                    (self.prepruning_finetune_steps == 0 and self.start_prune == False): 
                    self.start_prune = True
                    lr_steps = self.t_total - self.global_step
                    self.create_optimizer_and_scheduler(lr_steps, self.start_prune)
                    logger.info("Starting l0 regularization!")
                if self.start_prune:
                    zs = self.l0_module.forward(training=True) #! get the zs
                    self.fill_inputs_with_zs(zs, inputs) #! use the zs

                # self.zs is not None when finetune
                if self.zs is not None:
                    self.fill_inputs_with_zs(self.zs, inputs)

                loss_terms = self.training_step(model, teacher_model, inputs)
                tr_loss_step = loss_terms["loss"]
                ce_loss_step = loss_terms["ce_loss"]
                lag_loss_step = loss_terms["lagrangian_loss"]

                tr_loss += tr_loss_step
                ce_loss += ce_loss_step
                lag_loss += lag_loss_step if lag_loss_step is not None else 0.0

                self.total_flos += self.floating_point_ops(inputs)

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                        len(epoch_iterator) <= self.args.gradient_accumulation_steps
                        and (step + 1) == len(epoch_iterator)
                ):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), self.args.max_grad_norm)

                    # self.optimizer.step()
                    if self.deepspeed:
                        self.deepspeed.step()

                    if self.l0_module is not None and self.l0_optimizer is not None:
                        self.l0_optimizer.step()
                        self.lagrangian_optimizer.step()

                    if self.lr_scheduler is not None and not self.deepspeed:
                        self.lr_scheduler.step()

                    if self.l0_module is not None:
                        self.l0_module.constrain_parameters()

                    model.zero_grad()
                    if self.l0_module is not None:
                        self.l0_module.zero_grad()
                    if self.l0_optimizer is not None:
                        self.l0_optimizer.zero_grad()
                    if self.lagrangian_optimizer is not None:
                        self.lagrangian_optimizer.zero_grad()

                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)
                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs: Dict[str, float] = {}
                        tr_loss_scalar = tr_loss.item()
                        ce_loss_scalar = ce_loss.item()
                        lag_loss_scalar = lag_loss.item()

                        logs["loss"] = (
                            tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["ce_loss"] = (
                            ce_loss_scalar - logging_ce_loss_scalar) / self.args.logging_steps
                        logs["lag_loss"] = (
                            lag_loss_scalar - logging_lag_loss_scalar) / self.args.logging_steps

                        # backward compatibility for pytorch schedulers
                        if self.lr_scheduler is not None:
                            lr = self.lr_scheduler.get_last_lr()[0] if version.parse(
                                torch.__version__) >= version.parse("1.4") else self.lr_scheduler.get_lr()[0]
                        else:
                            lr = self.args.learning_rate

                        logs["learning_rate"] = lr
                        logs["lambda_1"] = self.l0_module.lambda_1.item() if self.l0_module is not None else None
                        logs["lambda_2"] = self.l0_module.lambda_2.item() if self.l0_module is not None else None
                        logs["expected_sparsity"] = loss_terms["expected_sparsity"]
                        logs["target_sparsity"] = loss_terms["target_sparsity"]
                        logging_loss_scalar = tr_loss_scalar
                        logging_ce_loss_scalar = ce_loss_scalar
                        logging_lag_loss_scalar = lag_loss_scalar

                        # self.log(logs)
                        if self.args.local_rank == 0:
                            for k, v in logs.items():
                                try:
                                    mlflow.log_metric(k, v, step=self.state.global_step)
                                except:
                                    pass

                        try:
                            self.l0_module.eval()
                            zs = self.l0_module.forward(training=False)
                            pruned_model_size_info = self.l0_module.calculate_model_size(zs)
                        except:
                            pruned_model_size_info = {}

                        logger.info(f"{logs}, {pruned_model_size_info}")
                        if self.args.local_rank == 0:
                            for k, v in pruned_model_size_info.items():
                                try:
                                    mlflow.log_metric(k, v, step=self.state.global_step)
                                except:
                                    pass

                        logger.info(f"{logs}, {pruned_model_size_info}")

                        if self.l0_module is None and self.zs is not None:
                            best_so_far = self.celoss_counter.update(
                                self.epoch, self.global_step, logs["loss"])
                            if best_so_far:
                                best_dir = "./best/"
                                if not os.path.exists(best_dir):
                                    try:
                                        os.makedirs(best_dir)
                                    except:
                                        pass

                                best_loss_log = {
                                    "best_eval_score_so_far": self.celoss_counter.best_ce_loss,
                                    "best_step": self.celoss_counter.global_step,
                                    "best_epoch": self.celoss_counter.epoch
                                }

                                if self.args.local_rank == 0:
                                    for k, v in best_loss_log.items():
                                        try:
                                            mlflow.log_metric(k, v, step=self.global_step)
                                        except:
                                            pass

                                # save model to container
                                lora_weights = {}
                                for n, m in model.named_parameters():
                                    if 'lora_' in n:
                                        gather = lora.should_gather(m)
                                        with gather:
                                            lora_weights[n.replace('module.','')] = m.data
                                if self.args.local_rank == 0:
                                    torch.save(lora_weights, './best/lora_weights.pt')
                                logger.info(f"Saving the best model so far: [Epoch {int(self.epoch)} | Step: {self.global_step} | Score: {round(self.celoss_counter.best_ce_loss, 5)}]")

                epoch_pbar.update(1)
                torch.cuda.empty_cache()
                if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                    break

            epoch_end = time.time()
            self.evaluate()
            torch.cuda.empty_cache()
            if os.path.exists("./best/"):
                os.system(f"cp -r ./best/ {self.args.output_dir}")

            # save model via azcopy
            lora_weights = {}
            for n, m in model.named_parameters():
                if 'lora_' in n:
                    gather = lora.should_gather(m)
                    with gather:
                        lora_weights[n.replace('module.','')] = m.data
            if self.args.local_rank == 0:
                try:
                    epoch_output_dir = '{}/epoch{}'.format(self.args.output_dir, epoch)
                    print("Epoch folder: ", epoch_output_dir)
                    if not os.path.exists(epoch_output_dir):
                        os.makedirs(epoch_output_dir)
                    torch.save(lora_weights,'{}/lora_weights.pt'.format(epoch_output_dir))
                    if self.zs == None and self.l0_module is not None:
                        self.save_model_mask(epoch_output_dir)
                except:
                    print("Save epoch results failed. Skip it.")

            logger.info(
                f"Epoch {epoch} finished. Took {round(epoch_end - epoch_start, 2)} seconds.")

            epoch_pbar.close()
            train_pbar.update(1)

            if self.args.max_steps > 0 and self.global_step >= self.args.max_steps:
                break
            
        train_pbar.close()

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")
        return TrainOutput(self.global_step, tr_loss.item() / self.global_step, None)
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        # 13.add deepspeed
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
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
        disable_tqdm = not self.is_local_process_zero() or self.args.disable_tqdm

        zs = None
        if self.start_prune:
            self.l0_module.eval()
            zs = self.l0_module.forward(training=False)
        if zs is not None:
            pruned_model_size_info = self.l0_module.calculate_model_size(zs)
        if self.zs is not None and self.l0_module is not None:
            pruned_model_size_info = self.l0_module.calculate_model_size(self.zs)
            print("[Finetuning Phase] pruned model size info", pruned_model_size_info)

        for ii, inputs in enumerate(tqdm(dataloader, desc=description, disable=disable_tqdm)):
            if zs is not None:
                if ii == 0:
                    logger.info(f"Putting zs {zs.keys()} into inputs:")
                self.fill_inputs_with_zs(zs, inputs) #! use the zs
            if self.zs is not None:
                if ii == 0:
                    logger.info(f"Putting zs {self.zs.keys()} into inputs:")
                self.fill_inputs_with_zs(self.zs, inputs)
            batch_size = inputs[list(inputs.keys())[0]].shape[0]

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            # logits = logits[0]
            inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if inputs_decode is not None:
                inputs_decode = self._pad_across_processes(inputs_decode)
                inputs_decode = self._nested_gather(inputs_decode)
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (ii + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        # Main evaluation loop
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
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
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        if all_losses is not None and len(all_losses) > 0:
            metrics["eval_loss"] = np.mean(all_losses)

        if zs is not None:
            lag_loss, expected_sparsity, target_sparsity = self.l0_module.lagrangian_regularization(
                self.global_step - self.prepruning_finetune_steps)

            expected_sparsity = round(expected_sparsity.item(), 5)
            metrics.update(pruned_model_size_info) # TODO: check here
            metrics["lag_loss"] = lag_loss
            metrics["expected_sparsity"] = expected_sparsity
            metrics["target_sparsity"] = target_sparsity

            if (not self.start_saving_best) and (expected_sparsity - self.additional_args.target_sparsity >= -self.additional_args.sparsity_epsilon):
                self.start_saving_best = True
                logger.info(f"Starting saving the best from epoch {int(self.epoch)} and step {self.global_step}")

        self.model.config.output_hidden_states = True
        self.model.config.output_attentions = True
        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics[f"{metric_key_prefix}_perplexity"] = perplexity

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
    
    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Tuple[Dict[str, float], List]:
        self._memory_tracker.start()

        logger.info("-" * 70)
        logger.info(f"time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
        output_metrics = {}
        for dataset_name in self.additional_args.eval_dataset_name:
            if dataset_name == "wikitext":
                eval_dataloader = self.get_eval_dataloader(eval_dataset)
                start_time = time.time()
                output = self.evaluation_loop(
                    eval_dataloader, description="Evaluation")

                eval_score = output.metrics["eval_accuracy"]

                # logger.info(f"starting saving best: {self.global_step} {self.start_saving_best}")
                total_batch_size = self.args.eval_batch_size * self.args.world_size
                output.metrics.update(
                    speed_metrics(
                        f"eval",
                        start_time,
                        num_samples=output.num_samples,
                        num_steps=math.ceil(output.num_samples / total_batch_size),
                    )
                )
                output_metrics.update({f"{dataset_name}_{k}": v for k, v in output.metrics.items()})

            loggable_output_metrics = {k: v for k, v in output_metrics.items() if not isinstance(v, list)}
            # self.log(loggable_output_metrics)
            if self.args.local_rank == 0:
                for k, v in loggable_output_metrics.items():
                    try:
                        mlflow.log_metric(k, v, step=self.global_step)
                    except:
                        pass
            logger.info(loggable_output_metrics)
            print_keys = [f"{dataset_name}_{item}" for item in [
                'step', "eval_loss", "eval_accuracy"
                'eval_expected_sparsity', 'eval_target_sparsity'
                "eval_layers", 'eval_hidden_dims', 'eval_intermediate_dims', 'eval_head_nums',
            ]]
            logger.info(f"Evaluating: {', '.join([f'{k}: {v}' for k, v in output_metrics.items() if k in print_keys])}")

        if self.start_saving_best:
            best_so_far = self.eval_counter.update(
                self.epoch, self.global_step, eval_score)
            if best_so_far:
                best_dir = os.path.join(self.args.output_dir, "best")
                if not os.path.exists(best_dir):
                    os.makedirs(best_dir)

                if self.l0_module is not None:
                    zs = self.l0_module.forward(training=False)
                    torch.save(zs, os.path.join(best_dir, "zs.pt"))
                    torch.save(self.l0_module, os.path.join(
                        best_dir, "l0_module.pt"))
                best_eval_score_log = {
                    "best_eval_score_so_far": self.eval_counter.best_eval_score,
                    "best_step": self.eval_counter.global_step,
                    "best_epoch": self.eval_counter.epoch
                }
                # self.log(best_eval_score_log)
                if self.args.local_rank == 0:
                    for k, v in best_eval_score_log.items():
                        try:
                            mlflow.log_metric(k, v, step=self.global_step)
                        except:
                            pass
                logger.info(f"Saving the best model so far: [Epoch {int(self.epoch)} | Step: {self.global_step} | Model size: {output.metrics['remaining_params'] if 'remaining_params' in output.metrics else 'Full' } | Score: {round(eval_score, 5)}]")
                self.model.save_pretrained(best_dir)

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output_metrics)
        self._memory_tracker.stop_and_update_metrics(output_metrics)

        return output_metrics

    def save_model_mask(self, output_dir: Optional[str] = None):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        torch.save(self.l0_module, os.path.join(output_dir, "l0_module.pt"))

        zs = self.l0_module.forward(training=False)
        torch.save(zs, os.path.join(output_dir, "zs.pt"))

    def calculate_layer_distillation_loss(self, teacher_outputs, student_outputs, zs):
        mse_loss = torch.nn.MSELoss(reduction="mean")
        layer_loss = 0
        if self.additional_args.do_layer_distill: #! only do layer distill
            mlp_z = None
            head_layer_z = None
            # logger.info(f"zs={zs}")
            if "mlp_z" in zs:
                mlp_z = zs["mlp_z"].detach().cpu()
            if "head_layer_z" in zs:
                head_layer_z = zs["head_layer_z"].detach().cpu()

            teacher_layer_output = teacher_outputs[3][1:] #! hidden states, with a length of 12. Every has a shape of [32, 65, 768]
            student_layer_output = student_outputs[3][1:] 

            # distilliting existing layers
            layer_losses = []
            if self.additional_args.layer_distill_version == 2:
                for layer_num, (t_layer_o, s_layer_o) in enumerate(zip(teacher_layer_output, student_layer_output)):
                    s_layer_o = self.model.layer_transformation(s_layer_o)
                    l = mse_loss(t_layer_o, s_layer_o)
                    layer_losses.append(l)
                    if mlp_z is None or mlp_z[layer_num] > 0:
                        layer_loss += l
                # print("Layer distill loss", layer_losses)

            # distilling layers with a minimal distance
            elif self.additional_args.layer_distill_version > 2:
                l = []
                if self.additional_args.layer_distill_version > 4:
                    specified_teacher_layers = [i for i in range(len(teacher_layer_output))]
                    if self.additional_args.layer_distill_version ==5:
                        specified_teacher_layers = sorted(random.sample(specified_teacher_layers, 4))
                    elif self.additional_args.layer_distill_version ==6:
                        result_layers_T= []
                        skip_window = len(specified_teacher_layers)//4
                        for i in range(0, len(specified_teacher_layers), skip_window):
                            result_layers_T.append(random.sample(specified_teacher_layers[i:i+skip_window], 1)[0])
                        specified_teacher_layers = result_layers_T
                    specified_teacher_layers[0] = max(2, specified_teacher_layers[0])
                else:
                    specified_teacher_layers = [2, 5, 8, 11]
                # logger.info(f"sampled teacher layers: {specified_teacher_layers}")
                transformed_s_layer_o = [self.model.layer_transformation(
                    s_layer_o) for s_layer_o in student_layer_output]
                specified_teacher_layer_reps = [
                    teacher_layer_output[i] for i in specified_teacher_layers] #! teacher: 4x[32,113,768]

                device = transformed_s_layer_o[0].device
                for t_layer_o in specified_teacher_layer_reps:
                    for i, s_layer_o in enumerate(transformed_s_layer_o): #! student: 12x[32,113,768]
                        l.append(mse_loss(t_layer_o, s_layer_o))
                layerwiseloss = torch.stack(l).reshape(
                    len(specified_teacher_layer_reps), len(student_layer_output)) #! [4,12]

                existing_layers = None
                if head_layer_z is not None:
                    existing_layers = head_layer_z != 0
                    existing_layers = existing_layers.to(layerwiseloss.device)

                #! no ordering restriction specified
                if self.additional_args.layer_distill_version == 3:
                    alignment = torch.argmin(layerwiseloss, dim=1)
                #! added the ordering restriction -> to choose the min loss in 4 student layers
                elif self.additional_args.layer_distill_version in (3, 4, 5, 6):
                    last_aligned_layer = 12
                    alignment = []
                    for search_index in range(len(specified_teacher_layers)-1, -1, -1):
                        indexes = layerwiseloss[search_index].sort()[1]
                        if existing_layers is not None:
                            align = indexes[(
                                indexes < last_aligned_layer) & existing_layers]
                        else:
                            align = indexes[indexes < last_aligned_layer]
                        if len(align) > 0:
                            align = align[0]
                        else:
                            align = last_aligned_layer
                        alignment.append(align)
                        last_aligned_layer = align
                    alignment.reverse()
                    alignment = torch.tensor(alignment).to(device)
                else:
                    logger.info(
                        f"{self.additional_args.layer_distill_version} version is not specified.")
                    sys.exit()

                layerwise = torch.arange(len(specified_teacher_layers)).to(device)
                layer_loss += layerwiseloss[layerwise, alignment].sum() #! layerwise: teacher (specified layers) / alignment: student (min loss layers) / layerwiseloss: [4,12]
                if self.global_step % 100 == 0:
                    logger.info(f"v{self.additional_args.layer_distill_version} Global step: {self.global_step}, Alignment: " + str(alignment))
            return layer_loss
        else:
            return None
          
    def calculate_distillation_loss(self, teacher_outputs, student_outputs, zs):
        layer_loss = self.calculate_layer_distillation_loss(teacher_outputs, student_outputs, zs)
        distill_loss = layer_loss

        ce_distill_loss = F.kl_div(
            input=F.log_softmax(
                student_outputs[1] / self.additional_args.distill_temp, dim=-1), #! logits: [32,3]
            target=F.softmax(
                teacher_outputs[1] / self.additional_args.distill_temp, dim=-1), #! distill_temp: 2.0
            reduction="batchmean") * (self.additional_args.distill_temp ** 2)

        loss = self.additional_args.distill_ce_loss_alpha * ce_distill_loss
        if distill_loss is not None:
            loss += self.additional_args.distill_loss_alpha * distill_loss

        return distill_loss, ce_distill_loss, loss

    def shortens_inputs(self, inputs):
        max_length = inputs["attention_mask"].sum(-1).max().item()
        inputs["input_ids"] = inputs["input_ids"][:, :max_length]
        inputs["attention_mask"] = inputs["attention_mask"][:, :max_length]
        if "token_type_ids" in inputs:
            inputs["token_type_ids"] = inputs["token_type_ids"][:, :max_length]

    def training_step(self, model: torch.nn.Module,teacher_model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> List[torch.Tensor]:
        model.train()
        # teacher_model.train()
        if self.l0_module is not None:
            self.l0_module.train()
        inputs = self._prepare_inputs(inputs)

        distill_loss = None
        distill_ce_loss = None
        if teacher_model is not None:
            with torch.no_grad():
                # only retain inputs of certain keys
                teacher_inputs_keys = ["input_ids", "attention_mask", "token_type_ids", "position_ids", "labels",
                                       "output_attentions", "output_hidden_states", "return_dict"]
                teacher_inputs = {key: inputs[key]
                                  for key in teacher_inputs_keys if key in inputs}
                self.shortens_inputs(teacher_inputs)
                teacher_outputs = teacher_model(**teacher_inputs)
            self.shortens_inputs(inputs)
            student_outputs = model(**inputs) #! get the two outputs

            zs = {key: inputs[key] for key in inputs if "_z" in key} #! extract the zs
            distill_loss, distill_ce_loss, loss = self.calculate_distillation_loss(
                teacher_outputs, student_outputs, zs)
        else:
            # infer_s = time.time()
            loss = self.compute_loss(model, inputs)
            # infer_e = time.time()
            # print("Computing loss time:", infer_e - infer_s)

        ce_loss = loss.clone()
        lagrangian_loss = None
        expected_sparsity = None
        target_sparsity = None
        
        if self.start_prune :
            lagrangian_loss, expected_sparsity, target_sparsity = self.l0_module.lagrangian_regularization(
                self.global_step - self.prepruning_finetune_steps)
            loss += lagrangian_loss
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            loss = loss / self.args.gradient_accumulation_steps

        if self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            # s = time.time()
            loss = self.deepspeed.backward(loss)  #10-22G * number of GPU
            # e = time.time()
            # print("Deepspeed backward. Time:{}".format(e - s))
        else:
            loss.backward()
        return {"loss": loss.detach(),
                "ce_loss": ce_loss.detach(),
                "lagrangian_loss": lagrangian_loss.detach() if lagrangian_loss is not None else None,
                "expected_sparsity": expected_sparsity.item() if expected_sparsity is not None else 0.0,
                "target_sparsity": target_sparsity if target_sparsity is not None else 0.0,
                "distill_layer_loss": distill_loss.detach() if distill_loss is not None else None,
                "distill_ce_loss": distill_ce_loss.detach() if distill_ce_loss is not None else None}

    def fill_inputs_with_zs(self, zs, inputs):
        for key in zs:
            inputs[key] = zs[key]
        if self.l0_module is not None:
            inputs["block_layer_start"] = self.l0_module.block_layer_start
            inputs["block_layer_end"] = self.l0_module.block_layer_end


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = all(inputs.get(k) is not None for k in self.label_names)
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                    else:
                        logits = outputs
                    # TODO: this needs to be fixed and made cleaner later.
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)
        # logits = nested_detach(logits)
        #if len(logits) == 1:
        logits = logits[0]
        logits = nested_detach(logits)
        return (loss, logits, labels)