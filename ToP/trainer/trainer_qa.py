import datasets
from transformers.trainer_utils import PredictionOutput
from trainer.trainer import ToPTrainer 
import os
import torch
import torch.nn.functional as F
from transformers.utils import logging
import time
import numpy as np

logger = logging.get_logger(__name__)

class ToPQATrainer(ToPTrainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        ToPTrainer.__init__(self, *args, **kwargs)
        
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None):

        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                # ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        # We might have removed columns from the dataset so we put them back.
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(type=eval_dataset.format["type"], columns=list(eval_dataset.features.keys()))

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
            metrics = self.compute_metrics(eval_preds)
            self.log(metrics)
        else:
            metrics = {}
        metrics.update(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)

        name = "f1"
        metrics["step"] = self.state.global_step
        loggable_output_metrics = {k: v for k, v in metrics.items() if not isinstance(v, list)}
        self.log(loggable_output_metrics)
        print_keys = [name, 'macs_sparsity', "expected_sparsity", 'target_sparsity', 'step', "token_prune_loc"]
        print("-" * 70)
        print("time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print(f"Evaluating: {', '.join([f'{k}: {v}' for k, v in metrics.items() if k in print_keys])}")
        if self.l0_module is not None:
            print(f"lambda_1: {self.l0_module.lambda_1.item():.4f}, lambda_2: {self.l0_module.lambda_2.item():.4f} lambda_3: {self.l0_module.lambda_3.item():.4f}")
        if "token_ratio_for_training" in metrics:
            print("train remain:", np.round(metrics["token_ratio_for_training"].detach().cpu().numpy(), 2))
            print("infer remain:", metrics["token_ratio_for_inference"])
            print("layerwise remain:", metrics["remaining_token_ratio"])
            for z in metrics["token_z"]:
                print(self.get_z_str(z))
        if self.best_eval_score_so_far is not None:
            print(f"Best eval score so far: {self.best_eval_score_so_far:.4f} @ step {self.best_step} epoch {self.best_epoch:.2f}")
            self.log({"best_eval_score_so_far": self.best_eval_score_so_far, "best_step": self.best_step, "best_epoch": self.best_epoch})

        eval_score = metrics[name]

        if self.start_saving_best:
            best_so_far = self.eval_counter.update(
                self.epoch, self.state.global_step, eval_score)
            if best_so_far:
                best_dir = os.path.join(self.args.output_dir, "best")
                if not os.path.exists(best_dir):
                    os.makedirs(best_dir)
                info_text = f"Saving the best model so far: [Epoch {int(self.epoch)} | Step: {self.state.global_step} | MACs sparsity: {metrics['macs_sparsity'] if 'macs_sparsity' in metrics else 'Full' } | Score: {round(eval_score, 5)}"
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
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(type=test_dataset.format["type"], columns=list(test_dataset.features.keys()))

        eval_preds = self.post_process_function(test_examples, test_dataset, output.predictions)
        metrics = self.compute_metrics(eval_preds)
        return PredictionOutput(predictions=eval_preds.predictions, label_ids=eval_preds.label_ids, metrics=metrics)

    def calculate_distillation_loss(self, teacher_outputs, student_outputs, zs, token_masks):
        distill_ce_loss = F.kl_div(
            input=F.log_softmax(student_outputs.start_logits / self.additional_args.distill_temp, dim=-1),
            target=F.softmax(teacher_outputs.start_logits / self.additional_args.distill_temp, dim=-1),
            reduction="batchmean",
        ) * (self.additional_args.distill_temp ** 2)

        distill_ce_loss += F.kl_div(
            input=F.log_softmax(student_outputs.end_logits / self.additional_args.distill_temp, dim=-1),
            target=F.softmax(teacher_outputs.end_logits / self.additional_args.distill_temp, dim=-1),
            reduction="batchmean",
        ) * (self.additional_args.distill_temp ** 2)

        distill_ce_loss = distill_ce_loss / 2
        loss = distill_ce_loss
        layer_distill_loss = None
        return layer_distill_loss, distill_ce_loss, loss
