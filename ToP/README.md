# ToP: Constraint-aware and Ranking-distilled Token Pruning for Efficient Transformer Inference

This repo contains the source code for our KDD'2023 paper titled ToP: Constraint-aware and Ranking-distilled Token Pruning for Efficient Transformer Inference. ToP is a constraint aware token pruning method that are applicable to various models such as BERT and RoBERTa, and various datasets such as GLUE and 20news. Check our [paper](https://arxiv.org/abs/2306.14393) for more details.

## Installation

```bash
conda create -n top python=3.8.8
conda activate top
pip3 -r requirements.txt
```

## Results and Models

| Task | Metric   | FLOPs Reduction | Score | Checkpoint                                                          | Training Log                                                                              |
| ---- | -------- | --------------- | ----- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| CoLA | Matthews | 10.39x          | 60.3  | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-cola) | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-cola/raw/main/cola-log.txt) |
| RTE  | Accuracy | 7.71x           | 68.3  | [link]()                                                               | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-rte/raw/main/rte-log.txt)   |
| QQP  | Accuracy | 12.41x          | 90.9  | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-qqp)  | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-qqp/raw/main/qqp-log.txt)   |
| MRPC | F1       | 7.71x           | 89.1  | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-mrpc) | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-mrpc/raw/main/mrpc-log.txt) |
| SST2 | Accuracy | 4.66x           | 93.4  | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-sst2) | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-sst2/raw/main/sst2-log.txt) |
| MNLI | Accuracy | 6.68x           | 83.4  | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-mnli) | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-mnli/raw/main/mnli-log.txt) |
| QNLI | Accuracy | 6.16x           | 89.0  | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-qnli) | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-qnli/raw/main/qnli-log.txt) |
| STSB | Pearson  | 7.20x           | 86.6  | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-stsb) | [link](https://huggingface.co/senfu/bert-base-uncased-top-pruned-stsb/raw/main/stsb-log.txt) |

## Evaluation

1. Download the checkpoint from the table above. For example, to download CoLA best checkpoint:

```bash
  # Download model checkpoint from huggingface.
  # Make sure you have git-lfs installed.
  #   sudo apt-get update
  #   sudo apt-get install git-lfs
  #   git lfs install
  git clone https://huggingface.co/senfu/bert-base-uncased-top-pruned-cola
```

2. Run the evaluation.

   ```bash
   bash scripts/run_Evaluation.sh $TASK $CHECKPOINT_FOLDER $GPU_ID
   ```

## Training

An example command to run ToP for SST-2:

```bash
bash run_token_prune.sh
```

There are a few parameters that we can tune to change the pruning behaviors and get better results:

* SPARSITY: the target token sparsity (excluding padding)
* PRUNE_LOCATION: the layers that we want to perform token pruning on. It can be either `2,3,4,5,6,7,8,9,10,11` or `3,4,5,6,7,8,9,10,11`.
* LEARNING_RATE: finetuning learning rate.
* REG_LEARNING_RATE: l0 regularization learning rate.
* DISTILL_RANK_LOSS_ALPHA: the loss factor of rank distillation loss

**NOTE on reproducing paper results:**

Due to the inevitable random cuda behavior introduced during the pruning process, the training results are different if you are using different environment. we recommend you to use the same environment listed below in order to correctly reproduce the results:

* Ubuntu 18.04
* NVIDIA V100 GPU, cuda11.1-cudnn8
* Python 3.8.8
* Torch 1.12.1+cu102
* Transformers 4.16.0
* Numpy 1.24.4
* Scipy 1.7.3

We conducted a grid search when producing the results reported in the paper. Following BERT finetuning guidance, we search over learning rate, l0 regularization learning rate and loss factor of rank distillation loss.

* learning rate: {6e-5, 5e-5, 4e-5, 3e-5, 2e-5, 1e-5}
* l0 regularization learning rate: {0.04, 0.02, 0.01}
* the loss factor of rank distillation loss: 1e-2 ~ 1e-5

For other parameters, we recommend using the configuration listed below:

| Hyperparameters | CoLA | RTE  | QQP  | MRPC | SST2 | MNLI | QNLI | STSB |
| --------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| BIN_NUM         | 20   | 100  | 50   | 50   | 25   | 50   | 50   | 30   |
| TOPK            | 10   | 20   | 20   | 20   | 20   | 20   | 20   | 20   |
| WARMUP_EPOCHS   | 50   | 50   | 10   | 150  | 10   | 10   | 10   | 50   |
| EPOCHS          | 100  | 80   | 40   | 200  | 40   | 40   | 40   | 150  |
| SPARSITY        | 0.43 | 0.59 | 0.65 | 0.67 | 0.4  | 0.5  | 0.58 | 0.7  |

## Deployment
Currently, token pruning acceleration for on-device deployment is missing in the code base. We are working on its implementation and plan to release the code soon. Stay tuned for updates.

## Citation

If ToP is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```
@article{li2023constraint,
  title={Constraint-aware and Ranking-distilled Token Pruning for Efficient Transformer Inference},
  author={Li, Junyan and Zhang, Li Lyna and Xu, Jiahang and Wang, Yujing and Yan, Shaoguang and Xia, Yunqing and Yang, Yuqing and Cao, Ting and Sun, Hao and Deng, Weiwei and others},
  booktitle = {Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  publisher = {Association for Computing Machinery},
  series = {KDD '23}
  year={2023}
}
```
