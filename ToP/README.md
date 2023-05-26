# ToP: Constraint-aware and Ranking-distilled Token Pruning

This repository contains code to reproduce the key results of the paper [Constraint-aware and Ranking-distilled Token Pruning for Efficient Transformer Inference]() in KDD 2023.

**ToP** is a constraint-aware and ranking-distilled token pruning method, which selectively removes unnecessary tokens as input sequence pass through layers, allowing the model to improve online inference speed while preserving accuracy. ToP reduces the average FLOPs of BERT by 8.1x while achieving competitive accuracy on GLUE, and provides a real latency speedup of up to 7.4x on an Intel CPU.

## Environment Setup

``` bash
pip3 -r requirements.txt
pip3 install transformers==4.15.0
```

NOTE: please use a lower version of transformers, because the latest version seems seems do not have ``hf_bucket_url`` in ``transformers.file_utils``.

## TBD

## Cite

If you found this work useful, please consider citing:

```
@article{microsoft/ToP,
  title={Constraint-aware and Ranking-distilled Token Pruning for Efficient Transformer Inference}, 
  TBD
}
```