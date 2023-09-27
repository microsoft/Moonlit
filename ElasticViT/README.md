# ElasticViT
This is the official implementation for our paper ElasticViT. 

[Paper](https://arxiv.org/pdf/2303.09730.pdf) [Poster](https://www.chentang.cc/assets/ICCV23_ElasticViT_poster.pdf) [Supernet Weight](https://drive.google.com/file/d/1M4AwUrFgPsLs-W6iPs-H4gKyns_tx1S8/view?usp=sharing)

We propose ElasticViT, a two-stage NAS approach that trains a high-quality ViT supernet over a very large search space for covering a wide range of mobile devices, and then searches an optimal sub-network (subnet) for direct deployment. However, current supernet training methods that rely on uniform sampling suffer from the gradient conflict issue: the sampled subnets can have vastly different model sizes (e.g., 50M vs. 2G FLOPs), leading to different optimization directions and inferior performance. To address this challenge, we propose two novel sampling techniques: complexity-aware sampling and performance-aware sampling. Complexity-aware sampling limits the FLOPs difference among the subnets sampled across adjacent training steps, while covering different-sized subnets in the search space. Performance-aware sampling further selects subnets that have good accuracy, which can reduce gradient conflicts and improve supernet quality. Our discovered models, ElasticViT models, achieve top-1 accuracy from 67.2% to 80.0% on ImageNet from 60M to 800M FLOPs without extra retraining, outperforming all prior CNNs and ViTs in terms of accuracy and latency. Our tiny and small models are also the first ViT models that surpass state-of-the-art CNNs with significantly lower latency on mobile devices. For instance, ElasticViT-S1 runs 2.62x faster than EfficientNet-B0 with 0.1% higher accuracy. 


## Environment Setup and Data Preparation

You can use the following command to setup the training/evaluation environment: 

```
git clone https://github.com/mapleam/Moonlit.git
cd ElasticViT
conda create -n ElasticViT python=3.8
conda activate ElasticViT
pip install -r requirements
```

We use the ImageNet dataset at http://www.image-net.org/. The training set is moved to /path_to_imagenet/imagenet/train and the validation set is moved to /path_to_imagenet/imagenet/val: 
```
/path_to_imagenet/imagenet/
  train/
    class0/
      img0.jpeg
      ...
    class1/
      img0.jpeg
      ...
    ...
  val/
    class0/
      img0.jpeg
      ...
    class1/
      img0.jpeg
      ...
    ...
```

## Supernet Training via Conflict-aware Techniques

Our training techniques, complexity-aware sampling, and performance-aware sampling are controlled by two main fields ```flops_sampling_method``` and ```model_sampling_method``` of our code. We provide the training scripts in configs/final_3min_space.yaml and you can directly run the training process by the following command: 

```
python -m torch.distributed.launch --nproc_per_node={GPU_PER_NODE} --node_rank={NODE_RANK} --nnodes={NUM_OF_NODES} --master_addr={MASTER_ADDR} --master_port={MASTER_PORT} train_eval_supernet.py configs/final_3min_space.yaml
```

You can also customize the search space (e.g., more layers, channels, v scale, etc.) and memory bank by modifying the YAML file. 

# Subnet Evaluation

See the configs/final_3min_space_eval_400M.yaml for evaluation with a specific model, please use the same settings (i.e., the search space) as training to construct the model. Please remember to enable the ```eval``` flag and give a specific architecture in ```arch```. 
Meanwhile, please put the path of supernet checkpoint in the YAML file's ```resume.path```. 

We provide the evaluation scripts of our searched 400 MFLOPs model in configs/final_3min_space_eval_400M.yaml and you can directly run it by the following command: 

```
python -m torch.distributed.launch --nproc_per_node=1 train_eval_supernet.py configs/final_3min_space_eval_400M.yaml
```

# Subnet Search

After loading the supernet checkpoint, you can also search the model by your constraint. You can run our evolution search by a specific FLOPs constraint with the following command: 

```
python -m torch.distributed.launch --nproc_per_node={GPU_PER_NODE} search_subnet_via_flops.py configs/final_3min_space.yaml --flops_limits {LIMITS}
```

# FLOPs Tables
To accelerate the sampling process, we sample multiple offline models for each memory bank before supernet training. You can generate the look-up table of one FLOPs by the following code, meanwhile, please remember to provide the multiple smallest subnets. 

```
python flops_look_up_table/build_look_up_table.py configs/final_3min_space.yaml --flops {FLOPS}
```

# Citation

If ElasticVit is useful or relevant to your research, please kindly recognize our contributions by citing our paper:

```
@inproceedings{tang2023elasticvit,
  title={ElasticViT: Conflict-aware Supernet Training for Deploying Fast Vision Transformer on Diverse Mobile Devices},
  author={Tang, Chen and Zhang, Li Lyna and Jiang, Huiqiang and Xu, Jiahang and Cao, Ting and Zhang, Quanlu and Yang, Yuqing and Wang, Zhi and Yang, Mao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```