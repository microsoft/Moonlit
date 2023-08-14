# SpaceEvo
This is the code for our paper SpaceEvo. We propose an algorithm to search for INT8-quantization-friendly search spaces 
based on block-wise knowledge distillation. Then we use NAS algorithms to train and quantize the supernet and 
search for efficient 8-bit subnets for deployment on different platforms.

## Checkpoints
Our trained supernets' checkpoints are available in
https://drive.google.com/drive/folders/1Bj-EdyAIKWtzKn86pNkLtOE8_HikDONL.

There are five supernets: spaceevo@pixel4, spaceevo@pixel4-medium, spaceevo@pixel4-tiny,
spaceevo@vnni and spaceevo@vnni-large.

Download them to `./checkpoints/supernet_training`.

## Evaluation
The configs of the specific supernets and subnets after search are in `./data/search_results.yaml`, which contains:
* five supernets: spaceevo@pixel4, spaceevo@pixel4-medium, spaceevo@pixel4-tiny,
spaceevo@vnni and spaceevo@vnni-large.
* ten subnets: SEQnet@pixel4-A[0-4], SEQnet@vnni-A[0-4].

You can evaluate these supernets max-subnet's and min-subnet's accuracy and subnet's accuracy by running `eval_specific_net.py`.
```shell
# evaluate supernet's int8 accuracy
python eval_specific_net.py --model_name spaceevo@vnni --dataset_dir ${IMAGENET_PATH} --quant_mode
# evaluate supernet's fp32 accuracy
python eval_specific_net.py --model_name spaceevo@vnni --dataset_dir ${IMAGENET_PATH}

# evaluate subnet's int8 accuracy
python eval_specific_net.py --model_name SEQnet@pixel4-A0 --dataset_dir ${IMAGENET_PATH} --quant_mode
python eval_specific_net.py --model_name SEQnet@vnni-A0 --dataset_dir ${IMAGENET_PATH} --quant_mode
# evaluate subnet's fp32 accuracy
python eval_specific_net.py --model_name SEQnet@pixel4-A0 --dataset_dir ${IMAGENET_PATH}
```

## Usage
The whole pipeline contains the following procedures: 
1. search space search based on block_kd
   1. train and lsq+ block, see *train_block_kd.py* and *lsq_block_kd.py*
   2. build block lut, see *eval_block_kd.py*
   3. search, see *search_block_kd.py*
2. supernet training and lsq+, see *train_supernet.py*
3. subnet search, see *search_subnet.py*
4. eval supernet and subnet (latency and accuracy)
   1. eval accuracy and predict latency, see *eval_supernet.py*
   2. benchmark latency on real device: see *onnx_tools/* and *tflite_tools/*

Assume all the checkpoints and results are stored in ${CHECKPOINT_DIR} (default is `./checkpoints`). 
The directory layout is:
```
${CHECKPOINT_DIR}
|   block_kd/teacher_checkpoint/efficientnet_b5/checkpoint.pth
|
└---block_kd/mobilew # block_kd checkpoints of mobilew
|   |   stage1_0/
|   |   .../
|   |   stage6_6/
|
└---block_kd/onnxw $ block_kd checkpoints of onnxw
|   |
|
└---block_kd/search # block_kd searhing results
|   |   mobilew/
|   |   onnxw/
|
└---supernet_training
|   |   mobilew-312120-155501-align0
|   |       |   checkpoint.pth
|   |       |   lsq.pth
|   |       |   ...
|   |   onnxw_-121121-023230-align0
|   |   spaceevo@pixel4
|   |   spaceevo@vnni
|
└---search_subnet
    |   onnx-111211/*.log
    |   ...
```

### Setup nn-Meter
We use nn-Meter to predict model's latency. To setup nn-Meter, do the following procedure:

First download and unzip *nn-meter-predictor.zip* from https://drive.google.com/drive/folders/1Bj-EdyAIKWtzKn86pNkLtOE8_HikDONL.
You will get a folder named *tflite-int8-predictor*. Assume the folder's path is `${tflite_predictor_path}`.

Set the `package_location` entry in *meta.yaml* to `${tflite_predictor_path}`.
```yaml
name: tflite27_cpu_int8
version: 1.0
category: cpu
package_location: <Change this to this meta.yaml file's directory path>
kernel_predictors:
    - conv-bn-relu
    ...
```
Then install nn-meter and register tflite-int8-predictor.
```shell
git clone https://github.com/microsoft/nn-Meter.git
cd nn-Meter
git checkout dev/quantize-predictor
python setup.py

# register tflite predictor
nn-meter register --predictor ${tflite_predictor_path}/meta.yaml
```

### search space search

#### train block kd
We use LSQ+ quantized efficientnet-b5 to distill blocks. The checkpoint is also in the above 
[google-drive link](https://drive.google.com/drive/folders/1Bj-EdyAIKWtzKn86pNkLtOE8_HikDONL).
Download it to `checkpoints/block_kd/teacher_checkpoint/efficientnet_b5/checkpoint.pth`

First you need to train and LSQ+ QAT all the blocks in stage1-stage6 in the hyperspace (superspace). 
Blocks are independent, thus can be trained in parallel. 
You can specify the block id in argument `--stage_list`.
```
# first train block in fp32 mode
python -m torch.distributed.launch --nproc_per_node=4 train_block_kd.py \
--superspace <superspace> \
--output_path ${CHECKPOINT_DIR}/block_kd/<superspace> \
--inplace_distill_from_teacher \
--num_epochs 5 \
--stage_list stage1_0 stage1_2 \
--hw_list 160 192 224 \
--dataset_path <path_to_imagenet>

# then lsq+ in quant mode
python -m torch.distributed.launch --nproc_per_node=4 lsq_block_kd.py \
--superspace <superspace> \
--output_path ${CHECKPOINT_DIR}/block_kd/<superspace> \
--inplace_distill_from_teacher \
--num_epochs 1 \
--stage_list stage1_0 stage1_2 \
--hw_list 160 192 224 \
--dataset_path <path_to_imagenet> \
--teacher_checkpoint_path ${CHECKPOINT_DIR}/block_kd/teacher_checkpoint/efficientnet_b5/checkpoint.pth \
--train_batch_size 32 \
--learning_rate_list 0.00125 0.00125 0.00125 0.00125 0.00125 0.00125
```
In the above example, we train 2 blocks: stage1_0 and stage1_2. 
`<superspace>`can be chosen in [mobilew | onnxw | ...].

#### build block lut
To speed up space search with block_kd. We first sample 1000 points from each possible candidate blocks in stage1-6,
which serve as a look-up table. During space search, all the points are sampled from this table, eliminating the need 
to do evaluation, which is efficient. 
```
python eval_block_kd.py \
--superspace [mobilew|onnxw] \
--platform [tflite27_cpu_int8|onnx_lut] \
--output_path ${CHECKPOINT_DIR}/block_kd/lut \
--stage_list stage3_0 stage3_1 \
--width_window_filter 0 1 2 \
--hw_list 160 192 224 \
--dataset_path ./dataset \
--checkpoint_path ${CHECKPOINT_DIR}/block_kd \
--teacher_checkpoint_path ${CHECKPOINT_DIR}/block_kd/teacher_checkpoint/efficientnet_b5/checkpoint.pth \
--debug
```
`--stage_list` and `--width_window_filter` specify the blocks and the width window candidates to build lut. 
The above scripts will build 6 luts: stage3_0_0, stage3_0_1, stage3_0_2, stage3_1_0, stage3_1_1, stage3_1_2. 
When `--debug` argument is set, evaluation is performed only on 10 batches. 
We found after a few batches, the loss becomes stable, so we set `--debug` flag when building block lut to speed up this process.

Each line in the output lut csv file represents a stage sampled from the dynamic stage. There are 6 items in a line, whose meanings are

| sub-stage-config | input shape  | nsr-loss | FLOPS(M) | Params(M) | pred int8 latency (ms) |
|------------------|--------------|----------|----------|-----------|------------------------|
| 5#32#8_3#40#3    | 1x24x112x112 | 0.2734   | 118.7395 | 0.0489    | 11.7578                |

If a substage has 2 blocks (depth=2) and each block bi has (kernel_size, width, expand_ratio) = (ki, wi, ei), then it can be encoded as *k1#w1#e1_k2#w2_e2*. 

The built LUT stores in `data/block_lut`.

#### search space search
Search space search is very fast because no neural network forward is needed. Subnets are sampled from the previously built look-up table. 
Thus search space search runs locally. All other training and searching processes run in the cluster.
```
# search on hyperspace mobilew with latency constraint {15 20}, latency_loss_t 0.08 and latency_loss_a 0.01
python search_block_kd.py --superspace mobilew --latency_constraint 15 20 --platform tflite27_cpu_int8 --latency_loss_t 0.08 --latency_loss_a 0.01 --lut_path data/block_lut --output_dir ${CHECKPOINT_DIR}/block_kd/search

# search on hyperspace onnxw with latency constraint 10
python search_block_kd.py --superspace onnxw --latency_constraint 10 --platform onnx_lut --latency_loss_t 0.08 --latency_loss_a 0.01 --lut_path data/block_lut --output_dir ${CHECKPOINT_DIR}/block_kd/search
```
You can also get the quality score of specific search spaces.
```
python search_block_kd.py --superspace onnxw --latency_constraint 15 --platform onnx_lut --latency_loss_t 0.08 --supernet_choices 111111_000000 222222_000000
```

### supernet training
```
# first train 360 epochs in fp32 mode
python -m torch.distributed.launch --nproc_per_node 8 train_supernet.py \
--config-file supernet_training_configs/train.yaml \
--superspace mobilew \
--supernet_choice 123214-012321 \
--batch_size_per_gpu 64 \
--resume
# then lsq+ for 50 epochs
python -m torch.distributed.launch --nproc_per_node 8 train_supernet.py \
--config-file supernet_training_configs/train.yaml \
--superspace mobilew \
--supernet_choice 123214-012321 \
--batch_size_per_gpu 32 \
--quant_mode
```
A supernet can be encoded as `<hyperspace>-<block_type_choices>-<width_window_choices>`, e.g., mobilew-111211-123211. The above script trains and QAT supernet mobilew-123214-012321. `Supernet.build_from_str` method builds a supernet torch model from a str encoding.

### search target subnet
```
python -m torch.distributed.launch --nproc_per_node 4 search_subnet.py \
--superspace onnxw \
--supernet_choice 121122-133333 \
--dataset_path <path_to_imagenet> \
--output_path ${CHECKPOINT_DIR}/search_subnet \
--checkpoint_path ${CHECKPOINT_DIR}/supernet_training \
--latency_constraint 15 \
--latency_delta 2 \
--batch_size 32 \
--num_calib_batches 20
```
Before searching, make sure the supernet's checkpoint after lsq+ qat exists. In the above example, the target checkpoint path is `${CHECKPOINT_DIR}/supernet_training/onnxw-121122-133333-align0/lsq.pth`. Also the code needs nn-meter installed and registered.
The valid latency range is specified by `--latency_constraint c` and `--latency_delta d`. The range is [c-d, c].  

### evaluate subnet
A subnet can be encoded as the depth, width, kernel_size, expand_ratio and resolution choices from the supernet, e.g.,  d1#1#2#4#5#4#6#2_k3#3#5#5#3#5#5#3#5#3#3#3#5#3#3#5#5#5#5#3#3#5#5#5#5_w32#32#32#48#48#48#64#64#96#112#96#112#80#144#144#144#128#240#256#256#256#256#256#432#432_e0#0.5#8.0#6.0#8.0#6.0#8.0#4.0#6.0#6.0#4.0#4.0#4.0#6.0#8.0#4.0#6.0#6.0#6.0#6.0#8.0#8.0#4.0#8.0#8.0_r224.
```
# evaluate the fp32 accuracy of a list of subnets in a supernet
python eval_supernet.py --superspace onnxw --supernet_choice 121122-133333 --mode acc --resume ${CHECKPOINT_DIR}/supernet_training --dataset_dir ./dataset --subnet_choice <subnet_choice1> <subnet_choice2> <...> 

# evaluate the int8 accuracy of a list of subnets in a supernet
python eval_supernet.py --superspace onnxw --supernet_choice 121122-133333 --mode acc --resume ${CHECKPOINT_DIR}/supernet_training --dataset_dir ./dataset --subnet_choice <subnet_choice1> <subnet_choice2> <...> --quant_mode

# also you can run this code with torch ddp to speed up evaluation: python -m torch.distributed.launch --nproc_per_node 4 eval_supernet.py ...

# predict the latency of a list of subnets
python eval_supernet.py --superspace onnxw --supernet_choice 121122-133333 --mode lat --subnet_choice <subnet_choice1> <subnet_choice2> <...> 
```

### benchmark subnet latency
```
##### benchmark onnx latency #####
# 1. write subnet encoding to onnx_tools/input.csv. Each line represents a subnet: <superspace>,<supernet_choice>,<subnet_choice>

# 2. export onnx
python onnx_tools/export_onnx.py --skip_weights

# 3. benchmark
python onnx_tools/benchmark.py


##### benchmark tflite latency #####
# 1. write subnet encoding to tflite_tools/input.csv. Each line represents a subnet: <superspace>,<supernet_choice>,<subnet_choice>

# 2. export and benchmark
python tflite_tools/benchmark.py

```

## LSQ+ Implementation
LSQ+ quantization is implemented in *modules/modeling/ops/lsq_plus.py*. The main components are
* function `quantize_activation(activation, scale, num_bits, beta, is_training)` fake quantize (quantize and then de-quantize) the activation using parameter *scale* and *beta*.
* function `quantize_weight(weight, scale, num_bits, is_training)` fake quantize the weight using *scale* (no offset parameter is needed because the weight quantization is symmetric).
* class `QBase` is the base class for a quantized OP. It initializes three parameters *activation_scale*, *activation_beta*, 
*weight_scale* and provides quantization parameters initial methods and fake quantize method. There are two initial methods: 
min_max_initial and lsq_initial. We use the first one, which is simple.
* function `set_quant_mode(model: nn.Module)` sets a torch model with lsq+ op to int8 mode.

A quantized op can be implemented by inheriting `QBase` and `nn.Module`, see *modules/modeling.ops/op.py*, which contains `QConv` and `QLinear`.

Because the normal training flow is first training in fp32 mode and then qat in int8 mode, models are initialized into fp32 mode, by setting the `nbits_w` and `nbits_a` attributes in `QBase` to 32. 
In forward pass, lsq+ ops in fp32 mode behave the same as normal torch modules. To change a model to int8 mode, all `nbits_w` and `nbits_a` attributes are needed to change to 8, which can be done by function `set_quant_mode`.

## DownStream Classification
```
python downstream_cls.py --subnet_name SEQnet@vnni-A0  --dataset CIFAR10 --imagenet_path xxx
```