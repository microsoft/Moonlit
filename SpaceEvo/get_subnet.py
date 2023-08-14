import os
import sys
import yaml

FILE_DIR = os.path.dirname(__file__)

import torch 

from modules.modeling.supernet import Supernet
from modules.modeling.ops.lsq_plus import set_quant_mode
from modules.alphanet_training.evaluate.supernet_eval import calibrate_bn_params
from modules.training.dataset.imagenet_dataloader import build_imagenet_dataloader
from modules.modeling.blocks import LogitsBlock

# load search_result
SEARCH_RESULT_PATH = os.path.join(FILE_DIR, 'data/search_result.yaml')
y = yaml.safe_load(open(SEARCH_RESULT_PATH, 'r').read())
specific_supernet_encoding_dict = y['specific_supernet_encoding_dict']
specific_subnet_encoding_dict = y['specific_subnet_encoding_dict']
CHECKPOINT_PATH = os.path.join(FILE_DIR, 'checkpoints/supernet_training')

POST_BN_CALIB_BATCH_NUM = 32


def get_spaceevo_int8_pretrained_subnet(model_name, imagenet_path, device):
    if 'pixel4' in model_name:
        supernet_name = 'spaceevo@pixel4'
    elif 'vnni' in model_name:
        supernet_name = 'spaceevo@vnni'
    else:
        raise ValueError()

    # build imagenet dataloader to do bn calibration
    train_loader, *_ = build_imagenet_dataloader(dataset_path=imagenet_path, train_batch_size=64, eval_batch_size=32, distributed=False, num_workers=8, augment='auto_augment_tf')

    # get supernet
    supernet_encoding = specific_supernet_encoding_dict[supernet_name]
    subnet_encoding = specific_subnet_encoding_dict[model_name][1]

    supernet = Supernet.build_from_str(supernet_encoding)
    set_quant_mode(supernet)


    # load checkpoint
    checkpoint_path = os.path.join(CHECKPOINT_PATH, supernet_name, 'lsq.pth')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    new_state_dict = {}
    for k, v in checkpoint['state_dict'].items():
        new_state_dict[k.replace('module.', '')] = v
    checkpoint['state_dict'] = new_state_dict
    supernet.load_state_dict(checkpoint['state_dict'])

    # get subnet
    supernet.set_active_subnet(subnet_encoding)
    subnet = supernet.get_active_subnet()
    set_quant_mode(subnet)
    subnet.eval()
    subnet = subnet.to(device)

    calibrate_bn_params(subnet, train_loader, POST_BN_CALIB_BATCH_NUM, device=device)

    return subnet


def replace_classifier(subnet, num_classes):
    in_features = subnet.stages.feature_mix[0].conv.out_channels
    subnet.stages.logits = LogitsBlock(in_features=in_features, out_features=num_classes)
    subnet.to(subnet.stages.feature_mix[0].conv.conv.weight.device)


def remove_classifier(subnet):
    subnet.stages.logits = torch.nn.Identity()


def freeze_weight(subnet):
    for param in subnet.parameters():
         param.requires_grad = False


if __name__ == '__main__':
    device = torch.device('mps')
    subnet = get_spaceevo_int8_pretrained_subnet('SEQnet@vnni-A0', '/Users/xudongwang/Downloads/imagenet/', device=device)
    freeze_weight(subnet)
    replace_classifier(subnet, 100)

    model_params = []
    for param in subnet.parameters():
        if param.requires_grad:
            model_params.append(param)

    optimizer = torch.optim.SGD(model_params, lr=1e-2)
    logits = subnet(torch.rand(10, 3, 224, 224).to(device))
    loss = torch.nn.MSELoss()(logits, torch.rand(10, 100).to(device))
    loss.backward()
    optimizer.step()

    print(subnet)