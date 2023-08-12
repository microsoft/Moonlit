import math

from nn_meter.predictor.quantize_block_predictor import BlockLatencyPredictor
from modules.modeling.network import QNetworkConfig
from modules.modeling.blocks import BaseBlockConfig


available_platform = [
    'tflite27_cpu_int8',
    'onnxruntime_int8'
]


class LatencyPredictor:

    def __init__(self, platform) -> None:
        self.platform = platform
        self.block_predictor = BlockLatencyPredictor(platform)

    def predict_block(self, block_config: BaseBlockConfig, resolution):
        return self.block_predictor.get_latency([self._get_block_args(block_config, resolution)])

    def _get_block_args(self, block_config: BaseBlockConfig, resolution):
        if self.platform == 'onnx_lut' and 'MobileNetV1Dual' in block_config.name: # tmp workaround for mobilenetv1dualblock
            expand_ratio = 0
        elif 'resnet' not in block_config.name.lower() and block_config.expand_ratio and block_config.expand_ratio * 10 % 10 == 0:
            expand_ratio = int(block_config.expand_ratio)
        else:
            expand_ratio = block_config.expand_ratio
        if self.platform == 'onnx_lut' and 'logits' in block_config.name.lower():
            resolution = 1
            block_config.stride = 0
            block_config.activation = 'relu'
        return dict(
            name=self._convert_name(block_config),
            hw=resolution,
            cin=block_config.cin,
            cout=block_config.cout,
            kernel_size=block_config.kernel_size,
            expand_ratio=expand_ratio,
            stride=block_config.stride,
            activation=block_config.activation,
            # activation=block_config.activation.replace('relu6', 'relu'),
        )
    
    def predict_subnet(self, subnet_config: QNetworkConfig, verbose=False):
        hw = subnet_config.resolution
        block_args_list = []
        for block_config in subnet_config.as_block_config_list():
            block_args_list.append(self._get_block_args(block_config, hw))
            hw = math.ceil(hw / (block_config.stride or 1))
        if not verbose:
            rv = self.block_predictor.get_latency(block_args_list)
        else:
            rv = 0
            for block_config, block_args in zip(subnet_config.as_block_config_list(), block_args_list):
                lat = self.block_predictor.get_latency([block_args])
                print(f'{block_config},{round(lat, 2)}')
                rv += lat
        return rv

    def _convert_dynamic_name(self, name):
        if name.startswith('Dynamic'):
            name = name[7:]
        return name

    def _convert_name(self, block_config: BaseBlockConfig) -> str:
        name = self._convert_dynamic_name(block_config.name)
        return name
