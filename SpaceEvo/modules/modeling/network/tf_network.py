from typing import Dict, List, OrderedDict, Tuple

import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Sequential, layers

from modules.modeling.blocks.tf_blocks import BaseBlock
from modules.modeling.network import QNetworkConfig


BATCH_SIZE = 1
CIN = 3


class TfNetwork(Sequential):
    
    def __init__(self, stages: List[Sequential], resolution) -> None:
        input_layer = layers.InputLayer([resolution, resolution, CIN], BATCH_SIZE)
        super().__init__(layers=[input_layer, *stages], name='TF-Subnet')
        self.resolution = resolution
        
    @property
    def config(self) -> QNetworkConfig:
        stage_configs = []
        for stage in self.layers:
            blocks = []
            for block in stage.layers:
                assert isinstance(block, BaseBlock)
                blocks.append(block.config)
            stage_configs.append(blocks)
        return QNetworkConfig(stage_configs, self.resolution)