# based on https://github.com/lukemelas/EfficientNet-PyTorch/tree/master/efficientnet_pytorch

__version__ = "0.7.1"
from .model import EfficientNet, VALID_MODELS
from .quant_model import QEfficientNet
from .utils import (
    GlobalParams,
    BlockArgs,
    BlockDecoder,
    efficientnet,
    get_model_params,
)