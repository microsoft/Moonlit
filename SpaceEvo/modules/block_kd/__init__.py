from .teacher.efficientnet import EfficientNet, QEfficientNet
from .teacher.regnet import regnet_y_1_6gf
from .manager import BlockKDManager, StagePlusProj


def get_efficientnet_teacher_model(name='efficientnet-b7'):
    teacher = EfficientNet.from_pretrained(name, drop_connect_rate=0, image_size=224, dropout_rate=0)
    teacher.eval()
    return teacher


def get_quant_efficientnet_teacher_model(name='efficientnet-b7'):
    teacher = QEfficientNet.from_pretrained(name, drop_connect_rate=0, image_size=224, dropout_rate=0)
    teacher.eval()
    return teacher


def get_regnet_teacher_model():
    teacher = regnet_y_1_6gf(pretrained=True)
    teacher.eval()
    return teacher