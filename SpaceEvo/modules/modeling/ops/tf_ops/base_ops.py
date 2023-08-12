from collections import OrderedDict
from typing import Callable, Optional

import tensorflow as tf
from tensorflow.keras import layers


class HSigmoid(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.relu6 = layers.ReLU(6)

    def call(self, x):
        return self.relu6(x + 3.) * (1. / 6.)

class HSwish(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.relu6 = layers.ReLU(6)

    def call(self, x):
        return x * self.relu6(x + 3.) * (1. / 6.)

class Relu6(tf.keras.Model):
    
    def __init__(self):
        super().__init__()
        self.relu6 = layers.ReLU(6)

    def call(self, x):
        return self.relu6(x)


class Swish(tf.keras.Model):

    def __init__(self) -> None:
        super().__init__()

    def call(self, x):
        return tf.keras.activations.swish(x)