# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.import random
import numpy as np

def mutate_dims(choices, prob=1):
    assert 0. <= prob <= 1.

    if random.random() < prob:
        return random.choice(choices)
    else:
        return min(choices)

def softmax(x, to_list: bool = False):
    y = np.exp(x) / np.sum(np.exp(x), axis=0)

    if to_list:
        y = y.tolist()
    return y