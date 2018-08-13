"""
tools to convert specified type
"""
import torch as t
import numpy as np


def tonumpy(data):
    if isinstance(data, np.ndarray):
        return data
    if isinstance(data, t._TensorBase):
        return data.cpu().numpy()
    if isinstance(data, t.autograd.Variable):
        return tonumpy(data.data)
