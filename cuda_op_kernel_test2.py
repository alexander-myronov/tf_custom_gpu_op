import pickle
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

mod = tf.load_op_library('./cuda_op_kernel.so')

inputs = np.random.uniform(0, 100, size=(1024, 64, 512))

with tf.Session() as sess:
    res = mod.max_kernel(inputs).eval()
    assert np.allclose(res, inputs.max(axis=2))
