import numpy as np
import tensorflow as tf

mod = tf.load_op_library('./cuda_op_kernel.so')
with tf.Session() as sess:
    start = np.arange(16, dtype=float)
    # np.random.shuffle(start)
    print(start)
    print(list(mod.add_one(start).eval()))



