import numpy as np
import tensorflow as tf

mod = tf.load_op_library('./cuda_op_kernel.so')
with tf.Session() as sess:
    # start = np.arange(16, dtype=float)
    start = np.array([[[1, 2, 3, 4], [3, 2, 1, 5], [3, 2, 1, 3]], [[4, 5, 6, 7], [4, 5, 6, 8], [7, 8, 9, 10]]])
    # start = start[0]
    # np.random.shuffle(start)
    print(start.shape)
    print(list(mod.add_one(start).eval()))



