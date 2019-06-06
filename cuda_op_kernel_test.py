import pickle
import time

import numpy as np
import tensorflow as tf
from tqdm import tqdm

# import pandas
mod = tf.load_op_library('./cuda_op_kernel.so')

results = []

for rep in range(7):

    for size in [256, 1024, 2048]:

        inputs = [np.random.uniform(0, 100, size=(size, 64, 512)) for _ in tqdm(range(10))]

    # expected_results = [inp.max(ax)]


        gpu_results = []

        with tf.Session() as sess:
            # start = np.arange(16, dtype=float)

            ph = tf.placeholder('float', shape=(size, 64, 512))
            res = mod.add_one(ph)

            start = time.time()
            for inp in inputs:
                res_cpu = sess.run(res, feed_dict={ph: inp})
                gpu_results.append(res_cpu)
            gpu_time = time.time() - start
            results.append(dict(method='GPU Custom', time=gpu_time, size=size))

            print('gpu time=%fs' % gpu_time)

        gpu_results_tf = []

        with tf.Session() as sess:
            # start = np.arange(16, dtype=float)

            ph = tf.placeholder('float', shape=(size, 64, 512))
            res = tf.reduce_max(ph, axis=2)

            start = time.time()
            for inp in inputs:
                res_cpu = sess.run(res, feed_dict={ph: inp})
                gpu_results.append(res_cpu)
            gpu_time_tf = time.time() - start
            results.append(dict(method='GPU TF', time=gpu_time_tf, size=size))
            print('gpu TF time=%fs' % gpu_time_tf)

        cpu_results = []
        start = time.time()
        for inp in inputs:
            cpu_results.append(inp.max(axis=2))
        cpu_time = time.time() - start
        print('cpu time=%fs' % cpu_time)
        results.append(dict(method='CPU', time=cpu_time, size=size))


        for gpu_r, cpu_r in zip(gpu_results, cpu_results):
            assert np.allclose(gpu_r, cpu_r)

with open('results.pkl', 'wb') as f:
    pickle.dump(results, f, -1)


# print(pd.DataFrame(results))

    # print(actual[0, :10])

    # print(expected[0, :10])


