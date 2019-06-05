import random

import numpy as np
import pandas as pd
import tensorflow as tf
import itertools

from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from matplotlib import pyplot as plt

def random_seq(n):
    return ''.join((random.choice(['A', 'B']) for _ in range(n)))

def seq_to_array(seq, maxlen=12):
    arr = np.zeros((maxlen, 2), dtype='float32')
    for i, l in enumerate(seq):
        if l == 'A':
            arr[i, 0] = 1.0
        else:
            arr[i, 1] = 1.0
    return arr



# N = 1000
df = pd.DataFrame(index=np.arange(10000), columns=['seq', 'y'])
for i in tqdm(range(len(df))):
    seq = random_seq(random.randint(5, 12))
    df.loc[i, 'seq'] = seq
    # df.loc[i, 'y'] = 1.0

df.loc[:, 'y'] = (df.seq.str.contains('ABA') | (df.seq.str.count('AAB') >= 2))

c1 = df.seq.str.contains('ABA').astype(float)
c2 = (df.seq.str.count('AAB') >= 2).astype(float)
#
print(confusion_matrix(c1, c2))
# print((c1 & c2).sum())
# print(c1.sum(), c2.sum())
# print(pd.crosstab(c1, c2, margins=True))
# help(pd.crosstab)

print(df.y.value_counts())
#
# exit()
# x

#
#



X = df.seq.apply(seq_to_array)
X = np.dstack(X)
X = np.rollaxis(X, 2)
# X = X[:, :, np.newaxis]
print(X.shape)
# print(X[0])
# print(df.iloc[0])
# exit()
y = df.y.values

x_train = X[:int(len(X) * 0.9)]
y_train = y[:int(len(X) * 0.9)]

x_val = X[int(len(X) * 0.9):]
y_val = y[int(len(X) * 0.9):]

model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Conv1D(1, kernel_size=3, activation=tf.nn.relu, use_bias=False, padding='valid'),
    # tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
    tf.keras.layers.GlobalMaxPool1D(),
    # tf.keras.layers.GlobalAveragePooling1D(),
    # tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation=tf.nn.sigmoid, use_bias=False)
])

# input = tf.keras.layers.Input(shape=(10, 1))
#
# h= tf.keras.layers.Conv1D(2, kernel_size=3, activation=tf.nn.relu)(input)
# h_mean = tf.keras.layers.GlobalAveragePooling1D()(h)
# h_max = tf.keras.layers.GlobalMaxPooling1D()(h)
# h = tf.keras.layers.Concatenate()([h_mean, h_max])
# o = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
#
#
# model =tf.keras.models.Model([input], )

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])



# model.build()

# print(model.summary())

model.fit(x_train, y_train, epochs=10, batch_size=16, shuffle=True, validation_data=(x_val, y_val), verbose=2)

trimers = ['BAA', 'AAB', 'AAA', 'ABA', 'BBB', 'BAB', 'BBA', 'ABB']

series = pd.Series(index=trimers)
x_trimers = np.dstack(pd.Series(series.index).apply(lambda s: seq_to_array(s, 3)))
x_trimers = np.rollaxis(x_trimers, 2)
o = model.layers[0](x_trimers)
f = tf.keras.backend.function([], [o])
y_trimers = f([])[0]
print(y_trimers.shape)
series = pd.Series(y_trimers.flatten(), index=trimers)
print(series.sort_values(ascending=False))

# f.eva

weights = model.get_weights()
for a in weights:
    print(a)
    plt.imshow(a[:,:,0])
    plt.colorbar()
    plt.show()
    print(a.shape)

# for l in model.layers:
#     print(l)
#     print(l, l.output_shape)
#
# print(model.summary())