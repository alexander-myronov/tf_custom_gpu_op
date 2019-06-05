import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

x_train = x_train[:, :, :, np.newaxis]
x_test = x_test[:, :, :, np.newaxis]

# x_train = x_train[:128]
# y_train = y_train[:128]

# x_test = x_test[:128]
# y_test = y_test[:128]


class MedianPooling2D(tf.keras.layers.Layer):
    def __init__(self, k=3):
        super(MedianPooling2D, self).__init__()
        # self.num_outputs = num_outputs
        self.k = k
        self.n_patches_w = None
        self.n_patches_h = None
        self.channels = None

    def compute_output_shape(self, input_shape):
        # print(type(input_shape))
        strides = rates = [1, 1, 1, 1]
        # batch_size = tf.shape(x)[0]
        self.n_channels = input_shape[-1]
        self.n_patches_h = (input_shape[1] - self.k) // strides[1] + 1
        self.n_patches_w = (input_shape[2] - self.k) // strides[2] + 1
        output_shape = tf.TensorShape(dims=(None, self.n_patches_h, self.n_patches_w, self.n_channels))
        # print(output_shape)
        return output_shape

    def call(self, input, **kwargs):
        x = input
        k = self.k

        strides = rates = [1, 1, 1, 1]
        patches = tf.extract_image_patches(x, [1, k, k, 1], strides, rates, 'VALID')
        batch_size = tf.shape(x)[0]
        # n_channels = tf.shape(x)[-1]
        # n_patches_h = (tf.shape(x)[1] - k) // strides[1] + 1
        # n_patches_w = (tf.shape(x)[2] - k) // strides[2] + 1
        # n_patches = tf.shape(patches)[-1] // n_channels
        patches = tf.reshape(patches, [batch_size, k, k, self.n_patches_h * self.n_patches_w, self.n_channels])
        medians = tf.contrib.distributions.percentile(patches, 50, axis=[1, 2])
        medians = tf.reshape(medians, (batch_size, self.n_patches_h, self.n_patches_w, self.n_channels))
        return medians
        # return tf.matmul(input, self.kernel)


class MeanTopKPooling2D(tf.keras.layers.Layer):
    def __init__(self, k=3):
        super(MeanTopKPooling2D, self).__init__()
        # self.num_outputs = num_outputs
        self.k = k
        self.n_patches_w = None
        self.n_patches_h = None
        self.n_channels = None

    def compute_output_shape(self, input_shape):
        # print(type(input_shape))
        strides = rates = [1, 1, 1, 1]
        # batch_size = tf.shape(x)[0]
        self.n_channels = input_shape[-1]
        self.n_patches_h = (input_shape[1] - self.k) // strides[1] + 1
        self.n_patches_w = (input_shape[2] - self.k) // strides[2] + 1
        output_shape = tf.TensorShape(dims=(input_shape[0], self.n_patches_h, self.n_patches_w, self.n_channels))
        # print(output_shape)
        # output_shape = tf.TensorShape((None, 3, 3, 10))
        return output_shape

    def call(self, inputs, **kwargs):
        x = inputs
        k = self.k

        strides = rates = [1, 1, 1, 1]

        input_shape = x.shape
        print(input_shape)

        self.n_channels = input_shape[-1]
        self.n_patches_h = (input_shape[1] - self.k) // strides[1] + 1
        self.n_patches_w = (input_shape[2] - self.k) // strides[2] + 1

        patches = tf.extract_image_patches(x, [1, k, k, 1], strides, rates, 'VALID')
        batch_size = tf.shape(x)[0]
        # patches = tf.reshape(patches, [batch_size, k, k, self.n_patches_h, self.n_patches_w, self.n_channels])
        patches = tf.reshape(patches, [batch_size, k * k, self.n_patches_h,  self.n_patches_w, self.n_channels])

        patches = tf.transpose(patches, [0, 2, 3, 4, 1])
        values, indices = tf.math.top_k(patches, k=4, sorted=False)
        # print(values.shape)
        mean_top_k = tf.reduce_mean(values, axis=4)
        print(mean_top_k.shape)
        return mean_top_k
        # raise Exception()


model = tf.keras.models.Sequential([
    # tf.keras.layers.Flatten(input_shape=(28, 28)),
    # tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Conv2D(10, kernel_size=3, activation=tf.nn.relu),
    # tf.keras.layers.MaxPool2D(pool_size=(3, 3)),
    MeanTopKPooling2D(k=3),
    tf.keras.layers.Flatten(),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1, batch_size=32, shuffle=True)
# print('test set accuracy', model.evaluate(x_test, y_test, batch_size=32)[1])
