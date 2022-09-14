import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from tensorflow.keras import Model

import numpy as np


class ConvLayer(Layer):
    def __init__(self, filters, kernel_size, padding: str = "same"):
        super(ConvLayer, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding

        self.conv = layers.Conv2D(
            filters=self.filters,
            kernel_size=self.kernel_size,
            padding=self.padding,
        )
        self.bn = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.pool = layers.MaxPool2D((2, 2))

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class Prototypical_Network(Model):
    def __init__(self, w: int = 28, h: int = 28, c: int = 1):
        super(Prototypical_Network, self).__init__()
        self.w, self.h, self.c = w, h, c

        self.encoder = tf.keras.Sequential(
            [
                ConvLayer(64, 3, "same"),
                ConvLayer(64, 3, "same"),
                ConvLayer(64, 3, "same"),
                ConvLayer(64, 3, "same"),
                layers.Flatten(),
            ]
        )

    def call(self, support, query):
        def _get_l2_distance(x, y):
            # print(a.shape) # (5, 64)
            # print(b.shape) # (25, 64)
            # - > (25, 5, 64)
            n_x, n_y = x.shape[0], y.shape[0]
            a = tf.tile(tf.expand_dims(x, 1), (1, n_y, 1))
            b = tf.tile(tf.expand_dims(y, 0), (n_x, 1, 1))

            return tf.reduce_mean(tf.math.pow(a - b, 2), axis=2)

        n_way = support.shape[0]
        n_support = support.shape[1]
        n_query = query.shape[1]

        reshaped_s = tf.reshape(
            support, (n_way * n_support, self.w, self.h, self.c)
        )
        reshaped_q = tf.reshape(
            query, (n_way * n_query, self.w, self.h, self.c)
        )

        # Embeddings are in the shape of (n_support+n_query, 64)
        embeddings = self.encoder(tf.concat([reshaped_s, reshaped_q], axis=0))

        # Support prototypes are in the shape of (n_way, n_support, 64)
        s_prototypes = tf.reshape(
            embeddings[: n_way * n_support],
            [n_way, n_support, embeddings.shape[-1]],
        )
        # Find the average of prototypes for each class in n_way
        s_prototypes = tf.math.reduce_mean(s_prototypes, axis=1)
        # Query embeddings are the remainding embeddings
        q_embeddings = embeddings[n_way * n_support :]

        loss = 0.0
        acc = 0.0
        ############### Your code here ###################
        # TODO: finish implementing this method.
        # For a given task, calculate the Euclidean distance
        # for each query embedding and support prototypes.
        # Then, use these distances to calculate
        # both the loss and the accuracy of the model.
        # HINT: you can use tf.nn.log_softmax()

        # 데이터 형태를 맞춰주고, 정확도 계산을 위한 one_hot vector로 변환한다.
        y = np.tile(np.arange(n_way)[:, np.newaxis], (1, n_query))
        # Convert to one-hot vector from index labels.
        y_true = tf.cast(tf.one_hot(y, n_way), tf.float32)

        # Calculate the Euclidean distance between query embeddings and support prototypes.
        distances = _get_l2_distance(q_embeddings, s_prototypes)
        # log softmax of calculated distances
        log_p_y = tf.nn.log_softmax(-distances, axis=-1)
        # 예측값 (=y_pred)
        y_hat = tf.reshape(log_p_y, [n_way, n_query, -1])

        # Loss should be the scalar value.
        loss = -tf.reduce_mean(
            tf.reshape(
                tf.reduce_sum(tf.multiply(y_true, y_hat), axis=-1), [-1]
            )
        )

        # Calculate the accruacy of prototypical networks.
        prediction = tf.argmax(y_hat, axis=-1)
        equality = tf.math.equal(tf.cast(prediction, tf.float32), y)
        acc = tf.reduce_mean(tf.cast(equality, tf.float32))

        return loss, acc
