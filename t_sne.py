import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances

# To do:
# - define get_probs()
# - define get_sigmas() https://towardsdatascience.com/understanding-t-sne-by-implementing-2baf3a987ab3

def get_sigmas():
    "obtains sigmas by a search"


def get_probs():
    "obtains probs based on distances and sigmas"


class LowDimCoordinates(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.coordinates = self.add_weight("coordinates", shape=(input_shape, 2), trainable=True)

    def call(self):
        return self.coordinates


@tf.function
def get_Q(Y):
    "Compute and return Q matrix from Y"
    probs = 1 / (1 + euclidean_distances(Y, Y, squared=True))

    return probs / tf.math.reduce_sum(probs)


class TSNELoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        "Compute and return KL divergence of P (y_true) and Q (get_Q(y_pred))"
        Q = get_Q(y_pred)

        return tf.math.reduce_sum(y_true*tf.math.log(y_true / Q))


def t_sne(X, perplexity=30, epochs=1):
    "Compute and return t-SNE of X"

    sample_size, _ = X.shape

    distances = euclidean_distances(X, X, squared=True)

    sigmas = get_sigmas(distances, perplexity)

    P = get_probs(distances, sigmas)

    inputs = tf.keras.layers.Input(shape=sample_size)
    
    output = LowDimCoordinates()(inputs)

    t_sne = tf.keras.models.Model(inputs, output)

    t_sne.compile('adam', TSNELoss())

    t_sne.fit(P, P, batch_size=sample_size, epochs=epochs)

    Y = t_sne()

    return Y