# Personal implementation of the t-SNE algorithm

import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import minimize_scalar


def get_perplexity(x):
    "Compute perplexity"
    return 2**(-np.sum(x*np.log2(x)))


def get_probs(distances, sigmas):
    "obtains probs based on distances and sigmas"
    n = len(sigmas)
    probs = np.exp(-distances / (2*np.repeat(sigmas, n).reshape(n, n)))

    probs = probs / np.repeat(np.sum(probs, axis=1), n).reshape(n, n)

    return (probs + probs.transpose()) / (2*n)


def get_sigmas(distances, perplexity):
    "obtains sigmas by a search"
    n = distances.shape[0]
    sigmas = np.zeros(shape=n)
    for i in range(n):
        func = lambda x: np.abs(get_perplexity(get_probs(distances, np.ones(n)*x*x)[i, :]) - perplexity)
        sigmas[i] = minimize_scalar(func, bounds=[np.finfo(np.float32).eps, 1000], method='bounded').x
        print(f'Finding sigmas: {int(np.round(i*100/n))}%')
    return sigmas


class LowDimCoordinates(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.coordinates = self.add_weight("coordinates", shape=(input_shape[1], 2), trainable=True)

    def call(self, _):
        return self.coordinates


@tf.function
def get_Q(Y):
    "Compute and return Q matrix from Y"
    R = tf.reshape(tf.reduce_sum(Y*Y, 1), [-1, 1])
    D = R - 2*tf.matmul(Y, tf.transpose(Y)) + tf.transpose(R)
    probs = 1 / (1 + D)

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

    Y = t_sne(P)

    return Y
