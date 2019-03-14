'''
This is a reference implementation of multiclass logistic regression. No high performance or usage for high dimension
problems expected
'''

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict(W, b, X):
    return sigmoid(np.add(np.dot(X, W), b))


def cross_entropy(W, b, X, Y):
    y_pred = predict(W, b, X)
    return -np.sum(np.multiply(Y, np.log(y_pred))) / Y.shape[0]


def train_n_iter(W, b, X, Y, alpha, n_iter):
    l_rate = alpha / X.shape[0]
    for _ in range(n_iter):
        y_pred = predict(W, b, X)
        delta = Y - y_pred
        W = W + l_rate * np.dot(X.T, delta)
        b = b + l_rate * np.sum(delta, axis=0)
    return W, b


if __name__ == '__main__':
    X = np.array([[0, 1],
                  [0, 2],
                  [0, 3],
                  [1, 2],
                  [1, 3],
                  [2, 3],
                  [1, 0],
                  [2, 0],
                  [2, 1],
                  [3, 0],
                  [3, 1],
                  [3, 2]], dtype=np.float)

    Y = np.array([[1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [1, 0],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1],
                  [0, 1]])

    W = np.random.rand(2, 2)
    b = np.random.rand(2)

    for epoch in range(10):
        W, b = train_n_iter(W, b, X, Y, 0.5, 1000)
        print('Epoch:', epoch, 'cross-entropy:', cross_entropy(W, b, X, Y))
