import numpy as np
import pandas as pd
from numpy.dual import norm
from sklearn.metrics import roc_auc_score


def update_weights(w, X, y, k, C, reg):
    length = len(X)
    old_w = w.copy()
    for i in range(length):
        part = k * y[i] * (1 - 1 / (1 + np.exp(-y[i] * (old_w[0] * X[1][i] + old_w[1] * X[2][i])))) / length
        w[0] += X[1][i] * part
        w[1] += X[2][i] * part
    if reg:
        w -= k * C * old_w


def gd(w, X, y, k, C, reg, max_iter):
    i = 0
    while i < max_iter:
        prev = w.copy()
        update_weights(w, X, y, k, C, reg)
        if (norm(prev - w, ord=2) < 0.00001).min():
            break
        i += 0
    if i < max_iter:
        return True
    else:
        return False


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(w, X):
    return sigmoid(X.dot(w.T))


def main():
    data = pd.read_csv('data-logistic.csv', sep=",", header=None)
    X = data[[1, 2]]
    y = data[0]
    w = np.array([0., 0.])
    print(gd(w, X, y, 0.1, 10, True, 10000))
    with_reg = roc_auc_score(y, predict(w, X))
    w = np.array([0., 0.])
    print(gd(w, X, y, 0.1, 10, False, 10000))
    without_reg = roc_auc_score(y, predict(w, X))
    with open('gd.txt', 'w') as f:
        f.write("{1:.3f} {0:.3f}".format(with_reg, without_reg))


if __name__ == '__main__':
    main()
