import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, cross_val_score, train_test_split
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def main():
    data = pd.read_csv('gbm-data.csv')
    values = data.values
    X = values[:, 1:]
    y = values[:, 0]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
    min_losses = []
    for i in [1, 0.5, 0.3, 0.2, 0.1]:
        clf = GradientBoostingClassifier(n_estimators=250, verbose=True, random_state=241, learning_rate=i)
        clf.fit(X_train, y_train)
        test_deviance = np.zeros(250, dtype=np.float64)
        losses = np.zeros(250, dtype=np.float64)
        min_loss = float('Inf')
        min_idx = 0
        for j, y_pred in enumerate(clf.staged_decision_function(X_test)):
            test_deviance[j] = clf.loss_(y_test, y_pred)
            y_pred_s = sigmoid(y_pred)
            losses[j] = log_loss(y_test, y_pred_s)
            if min_loss > losses[j]:
                min_loss = losses[j]
                min_idx = j
        min_losses.append((min_loss, min_idx))
        plt.figure()
        plt.plot(losses, 'r', linewidth=2)
        plt.plot(test_deviance, 'g', linewidth=2)
        plt.legend(['test', 'train'])
        plt.savefig('./' + str(i) + '.png')

    with open('task_2.txt', 'w') as f:
        f.write("{0:.2f} {1:.2f}".format(min_losses[3][0], min_losses[3][1]))

    min_loss, min_idx = min(min_losses)

    clf = RandomForestClassifier(n_estimators=min_idx, random_state=241)
    clf.fit(X_train, y_train)
    loss = log_loss(y_test, clf.predict_proba(X_test))
    with open('task_3.txt', 'w') as f:
        f.write("{0:.2f}".format(loss))


if __name__ == '__main__':
    main()
