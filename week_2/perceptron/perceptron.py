import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def main():

    train = pd.read_csv('perceptron-train.csv', sep=",", header=None)
    test = pd.read_csv('perceptron-test.csv', sep=",", header=None)

    scaler = StandardScaler()
    X_train = train[[1, 2]]
    X_test = test[[1, 2]]
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    y_train = train[0]
    y_test = test[0]

    clf = Perceptron(random_state=241)
    clf.fit(X_train_scaled, y_train)
    predictions = clf.predict(X_test)
    predictions_scaled = clf.predict(X_test_scaled)
    score = accuracy_score(y_test, predictions)
    score_scaled = accuracy_score(y_test, predictions_scaled)

    with open('task_1.txt', 'w') as f:
        f.write("{0:.3f}".format(score_scaled - score))


if __name__ == '__main__':
    main()
