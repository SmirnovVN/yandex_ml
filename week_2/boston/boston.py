import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston


def best_score(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = float('-Inf')
    k = 0
    for i in np.linspace(1, 10, 200):
        clf = KNeighborsRegressor(n_neighbors=5, weights='distance', p=i)
        scores = cross_val_score(clf, X,
                                 y=y, cv=kf, scoring='neg_mean_squared_error')
        mean = scores.mean()
        if score < mean:
            score = mean
            k = i
    return k


def main():
    data = load_boston()
    X = scale(data.data)
    y = data.target

    k = best_score(X, y)

    with open('task_1.txt', 'w') as f:
        f.write("{}".format(k))


if __name__ == '__main__':
    main()
