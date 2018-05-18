import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale


def best_score(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    score = 0
    k = 0
    for i in range(1, 51):
        clf = KNeighborsClassifier(n_neighbors=i)
        scores = cross_val_score(clf, X,
                                 y=y, cv=kf)
        mean = scores.mean()
        if score < mean:
            score = mean
            k = i
    return k, score


def main():
    data = pd.read_csv('wine.data', sep=",", header=None)
    data.columns = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols',
                    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue',
                    'OD280 OD315 of diluted wines', 'Proline']

    k, score = best_score(data[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                                'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                                'Color intensity', 'Hue', 'OD280 OD315 of diluted wines', 'Proline']], data['Class'])

    with open('task_1.txt', 'w') as f:
        f.write("{}".format(k))

    with open('task_2.txt', 'w') as f:
        f.write("{0:.2f}".format(score))

    X = scale(data[['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                    'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                    'Color intensity', 'Hue', 'OD280 OD315 of diluted wines', 'Proline']])

    print(X)

    k, score = best_score(X, data['Class'])

    with open('task_3.txt', 'w') as f:
        f.write("{}".format(k))

    with open('task_4.txt', 'w') as f:
        f.write("{0:.2f}".format(score))


if __name__ == '__main__':
    main()
