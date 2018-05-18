import pandas as pd
from sklearn.svm import SVC


def main():
    data = pd.read_csv('svm-data.csv', sep=",", header=None)

    X = data[[1, 2]]
    y = data[0]

    clf = SVC(random_state=241, C=100000, kernel='linear')
    clf.fit(X, y)

    with open('svm.txt', 'w') as f:
        f.write("{}".format(' '.join(map(lambda x: str(x+1), clf.support_))))


if __name__ == '__main__':
    main()
