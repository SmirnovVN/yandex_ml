import pandas as pd
from numpy import corrcoef
from sklearn.decomposition import PCA


def main():
    train = pd.read_csv('close_prices.csv')
    test = pd.read_csv('djia_index.csv')
    del train["date"]
    columns = train.columns
    pca = PCA(n_components=10)
    transformed = pca.fit_transform(train)

    it = iter(pca.explained_variance_ratio_)
    ratio = 0
    count = 0
    while ratio < 0.9:
        ratio += next(it)
        count += 1

    with open('task_1.txt', 'w') as f:
        f.write("{}".format(count))

    with open('task_2.txt', 'w') as f:
        f.write("{0:.2f}".format(corrcoef(transformed[:, 0], test['^DJI'])[1][0]))

    component = pca.components_[0]

    with open('task_3.txt', 'w') as f:
        f.write("{}".format(columns[component.argmax()]))


if __name__ == '__main__':
    main()
