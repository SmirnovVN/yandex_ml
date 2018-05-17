import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier


def main():
    data = pd.read_csv('./../titanic.csv', index_col='PassengerId')
    data.dropna(subset=['Pclass', 'Fare', 'Age', 'Sex', 'Survived'], inplace=True)
    data.replace(['female', 'male'], [0, 1], inplace=True)
    print(data[['Pclass', 'Fare', 'Age', 'Sex', 'Survived']])
    X = data[['Pclass', 'Fare', 'Age', 'Sex']]
    y = data[['Survived']]
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(X, y)
    importance = clf.feature_importances_.copy()
    first = np.argmax(importance)
    importance[first] = 0
    with open('survived.txt', 'w') as f:
        f.write("{} {}".format(X.columns[first], X.columns[np.argmax(importance)]))


if __name__ == '__main__':
    main()
