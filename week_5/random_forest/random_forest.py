import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score


def main():
    data = pd.read_csv('abalone.csv')
    data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
    X = data.loc[:, data.columns != 'Rings']
    y = data['Rings']
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    min_forest = 0
    for i in range(1, 51):
        clf = RandomForestRegressor(random_state=1, n_estimators=i)
        scores = cross_val_score(clf, X, y=y, cv=kf, scoring='r2')
        print(scores)
        if scores.mean() > 0.52:
            min_forest = i
            break

    with open('task_1.txt', 'w') as f:
        f.write("{}".format(min_forest))


if __name__ == '__main__':
    main()