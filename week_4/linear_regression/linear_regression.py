import pandas as pd
from numpy.core.defchararray import lower
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sympy.physics.quantum.matrixutils import scipy


def main():
    train = pd.read_csv('salary-train.csv')
    test = pd.read_csv('salary-test-mini.csv')
    train['LocationNormalized'].fillna('nan', inplace=True)
    train['ContractTime'].fillna('nan', inplace=True)

    enc = DictVectorizer()
    X_train_categ = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))
    X_test_categ = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))

    train['FullDescription'] = train['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    train['FullDescription'] = train['FullDescription'].apply(lower)
    test['FullDescription'] = test['FullDescription'].replace('[^a-zA-Z0-9]', ' ', regex=True)
    test['FullDescription'] = test['FullDescription'].apply(lower)

    vectorizer = TfidfVectorizer(min_df=5)
    X_train = vectorizer.fit_transform(train['FullDescription'])
    X_test = vectorizer.transform(test['FullDescription'])

    X_train = scipy.sparse.hstack([X_train, X_train_categ])
    X_test = scipy.sparse.hstack([X_test, X_test_categ])

    y_train = train['SalaryNormalized']

    clf = Ridge(alpha=1, random_state=241)
    clf.fit(X_train, y_train)
    predict = clf.predict(X_test)

    with open('linear_regression.txt', 'w') as f:
        f.write("{0:.2f} {1:.2f}".format(predict[0], predict[1]))


if __name__ == '__main__':
    main()
