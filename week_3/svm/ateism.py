import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn import datasets


def main():
    newsgroups = datasets.fetch_20newsgroups(
        subset='all',
        categories=['alt.atheism', 'sci.space']
    )
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(newsgroups.data)
    y = newsgroups.target
    grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cv = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
    gs.fit(X, y)

    parameters = max(zip(gs.cv_results_['mean_test_score'], gs.cv_results_['params']), key=lambda x: x[0])[1]

    clf = SVC(kernel='linear', random_state=241, C=parameters['C'])
    clf.fit(X, y)

    a = sorted(list(zip(map(np.math.fabs, clf.coef_.data), clf.coef_.indices)), reverse=True)[:10]
    feature_mapping = vectorizer.get_feature_names()
    words = [feature_mapping[i[1]] for i in a]

    with open('ateism.txt', 'w') as f:
        f.write("{}".format(' '.join(sorted(words))))


if __name__ == '__main__':
    main()
