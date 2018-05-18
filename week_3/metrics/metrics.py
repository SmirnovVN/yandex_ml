import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    precision_recall_curve


def main():
    classification = pd.read_csv('classification.csv', sep=",")

    true = classification['true']
    pred = classification['pred']
    tn, fp, fn, tp = confusion_matrix(true, pred).ravel()

    with open('task_1.txt', 'w') as f:
        f.write("{} {} {} {}".format(tp, fp, fn, tn))

    with open('task_2.txt', 'w') as f:
        f.write("{} {} {} {}".format(accuracy_score(true, pred), precision_score(true, pred),
                                     recall_score(true, pred), f1_score(true, pred)))

    scores = pd.read_csv('scores.csv', sep=",")
    true = scores['true']

    columns = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']
    best = 0
    best_clf = None
    for column in columns:
        score = roc_auc_score(true, scores[column])
        if best < score:
            best = score
            best_clf = column

    with open('task_3.txt', 'w') as f:
        f.write("{}".format(best_clf))

    columns = ['score_logreg', 'score_svm', 'score_knn', 'score_tree']
    best = 0
    best_clf = None
    for column in columns:
        curve = precision_recall_curve(true, scores[column])
        for precision, recall in zip(curve[0], curve[1]):
            if best < precision and recall > 0.7:
                best = precision
                best_clf = column

    with open('task_4.txt', 'w') as f:
        f.write("{}".format(best_clf))


if __name__ == '__main__':
    main()
