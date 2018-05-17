import pandas as pd


def main():
    data = pd.read_csv('./../titanic.csv', index_col='PassengerId')
    with open('task_5.txt', 'w') as f:
        f.write("{0:.2f}".format(data['SibSp'].corr(data['Parch'])))


if __name__ == '__main__':
    main()
