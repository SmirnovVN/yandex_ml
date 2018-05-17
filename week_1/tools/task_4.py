import pandas as pd


def main():
    data = pd.read_csv('./../titanic.csv', index_col='PassengerId')
    with open('task_4.txt', 'w') as f:
        f.write("{0:.2f} {1}".format(data['Age'].mean(), data['Age'].median()))


if __name__ == '__main__':
    main()
