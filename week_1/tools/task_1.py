import pandas as pd


def main():
    data = pd.read_csv('./../titanic.csv', index_col='PassengerId')
    counts = data['Sex'].value_counts()
    with open('task_1.txt', 'w') as f:
        f.write("{} {}".format(counts.get_value('male'), counts.get_value('female')))


if __name__ == '__main__':
    main()
