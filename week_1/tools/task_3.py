import pandas as pd


def main():
    data = pd.read_csv('./../titanic.csv', index_col='PassengerId')
    with open('task_3.txt', 'w') as f:
        f.write("{0:.2f}".format(100*len(data[(data['Pclass'] == 1)])/len(data)))


if __name__ == '__main__':
    main()
