import pandas as pd
import re


def main():
    data = pd.read_csv('./../titanic.csv', index_col='PassengerId')
    name = data[data['Sex'] == 'female']['Name'].apply(
        lambda x: pd.Series({'First name': re.sub(r"[^A-Za-z]+", '', x.strip().split(' ')[-1])})) \
        .groupby('First name').size().reset_index(name='counts').sort_values(['counts'], ascending=False) \
        .head(1)
    with open('task_6.txt', 'w') as f:
        f.write(name.get_value(name.index[0], 'First name'))


if __name__ == '__main__':
    main()
