import pandas as pd

dataset = 'gowalla'
train_filename = 'train.txt'
test_filename = 'test.txt'

rows, cols = [], []

with open('./data/{0}/{1}'.format(dataset, train_filename), 'r') as f:
    for line in f:
        all_elements = line.split(' ')
        if '\n' not in all_elements:
            for el in all_elements[1:]:
                rows.append(int(all_elements[0]))
                cols.append(int(el))
        else:
            print(f'User: {all_elements[0]} does not have items in the train.')

train = pd.concat([pd.Series(rows), pd.Series(cols)], axis=1)
train.columns = ['user', 'item']

rows, cols = [], []

with open('./data/{0}/{1}'.format(dataset, test_filename), 'r') as f:
    for line in f:
        all_elements = line.split(' ')
        if '\n' not in all_elements:
            for el in all_elements[1:]:
                rows.append(int(all_elements[0]))
                cols.append(int(el))
        else:
            print(f'User: {all_elements[0]} does not have items in the test.')
test = pd.concat([pd.Series(rows), pd.Series(cols)], axis=1)
test.columns = ['user', 'item']

train['user_id'] = train.groupby('user').grouper.group_info[0]
train['item_id'] = train.groupby('item').grouper.group_info[0]

test = test[test['user'].isin(train['user'])]
test = test[test['item'].isin(train['item'])]

train[['user_id', 'item_id']].to_csv('./data/{0}/train.tsv'.format(dataset), sep='\t', header=None, index=None)
test[['user', 'item']].to_csv('./data/{0}/test.tsv'.format(dataset), sep='\t', header=None, index=None)
