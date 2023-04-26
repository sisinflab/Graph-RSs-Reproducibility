import pandas as pd
import scipy.sparse as sp
import numpy as np


def dataframe_to_dict(d):
    ratings = d.set_index('userId')[['itemId', 'rating']].apply(lambda x: (x['itemId'], float(x['rating'])), 1) \
        .groupby(level=0).agg(lambda x: dict(x.values)).to_dict()
    return ratings


def build_sparse():
    rows_cols = [(u, i) for u, it in i_train_dict.items() for i in it.keys()]
    rows = [u for u, _ in rows_cols]
    cols = [i for _, i in rows_cols]
    data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                         shape=(len(users), len(items)))
    return data


data = 'allrecipes'

train = pd.read_csv(f'./data/{data}/train.tsv', sep='\t', header=None)
train.columns = ['userId', 'itemId', 'rating']
train_dict = dataframe_to_dict(train)
users = list(train_dict.keys())
items = list({k for a in train_dict.values() for k in a.keys()})
private_users = {p: u for p, u in enumerate(users)}
public_users = {v: k for k, v in private_users.items()}
private_items = {p: i for p, i in enumerate(items)}
public_items = {v: k for k, v in private_items.items()}

i_train_dict = {public_users[user]: {public_items[i]: v for i, v in items.items()}
                for user, items in train_dict.items()}

sp_i_train = build_sparse()
degree_one_users = np.array(sp_i_train.sum(axis=1))[:, 0]
degree_two_users = (np.array(sp_i_train.todense()) * np.array(sp_i_train.sum(axis=0))).sum(axis=1)
degree_three_users = (np.array(sp_i_train.dot(sp_i_train.transpose()).todense()) * np.array(sp_i_train.sum(axis=1))).sum(axis=1)

# degree one
user_groups = np.zeros((len(users)))
q1 = np.quantile(degree_one_users, 0.25)
q2 = np.quantile(degree_one_users, 0.50)
q3 = np.quantile(degree_one_users, 0.75)
user_groups[degree_one_users <= q1] = 0
user_groups[(degree_one_users > q1) & (degree_one_users <= q2)] = 1
user_groups[(degree_one_users > q2) & (degree_one_users <= q3)] = 2
user_groups[(degree_one_users > q3)] = 3
col1, col2 = [], []
for idx, u in enumerate(user_groups.tolist()):
    col1.append(private_users[idx])
    col2.append(int(u))
df = pd.DataFrame([], columns=[1, 2])
df[1] = pd.Series(col1)
df[2] = pd.Series(col2)
df.to_csv(f'./data/{data}/users_deg_1.tsv', sep='\t', header=None, index=None)

# degree two
user_groups = np.zeros((len(users)))
q1 = np.quantile(degree_two_users, 0.25)
q2 = np.quantile(degree_two_users, 0.50)
q3 = np.quantile(degree_two_users, 0.75)
user_groups[degree_two_users <= q1] = 0
user_groups[(degree_two_users > q1) & (degree_two_users <= q2)] = 1
user_groups[(degree_two_users > q2) & (degree_two_users <= q3)] = 2
user_groups[(degree_two_users > q3)] = 3
col1, col2 = [], []
for idx, u in enumerate(user_groups.tolist()):
    col1.append(private_users[idx])
    col2.append(int(u))
df = pd.DataFrame([], columns=[1, 2])
df[1] = pd.Series(col1)
df[2] = pd.Series(col2)
df.to_csv(f'./data/{data}/users_deg_2.tsv', sep='\t', header=None, index=None)

# degree three
user_groups = np.zeros((len(users)))
q1 = np.quantile(degree_three_users, 0.25)
q2 = np.quantile(degree_three_users, 0.50)
q3 = np.quantile(degree_three_users, 0.75)
user_groups[degree_three_users <= q1] = 0
user_groups[(degree_three_users > q1) & (degree_three_users <= q2)] = 1
user_groups[(degree_three_users > q2) & (degree_three_users <= q3)] = 2
user_groups[(degree_three_users > q3)] = 3
col1, col2 = [], []
for idx, u in enumerate(user_groups.tolist()):
    col1.append(private_users[idx])
    col2.append(int(u))
df = pd.DataFrame([], columns=[1, 2])
df[1] = pd.Series(col1)
df[2] = pd.Series(col2)
df.to_csv(f'./data/{data}/users_deg_3.tsv', sep='\t', header=None, index=None)