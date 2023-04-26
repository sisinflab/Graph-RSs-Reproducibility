import pandas as pd
from operator import itemgetter
from types import SimpleNamespace
import typing as t
from scipy import sparse as sp
import numpy as np
import random
from ast import literal_eval as make_tuple

np.random.seed(42)
random.seed(42)

"""
prefiltering:
    strategy: global_threshold|user_average|user_k_core|item_k_core|iterative_k_core|n_rounds_k_core|cold_users
    threshold: 3|average
    core: 5
    rounds: 2
"""


class NegativeSampler:

    @staticmethod
    def sample(ns: SimpleNamespace, public_users: t.Dict, public_items: t.Dict, private_users: t.Dict,
               private_items: t.Dict, i_train: sp.csr_matrix,
               val: t.Dict = None, test: t.Dict = None):

        val_negative_items, val_negative_items_set = NegativeSampler.process_sampling(ns, public_users, public_items, private_users,
                                                              private_items, i_train,
                                                              test, validation=True) if val != None else (None, None)

        test_negative_items, test_negative_items_set = NegativeSampler.process_sampling(ns, public_users, public_items, private_users,
                                                              private_items, i_train,
                                                               test) if test != None else (None, None)

        if val_negative_items_set and test_negative_items_set:
            selected_items = val_negative_items_set.union(test_negative_items_set)

        elif val_negative_items_set:
            selected_items = val_negative_items_set

        elif test_negative_items_set:
            selected_items = test_negative_items_set

        else:
            selected_items = None

        return (val_negative_items, test_negative_items, selected_items) if val_negative_items else (test_negative_items, test_negative_items, selected_items)

    @staticmethod
    def process_sampling(ns: SimpleNamespace, public_users: t.Dict, public_items: t.Dict, private_users: t.Dict,
                         private_items: t.Dict, i_train: sp.csr_matrix,
                         test: t.Dict, validation=False) -> sp.csr_matrix:

        i_test = [(public_users[user], public_items[i])
                  for user, items in test.items() if user in public_users.keys()
                  for i in items.keys() if i in public_items.keys()]
        rows = [u for u, _ in i_test]
        cols = [i for _, i in i_test]
        i_test = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                               shape=(len(public_users.keys()), len(public_items.keys())))

        # quando è fixed questo non viene usato
        # candidate_negatives = ((i_test + i_train).astype('bool') != True)
        ns = ns.negative_sampling

        strategy = getattr(ns, "strategy", None)

        if strategy == "random":
            num_items = getattr(ns, "num_items", None)
            file_path = getattr(ns, "file_path", None)
            if num_items is not None:
                if str(num_items).isdigit():
                    negative_items = NegativeSampler.sample_by_random_uniform(candidate_negatives, num_items)

                    nnz = negative_items.nonzero()
                    old_ind = 0
                    basic_negative = []
                    for u, v in enumerate(negative_items.indptr[1:]):
                        basic_negative.append([(private_users[u],), list(map(private_items.get, nnz[1][old_ind:v]))])
                        old_ind = v

                    with open(file_path, "w") as file:
                        for ele in basic_negative:
                            line = str(ele[0]) + '\t' + '\t'.join(map(str, ele[1]))+'\n'
                            file.write(line)

                    pass
                else:
                    raise Exception("Number of negative items value not recognized")
            else:
                raise Exception("Number of negative items option is missing")
        elif strategy == "fixed":
            files = getattr(ns, "files", None)
            if files is not None:
                if not isinstance(files, list):
                    files = [files]
                file_ = files[0] if validation == False else files[1]
                negative_items = NegativeSampler.read_from_files(public_users, public_items, file_)
            pass
        else:
            raise Exception("Missing strategy")

        return negative_items

    @staticmethod
    def sample_by_random_uniform(data: sp.csr_matrix, num_items=99) -> sp.csr_matrix:
        rows = []
        cols = []
        for row in range(data.shape[0]):
            candidate_negatives = list(zip(*data.getrow(row).nonzero()))
            sampled_negatives = np.array(candidate_negatives)[random.sample(range(len(candidate_negatives)), num_items)]
            rows.extend(list(np.ones(len(sampled_negatives), dtype=int) * row))
            cols.extend(sampled_negatives[:, 1])
        negative_samples = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='bool',
                                         shape=(data.shape[0], data.shape[1]))
        return negative_samples

    @staticmethod
    def read_from_files(public_users: t.Dict, public_items: t.Dict, filepath: str):

        def map_to_public(row):
            user = public_users[int(make_tuple(row[0])[0])]
            int_set = set(list(itemgetter(*row[1:])(public_items)))
            return user, [public_items[int(make_tuple(row[0])[1])]] + list(int_set)

        df = pd.read_csv(filepath, sep='\t', header=None)
        print(f'''Mapping for {'validation' if 'validation' in filepath else 'test'} has started...''')
        df = df.apply(lambda x: pd.Series([*map_to_public(x)]), axis=1)
        df.columns = ['user', 'negative_items']
        print(f'''Mapping for {'validation' if 'validation' in filepath else 'test'} is complete...''')
        selected_items = set(np.array(df['negative_items'].tolist()).reshape(-1).tolist())
        return df.set_index('user').to_dict()['negative_items'], selected_items

    @staticmethod
    def build_sparse(map_ : t.Dict, nusers: int, nitems: int):

        rows_cols = [(u, i) for u, items in map_.items() for i in items.keys()]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(nusers, nitems))
        return data