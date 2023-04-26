import numpy as np
import random
import torch
import torch.utils.data as data


class Sampler:
    def __init__(self, edge_index, num_items, interacted_items, negative_num, batch_size, sampling_sift_pos, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        self.num_items = num_items
        self.negative_num = negative_num
        self.sampling_sift_pos = sampling_sift_pos
        self.interacted_items = interacted_items
        self.train_loader = data.DataLoader(
            dataset=edge_index,
            batch_size=batch_size,
            shuffle=True
        )

    def step(self, pos_train_data):
        neg_candidates = np.arange(self.num_items)

        if self.sampling_sift_pos:
            neg_items = []
            for u in pos_train_data[0]:
                probs = np.ones(self.num_items)
                probs[self.interacted_items[u]] = 0
                probs /= np.sum(probs)

                u_neg_items = np.random.choice(neg_candidates, size=self.negative_num, p=probs, replace=True).reshape(1, -1)

                neg_items.append(u_neg_items)

            neg_items = np.concatenate(neg_items, axis=0)
        else:
            neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), self.negative_num), replace=True)

        neg_items = torch.from_numpy(neg_items)

        return pos_train_data[0].long(), pos_train_data[1].long(), neg_items.long()  # users, pos_items, neg_items
