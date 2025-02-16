import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from .dataset import BasicDataset
from evaluator.popularity import calculate_popularity_vector

INT_MAX = 2**31 - 1

@dataclass
class DICEBatch:
    users: np.ndarray
    pos_items: np.ndarray
    neg_items: np.ndarray
    masks: np.ndarray

class DICEDataset(BasicDataset):
    def __init__(self, 
        train_path: str, 
        valid_path: str, 
        test_path: str, 
        batch_size: int = 1024,
        neg_sample: int = 1,
        margin: float = 40,
        pool: int = 40
    ):
        super().__init__(train_path, valid_path, test_path, batch_size, neg_sample)
        self.popularity = None
        self.margin = margin
        self.pool = pool
        print("Popularity not set, set popularity using set_popularity method")
    
    def set_popularity(self, period: int=None):
        if period is None:
            data = self.train.data
        else:
            assert period in self.train.data['period'].unique(), f"Period {period} not in training set"
            data = self.train.data.query(f"period == {period}")
    
        self.popularity = calculate_popularity_vector(
            data, self.NUM_ITEMS, min_popularity=0.0
        )
        # times total number of interactions
        self.popularity = self.popularity * len(data)
    
    def _sampling(self, users):
        if self.popularity is None:
            raise ValueError("Popularity not set, set popularity using set_popularity method")
        
        pos_items, neg_items, masks = [], [], []
        for user in users:
            user_items = self.train.user_attr_item[user]
            # sample positive item
            # reset if all items are sampled
            if len(self.train.user_attr_item_sampled[user]) == len(user_items):
                self.train.user_attr_item_sampled[user] = []
            remain_pos_items = list(set(user_items) - set(self.train.user_attr_item_sampled[user]))
            pos_item = np.random.choice(remain_pos_items)
            self.train.user_attr_item_sampled[user].append(pos_item)
            pos_items.append(pos_item)
            
            # get popularity
            pos_item_pop = self.popularity[pos_item]

            # find pop-items not in user_items
            pop_items = np.nonzero(self.popularity > pos_item_pop + self.margin)[0]
            pop_items = pop_items[np.logical_not(np.isin(pop_items, user_items))]
            num_pop_items = len(pop_items)

            # find nonpop-items not in user_items
            nonpop_items = np.nonzero(self.popularity < pos_item_pop * 0.5)[0]
            nonpop_items = nonpop_items[np.logical_not(np.isin(nonpop_items, user_items))]
            num_nonpop_items = len(nonpop_items)

            # user click a very popular item, most likely due to comformity
            # sample a neg-item that less popular than pos-item for Case 1
            if num_pop_items < self.pool:
                neg_items_ = sample_neg_items(self.neg_sample, nonpop_items)
                masks_ = [False] * self.neg_sample
            # user click a very unpopular item, most likely due to interest
            # sample a neg-item that more popular than pos-item for Case 2
            elif num_nonpop_items < self.pool:
                neg_items_ = sample_neg_items(self.neg_sample, pop_items)
                masks_ = [True] * self.neg_sample
            else:
                neg_items_, masks_ = [], []
                while len(neg_items_) < self.neg_sample:
                    if np.random.rand() < 0.5:
                        neg_item = sample_neg_items(1, pop_items)[0]
                        mask = True
                    else:
                        neg_item = sample_neg_items(1, nonpop_items)[0]
                        mask = False
                    neg_items_.append(neg_item)
                    masks_.append(mask)
            neg_items.extend(neg_items_)
            masks.extend(masks_)
        
        # extend users
        users = np.repeat(users, self.neg_sample)
        pos_items = np.repeat(pos_items, self.neg_sample)
        return DICEBatch(
            users=np.array(users),
            pos_items=np.array(pos_items),
            neg_items=np.array(neg_items),
            masks=np.array(masks).astype(int)
        )

    def create_tf_dataset(self, max_workers: int=1, max_steps: int=None):
        def _generator():
            data_loader = self.create_dataset_iter(max_workers=max_workers, max_steps=max_steps)
            for batch in data_loader:
                yield (batch.users, batch.pos_items, batch.neg_items, batch.masks)
        return tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
            )
        )

    def decay_margin(self, decay: float):
        self.margin = self.margin * decay


def sample_neg_items(neg_sample: int, items):
    """sample neg_sample neg-items from given items list"""
    neg_items = []
    while len(neg_items) < neg_sample:
        neg_item = np.random.choice(items)
        while neg_item in neg_items:
            neg_item = np.random.choice(items)
        neg_items.append(neg_item)
    
    return neg_items