import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from .dataset import BasicDataset

INT_MAX = 2**31 - 1

@dataclass
class PairwiseBatch:
    users: np.ndarray
    pos_items: np.ndarray
    neg_items: np.ndarray

class PairwiseDataset(BasicDataset):
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        batch_size: int=1024,
        neg_sample: int=1
    ):
        super().__init__(train_path, valid_path, test_path, batch_size, neg_sample)

    def _sampling(self, users):
        pos_items, neg_items = [], []
        for user in users:
            user_items = self.train.user_attr_item[user]
            # sample positive items
            # reset if all items are sampled
            if len(self.train.user_attr_item_sampled[user]) == len(user_items):
                self.train.user_attr_item_sampled[user] = []
            remain_pos_items = list(set(user_items) - set(self.train.user_attr_item_sampled[user]))
            pos_item = np.random.choice(remain_pos_items)
            self.train.user_attr_item_sampled[user].append(pos_item)
            pos_items.append(pos_item)
            
            # sample positive item
            neg_items_ = []
            while len(neg_items_) < self.neg_sample:
                neg_item = np.random.choice(self.items)
                while neg_item in user_items or neg_item in neg_items_:
                    neg_item = np.random.choice(self.items)
                neg_items_.append(neg_item)
            neg_items.extend(neg_items_)

        # extend users
        users = np.repeat(users, self.neg_sample)
        pos_items = np.repeat(pos_items, self.neg_sample)
        return PairwiseBatch(
            users=np.array(users),
            pos_items=np.array(pos_items),
            neg_items=np.array(neg_items),
        )

    def create_tf_dataset(self, max_workers: int=1, max_steps: int=None):
        def _generator():
            data_loader = self.create_dataset_iter(max_workers=max_workers, max_steps=max_steps)
            for batch in data_loader:
                yield (batch.users, batch.pos_items, batch.neg_items)
        return tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
            )
        )