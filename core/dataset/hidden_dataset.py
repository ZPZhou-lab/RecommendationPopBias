import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from .pairwise_dataset import PairwiseDataset, PairwiseBatch
from .pda_dataset import PDABatch, PDADataset
from evaluator.popularity import calculate_popularity_vector

@dataclass
class HiddenBatch(PDABatch):
    periods: np.ndarray

@dataclass
class RandomClicksBatch:
    users: np.ndarray
    items: np.ndarray
    periods: np.ndarray
    clicks: np.ndarray

class HiddenDataset(PDADataset):
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        batch_size: int=1024,
        neg_sample: int=1,
        global_pop: bool=True,
        pair_wise: bool=True,
        **kwargs
    ):
        super().__init__(train_path, valid_path, test_path, batch_size, neg_sample, global_pop, **kwargs)
        self.pair_wise = pair_wise
        self.NUM_PERIODS = len(self.popularity)

    def _sampling(self, users):
        if self.popularity is None:
            raise ValueError('Popularity is not set, please set popularity using set_popularity method before sampling')
        
        # sample pair wise dataset
        if self.pair_wise:
            pos_items, neg_items, pos_items_pop, neg_items_pop, periods = [], [], [], [], []
            for user in users:
                user_items = self.train.user_attr_item[user]
                user_period = self.train.user_attr_period[user]
                # sample positive items
                # reset if all items are sampled
                if len(self.train.user_attr_item_sampled[user]) == len(user_items):
                    self.train.user_attr_item_sampled[user] = []
                remain_pos_items = list(set(user_items) - set(self.train.user_attr_item_sampled[user]))
                pos_item = np.random.choice(remain_pos_items)
                self.train.user_attr_item_sampled[user].append(pos_item)
                pos_period = 0 if self.global_pop else user_period[user_items.index(pos_item)]
                pop_item_pop = self.popularity[pos_period][pos_item]
                # add positive items
                pos_items.append(pos_item)
                pos_items_pop.append(pop_item_pop)
                periods.append(pos_period)

                # sample negative items
                neg_items_, neg_items_pop_ = [], []
                while len(neg_items_) < self.neg_sample:
                    neg_item = np.random.choice(self.items)
                    while neg_item in user_items or neg_item in neg_items_:
                        neg_item = np.random.choice(self.items)
                    neg_items_.append(neg_item)
                    neg_items_pop_.append(self.popularity[pos_period][neg_item])
                neg_items.extend(neg_items_)
                neg_items_pop.extend(neg_items_pop_)

            # extend users
            users = np.repeat(users, self.neg_sample)
            pos_items = np.repeat(pos_items, self.neg_sample)
            pos_items_pop = np.repeat(pos_items_pop, self.neg_sample)
            periods = np.repeat(periods, self.neg_sample)
            return HiddenBatch(
                users=np.array(users),
                pos_items=np.array(pos_items),
                neg_items=np.array(neg_items),
                pos_items_pop=np.array(pos_items_pop),
                neg_items_pop=np.array(neg_items_pop),
                periods=np.array(periods)
            )
        # sample from uniform clicks
        else:
            # generate random indices (n_users, n_items)
            items = np.random.choice(self.NUM_ITEMS, size=len(users), replace=True)
            periods, clicks = [], []
            for user, item in zip(users, items):
                user_items = self.train.user_attr_item[user]
                user_period = self.train.user_attr_period[user]
                if self.global_pop:
                    period = 0
                else:
                    if item in user_items:
                        period = user_period[user_items.index(item)]
                    else:
                        period = np.random.choice(list(self.popularity.keys()))

                # add to batch
                periods.append(period)
                clicks.append(1 if item in user_items else 0)
            return RandomClicksBatch(
                users=np.array(users),
                items=np.array(items),
                periods=np.array(periods),
                clicks=np.array(clicks)
            )

    
    def create_tf_dataset(self, max_workers: int=1, max_steps: int=None):
        def _generator():
            data_loader = self.create_dataset_iter(max_workers=max_workers, max_steps=max_steps)
            for batch in data_loader:
                if self.pair_wise:
                    yield (batch.users, batch.pos_items, batch.neg_items, batch.pos_items_pop, batch.neg_items_pop, batch.periods)
                else:
                    yield (batch.users, batch.items, batch.periods, batch.clicks)
        if self.pair_wise:
            return tf.data.Dataset.from_generator(
                _generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                    tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                    tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                    tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.float32),
                    tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32)
                )
            )
        else:
            return tf.data.Dataset.from_generator(
                _generator,
                output_signature=(
                    tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32),
                    tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32),
                    tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32),
                    tf.TensorSpec(shape=(self.batch_size,), dtype=tf.int32)
                )
            )