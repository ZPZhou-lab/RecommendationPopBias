import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from .pairwise_dataset import PairwiseDataset, PairwiseBatch
from evaluator.popularity import calculate_popularity_vector

@dataclass
class PDABatch(PairwiseBatch):
    pos_items_pop: np.ndarray
    neg_items_pop: np.ndarray

class PDADataset(PairwiseDataset):
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        batch_size: int=1024,
        neg_sample: int=1,
        global_pop: bool=True,
        **kwargs
    ):
        super().__init__(train_path, valid_path, test_path, batch_size, neg_sample)
        self.global_pop = global_pop
        self.popularity = self.build_popularity(pop_eps=kwargs.get('pop_eps', 1e-8))
    
    def set_popularity(self, popularity: Dict[int, np.ndarray]):
        """
        set popularity of items for training data

        Args:
            popularity: Dict[int, np.ndarray], popularity of items at each period
        """
        self.popularity = {}

        # check periods included in training data
        periods = popularity.keys()
        train_periods = self.train.data['period'].unique()
        try:
            assert np.isin(train_periods, list(periods)).all()
        except:
            exclude_periods = list(set(train_periods) - set(periods))
            raise ValueError(f'Periods in training data {exclude_periods} are not included in popularity data')

        for period, pop in popularity.items():
            try:
                assert len(pop) == self.NUM_ITEMS
            except:
                raise ValueError(f'Popularity for period {period} does not match the number of items in training data')
            self.popularity[period] = pop

    def _sampling(self, users):
        if self.popularity is None:
            raise ValueError('Popularity is not set, please set popularity using set_popularity method before sampling')
        
        pos_items, neg_items, pos_items_pop, neg_items_pop = [], [], [], []
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
        return PDABatch(
            users=np.array(users),
            pos_items=np.array(pos_items),
            neg_items=np.array(neg_items),
            pos_items_pop=np.array(pos_items_pop),
            neg_items_pop=np.array(neg_items_pop),
        )
    
    def create_tf_dataset(self, max_workers: int=1, max_steps: int=None):
        def _generator():
            data_loader = self.create_dataset_iter(max_workers=max_workers, max_steps=max_steps)
            for batch in data_loader:
                yield (batch.users, batch.pos_items, batch.neg_items, batch.pos_items_pop, batch.neg_items_pop)
        return tf.data.Dataset.from_generator(
            _generator,
            output_signature=(
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.int32),
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.float32),
                tf.TensorSpec(shape=(self.batch_size * self.neg_sample,), dtype=tf.float32),
            )
        )
    
    def build_popularity(self, pop_eps: float=1e-8):
        # calculate popularity for each period
        popularity = {}
        if self.global_pop:
            data = self.train.data
            pop = calculate_popularity_vector(data, self.NUM_ITEMS, normalize=True)
            pop[pop < pop_eps] = pop_eps
            pop[pop > 1.0] = 1.0
            popularity[0] = pop
            print(f"The global popularity is built")
        else:
            periods = self.train.data['period'].unique()
            for period in periods:
                data = self.train.data[self.train.data['period'] == period]
                period_pop = calculate_popularity_vector(data, self.NUM_ITEMS, normalize=True)
                # add small value to avoid zero popularity
                period_pop[period_pop < pop_eps] = pop_eps
                period_pop[period_pop > 1.0] = 1.0
                popularity[period] = period_pop
            print(f"The popularity is built for {len(popularity)} periods")

        return popularity