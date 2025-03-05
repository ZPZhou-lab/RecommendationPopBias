import numpy as np
import pandas as pd
import tensorflow as tf
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
from multiprocessing import Process, Queue
import multiprocessing as mp
import math
from abc import abstractmethod

INT_MAX = 2**31 - 1

@dataclass
class UserItemDataset:
    size: int
    users: List[int]
    items: List[int]
    num_users: int
    num_items: int
    user_attr_item:     Dict[int, List[int]]
    user_attr_period:   Dict[int, List[int]]
    item_attr_user:     Dict[int, List[int]]
    data:               pd.DataFrame = None
    # sampling records
    user_attr_item_sampled: Dict[int, List[int]] = None


class BasicDataset:
    def __init__(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        batch_size: int=1024,
        neg_sample: int=1
    ):
        """
        Args:
            `train_path`: path to load training data
            `valid_path`: path to load validation data
            `test_path`: path to load test data
            `batch_size`: batch size of users to sample in training
            `neg_sample`: number of negative samples for each user-item positive pair
        """
        self.train  = load_and_create_dataset(train_path)
        self.valid  = load_and_create_dataset(valid_path)
        self.test   = load_and_create_dataset(test_path)
        self.NUM_USERS = max(self.train.num_users, self.valid.num_users, self.test.num_users)
        self.NUM_ITEMS = max(self.train.num_items, self.valid.num_items, self.test.num_items)
        self.users = np.arange(self.NUM_USERS)
        self.items = np.arange(self.NUM_ITEMS)

        print(f"Dataset Size: {self.train.size} in train, {self.valid.size} in valid, {self.test.size} in test")
        print(f"num_users: {self.NUM_USERS}, num_items: {self.NUM_ITEMS}")
        print(f"Sparsity: {self.train.size / (self.NUM_USERS * self.NUM_ITEMS):.4f}")

        self.batch_size = batch_size
        self.max_steps = None
        self.neg_sample = neg_sample
        # init sampling records
        self.train.user_attr_item_sampled = {user: [] for user in self.train.users}            
    
    def add_unbias_data(self, unbias_path: str, periods: List[int]=None):
        self.unbias = load_and_create_dataset(unbias_path, periods)
        print(f"Unbias Dataset Size: {self.unbias.size}")
    
    def sample(self, batch_size: int=None):
        """
        To sample a pair-wise batch of data in training set
        """
        batch_size = batch_size or self.batch_size
        users = np.random.choice(self.train.users, size=batch_size)
        return self._sampling(users)
    
    def reset_sampling(self):
        self.train.user_attr_item_sampled = {user: [] for user in self.train.users}
    
    @abstractmethod
    def _sampling(self, users):
        """samling logic"""
        raise NotImplementedError
    
    @abstractmethod
    def create_tf_dataset(self, max_workers: int=4, max_steps: int=None):
        """create tensorflow dataset generator"""
        raise NotImplementedError

    def create_dataset_iter(self,
        max_workers: int=4,
        max_steps: int=None,
    ):
        """
        To create a pair-wise dataloader for training using multiprocessing for data sampling,\
        split users into chunks so as to each user can be drawn in every epoch.

        Args:
            max_workers: number of workers for multiprocessing sampling
            max_steps: maximum number of steps to sample, default `None`,
            if `None`, `max_steps` is set to `len(self.train.data) // self.batch_size`
        """
        def _worker(queue: Queue, users_chunk, steps, queue_info):
            step, start = 0, 0
            while step < steps:
                users = users_chunk[start:start+self.batch_size]
                # padding users to batch_size
                if len(users) < self.batch_size:
                    users += users_chunk[:(self.batch_size - len(users))]
                    start = self.batch_size - len(users)
                else:
                    start += self.batch_size
                batch = self._sampling(users)
                queue.put(batch)
                step += 1
            out = queue_info.get()
            queue.cancel_join_thread()
        
        # shuffle users and split into chunks
        batch_queue = Queue(maxsize=1024)
        queue_info  = mp.Manager().Queue(max_workers)

        # set max_steps
        max_steps = math.ceil(self.train.size / self.batch_size) if max_steps is None else max_steps
        self.max_steps = max_steps if self.max_steps is None else self.max_steps

        # generate seed in [0, INT_MAX)]，shuffle users and split into chunks
        seed = np.random.randint(0, INT_MAX)
        gen = np.random.RandomState(seed)
        users = gen.permutation(self.train.users)
        users_chunks = np.array_split(users, max_workers)

        pool = []
        # Start worker processes
        for rank in range(max_workers):
            users_chunk = users_chunks[rank].tolist()
            # assign steps per worker
            num_pos_worker = sum([len(self.train.user_attr_item[user]) for user in users_chunk])
            worker_steps = math.ceil(max_steps * num_pos_worker / self.train.size)
            pool.append(Process(target=_worker, args=(batch_queue, users_chunk, worker_steps, queue_info)))
        for worker in pool:
            worker.start()

        # Stop worker processes
        for _ in range(self.max_steps):
            batch = batch_queue.get(True, timeout=10)
            yield batch

        # stop workers
        for _ in range(max_workers):
            queue_info.put('out')
        for worker in pool:
            worker.join()


def load_and_create_dataset(
    data_path: str,
    periods: List[int]=None
):
    # load [user_id, item_id, period, rating] data
    data = pd.read_csv(data_path, index_col=0)
    data.reset_index(drop=True, inplace=True)
    if periods is not None:
        data = data[data['period'].isin(periods)]
    # create user, item sets
    users, items = data['user_id'].unique(), data['item_id'].unique()
    num_users, num_items = users.max() + 1, items.max() + 1

    # get each user's item, period list
    user_attr = data.groupby('user_id')[['item_id', 'period']].agg(list)
    user_attr_item = dict(zip(user_attr.index, user_attr['item_id']))
    user_attr_period = dict(zip(user_attr.index, user_attr['period']))

    # get each item's user list
    item_attr = data.groupby('item_id')[['user_id']].agg(list)
    item_attr_user = dict(zip(item_attr.index, item_attr['user_id']))

    # create UserItemDataset
    user_item_dataset = UserItemDataset(
        size=len(data),
        users=users,
        items=items,
        num_users=num_users,
        num_items=num_items,
        user_attr_item=user_attr_item,
        user_attr_period=user_attr_period,
        item_attr_user=item_attr_user,
        data=data
    )
    return user_item_dataset