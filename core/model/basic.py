import tensorflow as tf
import numpy as np
from keras import Model
from dataclasses import dataclass
from core.dataset import PairwiseDataset
from abc import abstractclassmethod

class BaseRecommender(Model):
    def __init__(self, num_users: int, num_items: int, **kwargs):
        super(BaseRecommender, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
    
    def recommend(self, users, items=None, top_k: int=None, batch_size: int=1024):
        """
        recommend items to users
        """
        # use batch inference to avoid OOM
        batchs = []
        for i in range(0, len(users), batch_size):
            batch_users = users[i:i+batch_size]
            batchs.append(self._recommend_batch(batch_users, items, top_k))
        return np.concatenate(batchs, axis=0)
    
    def _recommend_batch(self, users, items=None, top_k: int=None) -> np.ndarray:
        raise NotImplementedError("_recommend_batch method not implemented")
    
    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None, validation_split=0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None, validation_batch_size=None, validation_freq=1, max_queue_size=10, workers=1, use_multiprocessing=False):
        return super().fit(x, y, batch_size, epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight, sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_batch_size, validation_freq, max_queue_size, workers, use_multiprocessing)