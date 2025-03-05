import tensorflow as tf
from keras import Model
from dataclasses import dataclass
from core.dataset import PairwiseDataset
from evaluator.popularity import calculate_popularity_vector
from .basic import BaseRecommender

@dataclass
class BPRMFOutput:
    users_embed: tf.Tensor
    pos_items_embed: tf.Tensor
    pos_score: tf.Tensor
    neg_items_embed: tf.Tensor = None
    neg_score: tf.Tensor = None

class MostPopModel(BaseRecommender):
    def __init__(self,         
        num_users: int, 
        num_items: int,
    ):
        super(MostPopModel, self).__init__(num_users, num_items)

    def fit(self, dataset: PairwiseDataset, period: int=None):
        # get the popularity of each item in training set
        if period is None:
            data = dataset.train.data
        else:
            assert period in dataset.train.data['period'].unique(), f"Period {period} not in training set"
            data = dataset.train.data.query(f"period == {period}")
        
        self.popularity = calculate_popularity_vector(
            data=data,
            num_items=self.num_items
        ).values
        self.popularity = tf.constant(self.popularity, dtype=tf.float32)

    def _recommend_batch(self, users, items=None, top_k: int=None, unbias: bool=False):
        # get the popularity of each item
        num_users = len(users)
        if items is None:
            # get the top_k items according to popularity
            top_k = self.num_items if top_k is None else top_k
            top_k = tf.math.top_k(self.popularity, k=top_k).indices
        else:
            # look up the popularity of the given items and then get the top_k items
            top_k = min(len(items), top_k) if top_k is not None else len(items)
            # mask the popularity = -1 for non-given items
            non_given_items = tf.ones(self.num_items, dtype=tf.bool)
            non_given_items = tf.tensor_scatter_nd_update(
                non_given_items, 
                items[:, tf.newaxis],
                tf.zeros_like(items, dtype=tf.bool))
            popularity = tf.where(
                non_given_items,
                tf.zeros_like(self.popularity) - 1,
                self.popularity
            )
            top_k = tf.math.top_k(popularity, k=top_k).indices
        
        # repeat the top_k items for each user
        with tf.device('/CPU:0'):
            top_k = tf.tile(tf.expand_dims(top_k, axis=0), [num_users, 1])
            return top_k.numpy()