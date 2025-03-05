import tensorflow as tf
from keras import Model
from keras.layers import Embedding, Input, Dense
from dataclasses import dataclass
from .bprmf import BPRMatrixFactorization, BPRMFOutput

@dataclass
class PDAMFOutput(BPRMFOutput):
    ...

class PDAMatrixFactorization(BPRMatrixFactorization):
    """Popularity-bias Deconfounding and Adjusting Matrix Factorization"""
    def __init__(self, 
        num_users: int, 
        num_items: int, 
        embed_size: int = 32, 
        loss_func: str = 'BPR', 
        add_bias: bool = False,
        gamma: float = 0.1,
        adjust: bool = False,
        **kwargs
    ):
        """
        Args:
            `num_users`: number of users
            `num_items`: number of items
            `embed_size`: user, item embedding size
            `loss_func`: loss function, either 'BPR' or 'BCE'
            `add_bias`: whether to add user, item and global bias in the scores
            `gamma`: float, temperature parameter for softmax function
            `adjust`: whether to adjust the scores by popularity, default is `False`,\
                if `False`, the model is `PD` model, if `True`, the model is `PDA` model
            `l2_reg`: L2 regularization for embeddings
        """
        super().__init__(num_users, num_items, embed_size, loss_func, add_bias, **kwargs)
        self.gamma = gamma
        self.adjust = adjust
        self.popularity = kwargs.get('popularity', None)
    
    def set_popularity(self, popularity):
        assert len(popularity) == self.num_items, 'Popularity length must be equal to the number of items'
        self.popularity = tf.constant(popularity, dtype=tf.float32)
    
    def call(self, inputs):
        users, pos_items, neg_items, pos_pops, neg_pops = inputs
        # get scores
        outputs = super().call(inputs=(users, pos_items, neg_items))

        # use ELU to make scores positive
        outputs.pos_score = tf.nn.elu(outputs.pos_score) + 1.0
        outputs.neg_score = tf.nn.elu(outputs.neg_score) + 1.0

        # set popularity
        outputs.pos_score *= tf.pow(pos_pops, self.gamma)
        outputs.neg_score *= tf.pow(neg_pops, self.gamma)
    
        return PDAMFOutput(
            users_embed=outputs.users_embed,
            pos_items_embed=outputs.pos_items_embed,
            pos_score=outputs.pos_score,
            neg_items_embed=outputs.neg_items_embed,
            neg_score=outputs.neg_score
        )
    
    def _recommend_batch(self, users, items=None, top_k: int=None, unbias: bool=False):
        if self.adjust:
            assert self.popularity is not None, 'Popularity is not set, please set popularity using set_popularity method before sampling'
            # get scores
            scores = super().recommend(users, items) # (num_users, num_items)
            # adjust scores
            scores = (tf.nn.elu(scores) + 1.0) * tf.pow(tf.gather(self.popularity, items), self.gamma)
            # get top_k items
            with tf.device('/CPU:0'):
                if top_k is None:
                    return scores.numpy()
                else:
                    top_k = tf.nn.top_k(scores, top_k).indices
                    return top_k.numpy()
        else:
            # PD model, only return the scores
            return super()._recommend_batch(users, items, top_k, unbias)
    
    def train_step(self, data):
        return super().train_step(data)
    
    # gamma setter
    def set_gamma(self, gamma):
        self.gamma = gamma
