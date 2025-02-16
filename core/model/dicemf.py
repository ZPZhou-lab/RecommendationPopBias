import tensorflow as tf
from keras import Model
from keras.layers import Embedding, Input, Dense
from dataclasses import dataclass
from core.dataset import DICEDataset
from core.model.bprmf import bpr_loss_func
from .basic import BaseRecommender
from keras.callbacks import Callback
from typing import Union

@dataclass
class DICEMFOutput:
    users_embed: tf.Tensor
    # positive samples
    pos_items_embed: tf.Tensor
    pos_score_int: tf.Tensor
    pos_score_pop: tf.Tensor
    pos_score: tf.Tensor
    # negative samples
    neg_items_embed: tf.Tensor = None
    neg_score_int: tf.Tensor = None
    neg_score_pop: tf.Tensor = None
    neg_score: tf.Tensor = None


class DICEMatrixFactorization(BaseRecommender):
    def __init__(self, 
        num_users: int, 
        num_items: int, 
        embed_size: int = 64,
        disc_loss: str = "DCOR",
        disc_penalty: float = 0.01,
        intests_weight: float = 0.1,
        popularity_weight: float = 0.1,
        **kwargs
    ):
        super().__init__(num_users, num_items, **kwargs)
        self.embed_size = embed_size
        self.disc_penalty = disc_penalty
        self.int_weight = intests_weight
        self.pop_weight = popularity_weight

        # embedding table
        self.user_int = Embedding(num_users, embed_size, name='user_int', 
                                    embeddings_initializer='glorot_uniform')
        self.item_int = Embedding(num_items, embed_size, name='item_int',
                                    embeddings_initializer='glorot_uniform')
        self.user_pop = Embedding(num_users, embed_size, name='user_pop',
                                    embeddings_initializer='glorot_uniform')
        self.item_pop = Embedding(num_items, embed_size, name='item_pop',
                                    embeddings_initializer='glorot_uniform')
        
        # config dist loss
        if disc_loss == "L1":
            self.disc_loss = tf.keras.losses.MeanAbsoluteError()
        elif disc_loss == "L2":
            self.disc_loss = tf.keras.losses.MeanSquaredError()
        elif disc_loss == "DCOR":
            self.disc_loss = self._dcor_loss
        else:
            raise ValueError(f"Unsupported dist loss: {disc_loss}")
        
    def decay_loss_weight(self, decay):
        """adapt the weights of interests and popularity"""
        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay
    
    def _dcor_loss(self, vec_a, vec_b):
        """compute distance correlation loss"""
        # Pairwise Euclidean distance matrices for x and y, Shape: [n, n]
        a = tf.sqrt(tf.reduce_sum(
            tf.square(vec_a[:, None] - vec_a), axis=2) + 1e-8)
        b = tf.sqrt(tf.reduce_sum(
            tf.square(vec_b[:, None] - vec_b), axis=2) + 1e-8)

        # Double centering
        A = a - tf.reduce_mean(a, axis=0, keepdims=True) - tf.reduce_mean(a, axis=1, keepdims=True) + tf.reduce_mean(a)
        B = b - tf.reduce_mean(b, axis=0, keepdims=True) - tf.reduce_mean(b, axis=1, keepdims=True) + tf.reduce_mean(b)

        n = tf.cast(tf.shape(vec_a)[0], tf.float32) 

        # Distance covariance
        dcov2_xy = tf.reduce_sum(A * B) / (n * n)
        dcov2_xx = tf.reduce_sum(A * A) / (n * n)
        dcov2_yy = tf.reduce_sum(B * B) / (n * n)

        # Distance correlation
        dcor = -tf.sqrt(dcov2_xy) / tf.sqrt(tf.sqrt(dcov2_xx) * tf.sqrt(dcov2_yy))

        return dcor
    
    def call(self, inputs):
        users, pos_items, neg_items, _ = inputs
        # do the embedding
        users_int = self.user_int(users)
        users_pop = self.user_pop(users)
        pos_items_int = self.item_int(pos_items)
        pos_items_pop = self.item_pop(pos_items)
        neg_items_int = self.item_int(neg_items)
        neg_items_pop = self.item_pop(neg_items)

        # compute scores
        pos_scores_int = tf.reduce_sum(users_int * pos_items_int, axis=1)
        neg_scores_int = tf.reduce_sum(users_int * neg_items_int, axis=1)
        pos_scores_pop = tf.reduce_sum(users_pop * pos_items_pop, axis=1)
        neg_scores_pop = tf.reduce_sum(users_pop * neg_items_pop, axis=1)

        # total scores = intests + popularity
        pos_scores = pos_scores_int + pos_scores_pop
        neg_scores = neg_scores_int + neg_scores_pop
        
        return DICEMFOutput(
            users_embed=users_int,
            pos_items_embed=pos_items_int,
            pos_score_int=pos_scores_int,
            pos_score_pop=pos_scores_pop,
            pos_score=pos_scores,
            neg_items_embed=neg_items_int,
            neg_score_int=neg_scores_int,
            neg_score_pop=neg_scores_pop,
            neg_score=neg_scores
        )
    
    def _recommend_batch(self, users, items=None, top_k: int=None):
        items = items if items is not None else tf.range(self.num_items)
        user_embed = self.user_int(users) + self.user_pop(users)
        item_embed = self.item_int(items) + self.item_pop(items)

        # scores shape (num_users, num_items)
        scores = tf.matmul(user_embed, item_embed, transpose_b=True)
        # execute on CPU
        with tf.device('/CPU:0'):
            if top_k is not None:
                top_k = tf.math.top_k(scores, k=top_k).indices
                return top_k.numpy()
            else:
                return scores.numpy()

    def train_step(self, data):
        with tf.GradientTape() as tape:
            outputs = self(data)
            O2_masks = data[3]
            O1_masks = 1 - O2_masks
            # calculate the loss
            # interst loss (in O2 subset)
            int_loss = bpr_loss_func_with_mask(outputs.pos_score_int, outputs.neg_score_int, O2_masks)
            # comformity loss:
            # O1: ~masks: user click pos-item because comformity
            # O2: masks: user click neg-item because comformity
            pop_loss = bpr_loss_func_with_mask(outputs.pos_score_pop, outputs.neg_score_pop, O1_masks) \
                     + bpr_loss_func_with_mask(outputs.neg_score_pop, outputs.pos_score_pop, O2_masks)
            int_loss = int_loss * self.int_weight
            pop_loss = pop_loss * self.pop_weight
            bpr_loss = bpr_loss_func(outputs.pos_score, outputs.neg_score)
            
            # get all items
            items = tf.unique(tf.concat([data[1], data[2]], axis=0))[0]
            items_int = self.item_int(items)
            items_pop = self.item_pop(items)
            # get all users
            users = tf.unique(data[0])[0]
            users_int = self.user_int(users)
            users_pop = self.user_pop(users)

            # compute the discrepancy loss
            # the larger the discrepancy, the more different the int and pop embeddings
            disc_loss = self.disc_loss(items_int, items_pop) + self.disc_loss(users_int, users_pop)
            disc_loss = disc_loss * self.disc_penalty

            loss = bpr_loss + int_loss + pop_loss - disc_loss
            # loss = bpr_loss
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return {
            "bpr_loss": bpr_loss,
            "int_loss": int_loss,
            "pop_loss": pop_loss,
            "disc_loss": disc_loss
        }


class MarginDecayCallback(Callback):
    """Callback to decay the margin of the DICE Dataset Generator"""
    def __init__(self,
        dataset: DICEDataset, 
        decay_rate=0.9,
        decay_period=1
    ):
        """
        Args:
            dataset (DICEDataset): the dataset to generate the samples
            decay_rate (float): the decay rate of the margin, `margin = margin * decay_rate`
            decay_period (int): the period to decay the margin for each epoch
        """
        self.dataset = dataset
        self.decay_rate = decay_rate
        self.decay_period = decay_period
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.decay_period == 0:
            self.dataset.decay_margin(self.decay_rate)
            print(f"Decay the margin of the DICE Dataset Generator to {self.dataset.margin:.2f}")


class IntPopWeightDecayCallback(Callback):
    """Callback to decay the weights of interests and popularity"""
    def __init__(self,
        decay_rate=0.9,
        decay_period=1
    ):
        """
        Args:
            decay_rate (float): the decay rate of the weights, `weight = weight * decay_rate`
            decay_period (int): the period to decay the weights for each epoch
        """
        self.decay_rate = decay_rate
        self.decay_period = decay_period
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.decay_period == 0:
            self.model.decay_loss_weight(self.decay_rate)
            print(f"Decay the weights of interests and popularity loss to {self.model.int_weight:.2f} and {self.model.pop_weight:.2f}")


def bpr_loss_func_with_mask(pos_score, neg_score, mask=None):
    eps = 1e-10
    if mask is None:
        return -tf.reduce_mean(tf.math.log_sigmoid(pos_score - neg_score))
    mask = tf.cast(mask, tf.float32)
    return -tf.reduce_mean(mask * tf.math.log_sigmoid(pos_score - neg_score))