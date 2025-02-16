import tensorflow as tf
from keras import Model
from keras.layers import Embedding, Input, Dense
from dataclasses import dataclass
from .basic import BaseRecommender

@dataclass
class BPRMFOutput:
    users_embed: tf.Tensor
    pos_items_embed: tf.Tensor
    pos_score: tf.Tensor
    neg_items_embed: tf.Tensor = None
    neg_score: tf.Tensor = None

class BPRMatrixFactorization(BaseRecommender):
    """Bayesian Personalized Ranking Matrix Factorization"""
    def __init__(self, 
        num_users: int, 
        num_items: int, 
        embed_size: int = 32,
        loss_func: str = 'BPR',
        add_bias: bool = False,
        **kwargs
    ):
        """
        Args:
            `num_users`: number of users
            `num_items`: number of items
            `embed_size`: user, item embedding size
            `loss_func`: loss function, either 'BPR' or 'BCE'
            `add_bias`: whether to add user, item and global bias in the scores
            `l2_reg`: L2 regularization for embeddings
        """
        super(BPRMatrixFactorization, self).__init__(num_users, num_items)
        self.embed_size = embed_size
        self.add_bias = add_bias
        # embedding table
        self.user_embed = Embedding(num_users, embed_size, name='user_embed', 
                                    embeddings_initializer='glorot_uniform')
        self.item_embed = Embedding(num_items, embed_size, name='item_embed', 
                                    embeddings_initializer='glorot_uniform')
        if add_bias:
            self.user_bias = tf.Variable(tf.zeros([num_users]), name='user_bias')
            self.item_bias = tf.Variable(tf.zeros([num_items]), name='item_bias')
            self.global_bias = tf.Variable(tf.zeros([]), name='global_bias')

        self.loss_func = loss_func.upper()
        # regularization
        self.l2_reg = kwargs.get('l2_reg', 1e-3)

    def call(self, inputs):
        # all has shape (batch_size, )
        users, pos_items, neg_items = inputs
        # do the embedding
        user_embed = self.user_embed(users)
        pos_item_embed = self.item_embed(pos_items)
        pos_score = tf.reduce_sum(user_embed * pos_item_embed, axis=1)
        if self.add_bias:
            pos_score += tf.gather(self.user_bias, users) \
                      + tf.gather(self.item_bias, pos_items) \
                      + self.global_bias

        neg_item_embed = self.item_embed(neg_items)
        neg_score = tf.reduce_sum(user_embed * neg_item_embed, axis=1)
        if self.add_bias:
            neg_score += tf.gather(self.user_bias, users) \
                      + tf.gather(self.item_bias, neg_items) \
                      + self.global_bias

        outputs = BPRMFOutput(
            users_embed=user_embed,
            pos_items_embed=pos_item_embed,
            pos_score=pos_score,
            neg_items_embed=neg_item_embed,
            neg_score=neg_score
        )
        return outputs

    def _recommend_batch(self, users, items=None, top_k: int=None):
        items = items if items is not None else tf.range(self.num_items)
        user_embed = self.user_embed(users)
        item_embed = self.item_embed(items)
        # scores shape (num_users, num_items)
        scores = tf.matmul(user_embed, item_embed, transpose_b=True)
        if self.add_bias:
            scores += tf.gather(self.user_bias, users)[:, tf.newaxis] \
                    + tf.gather(self.item_bias, items) \
                    + self.global_bias

        # execute in CPU
        with tf.device('/CPU:0'):
            if top_k is not None:
                top_k = tf.math.top_k(scores, k=top_k).indices
                return top_k.numpy()
            else:
                return scores.numpy()
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            outputs = self(data)
            pos_score, neg_score = outputs.pos_score, outputs.neg_score
             
            # calculate the loss
            if self.loss_func == 'BPR':
                mf_loss = tf.reduce_mean(bpr_loss_func(pos_score, neg_score))
            elif self.loss_func == 'BCE':
                pos_loss, neg_loss = bce_loss_func(pos_score, neg_score)
                mf_loss = tf.reduce_mean(pos_loss + neg_loss)
            else:
                raise ValueError('Invalid loss function')
            
            # regularization
            reg_loss = reg_loss_func(outputs.users_embed, outputs.pos_items_embed, outputs.neg_items_embed)
            reg_loss = self.l2_reg * reg_loss / tf.cast(tf.shape(pos_score)[0], tf.float32)
            loss = mf_loss + reg_loss

        # compute the gradients
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return {f'{self.loss_func}_loss': mf_loss, 'reg_loss': reg_loss}
    

def bpr_loss_func(pos_score, neg_score):
    eps = 1e-10
    return -tf.math.log(tf.nn.sigmoid(pos_score - neg_score))

def bce_loss_func(pos_score, neg_score):
    pos_loss = -tf.math.log(tf.nn.sigmoid(pos_score))
    neg_loss = -tf.math.log(1 - tf.nn.sigmoid(neg_score))
    return pos_loss, neg_loss

def reg_loss_func(users_embed, pos_items_embed, neg_items_embed):
    reg_loss = tf.nn.l2_loss(users_embed) + tf.nn.l2_loss(pos_items_embed) + tf.nn.l2_loss(neg_items_embed)
    return reg_loss