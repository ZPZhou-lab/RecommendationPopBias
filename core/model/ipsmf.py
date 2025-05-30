import tensorflow as tf
from keras.layers import Embedding
from dataclasses import dataclass
from .bprmf import BPRMatrixFactorization, reg_loss_func, bce_loss_func

class IPSMMatrixFactorization(BPRMatrixFactorization):
    """Inverse Propensity Score Matrix Factorization"""
    def __init__(self, 
        num_users: int, 
        num_items: int,
        embed_size: int = 32,
        loss_func: str = 'default',
        add_bias: bool = False,
        **kwargs
    ):
        super(IPSMMatrixFactorization, self).__init__(num_users, num_items, embed_size, loss_func, add_bias, **kwargs)
        self.popularity = kwargs.get('popularity', None)
        self.loss_func = loss_func.lower()
        self.tau = kwargs.get('tau', 1000)

    def set_popularity(self, popularity, eps: float=1e-6):
        assert len(popularity) == self.num_items, 'Popularity length should be equal to the number of items'
        # set the lower bound of popularity to eps
        popularity[popularity < eps] = eps
        self.popularity = tf.constant(popularity, dtype=tf.float32)
    
    def train_step(self, data):
        if self.popularity is None:
            raise ValueError('Popularity is not set. Please set the popularity before training')
        
        with tf.GradientTape() as tape:
            outputs = self(data)
            pos_score, neg_score = outputs.pos_score, outputs.neg_score
             
            # calculate the loss
            pos_loss, neg_loss = bce_loss_func(pos_score, neg_score)
            pos_weight = 1.0 / tf.gather(self.popularity, data[1])
            neg_weight = 1.0 / tf.gather(self.popularity, data[2])

            # normalize the weights
            if self.loss_func == 'normalized':
                pos_weight = pos_weight / (tf.reduce_sum(pos_weight))
                neg_weight = neg_weight / (tf.reduce_sum(neg_weight))
                ips_loss = pos_weight * pos_loss + neg_weight * neg_loss
                scale_factor = 1.0
            # IPS loss
            elif self.loss_func == 'clip':
                # clip the weights
                pos_weight = tf.clip_by_value(pos_weight, 0, self.tau)
                neg_weight = tf.clip_by_value(neg_weight, 0, self.tau)
                ips_loss = pos_weight * pos_loss + neg_weight * neg_loss
                scale_factor = tf.cast(tf.shape(pos_score)[0], tf.float32)
             
            ips_loss = tf.where(tf.math.is_inf(ips_loss), tf.zeros_like(ips_loss), ips_loss)
            ips_loss = tf.where(tf.math.is_nan(ips_loss), tf.zeros_like(ips_loss), ips_loss)
            ips_loss = tf.reduce_sum(ips_loss) / scale_factor

            # regularization
            reg_loss = reg_loss_func(outputs.users_embed, outputs.pos_items_embed, outputs.neg_items_embed)
            reg_loss = self.l2_reg * reg_loss / tf.cast(tf.shape(pos_score)[0], tf.float32)
            loss = ips_loss + reg_loss

        # compute the gradients
        grads = tape.gradient(loss, self.trainable_variables)
        # clip the gradients for nan and inf values
        grads = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad) for grad in grads]
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return {f'ips_loss': ips_loss, 'reg_loss': reg_loss}
    
