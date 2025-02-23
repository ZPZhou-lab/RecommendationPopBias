import tensorflow as tf
import numpy as np
from keras import Model
from keras.layers import Embedding, Input, Dense
from dataclasses import dataclass
from tests.mocks.utils.variational import LogNormal
from keras.callbacks import Callback
from .basic import BaseRecommender
from .bprmf import bce_loss_func, bpr_loss_func, reg_loss_func

@dataclass
class HiddenPopMFOutput:
    users_embed: tf.Tensor
    pos_items_embed: tf.Tensor
    pos_score: tf.Tensor
    neg_items_embed: tf.Tensor = None
    neg_score: tf.Tensor = None

class HiddenPopMatrixFactorization(BaseRecommender):
    """Bayesian Personalized Ranking Matrix Factorization"""
    def __init__(self, 
        num_users: int, 
        num_items: int, 
        embed_size: int = 32,
        loss_func: str = 'BCE',
        adjust: bool = False,
        pair_wise: bool = True,
        global_pop: bool = True,
        num_periods: int = 1,
        pop_eps: float = 1e-6,
        **kwargs
    ):
        """
        Args:
            `num_users`: number of users
            `num_items`: number of items
            `embed_size`: user, item embedding size
            `loss_func`: loss function, either 'BPR' or 'BCE'
            `adjust`: whether to adjust the popularity bias
            `pair_wise`: whether to use pair-wise sampling when training
            `global_pop`: whether to use global popularity bias
            `num_periods`: number of periods in the dataset
            `pop_eps`: epsilon for popularity bias
            `l2_reg`: L2 regularization for embeddings
        """
        super(HiddenPopMatrixFactorization, self).__init__(num_users, num_items)
        self.embed_size = embed_size
        self.adjust = adjust
        self.pair_wise = pair_wise
        self.global_pop = global_pop
        self.num_periods = 1 if global_pop else num_periods
        self.pop_eps = pop_eps
        # embedding table
        self.user_embed = Embedding(num_users, embed_size, name='user_embed', 
                                    embeddings_initializer='glorot_uniform')
        self.item_embed = Embedding(num_items, embed_size, name='item_embed', 
                                    embeddings_initializer='glorot_uniform')
        # pop_bias_eps is used for reparametrization for exponential distribution
        # eps is sampled from uniform distribution
        self.log_beta  = tf.Variable(tf.zeros([self.num_periods, 1]), trainable=True, name='log_beta')
        self.log_pop_bias_eps = tf.Variable(tf.zeros([self.num_periods, num_items, ]), trainable=True, name='log_pop_bias_eps')

        self.loss_func = loss_func.upper()
        # regularization
        self.l2_reg = kwargs.get('l2_reg', 1e-3)
        self.intercept = 0
    
    @property
    def beta(self):
        return tf.exp(self.log_beta)
    
    @property
    def pop_bias(self):
        """repameterize pop_bias from exponential distribution"""
        eps = tf.nn.sigmoid(self.log_pop_bias_eps) # transform to [0, 1]
        pops = -1.0 * self.beta * tf.math.log(1 - eps)
        pops = tf.clip_by_value(pops, self.pop_eps, 10)
        return pops
    
    def estimate_beta_map(self, steps: int=10000, epsilon: float=1e-6):
        pop_bias = tf.constant(self.pop_bias.numpy(), dtype=tf.float32)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
        eps, step = 1, 0
        while eps > epsilon and step < steps:
            beta_bef = self.beta.numpy()
            with tf.GradientTape() as tape:
                map_loss = tf.reduce_mean(self.log_beta + pop_bias / self.beta, axis=1)
            trainable_vars = [self.log_beta]
            grads = tape.gradient(map_loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            step += 1
            beta_aft = self.beta.numpy()
            eps = np.linalg.norm(beta_aft - beta_bef)
        
        inv_sigmoid = lambda x: tf.math.log(x / (1 - x))
        eps = 1 - tf.exp(-pop_bias / self.beta)
        self.log_pop_bias_eps.assign(inv_sigmoid(eps))


    def call(self, inputs):
        # all has shape (batch_size, )
        if self.pair_wise:
            users, pos_items, neg_items, pos_items_pop, neg_items_pop, periods = inputs
            
            pop_bias = self.pop_bias
            # do the embedding
            user_embed = self.user_embed(users)
            pos_item_embed = self.item_embed(pos_items)
            pos_score = tf.reduce_sum(user_embed * pos_item_embed, axis=1) + self.intercept
            pos_score += tf.gather_nd(pop_bias, tf.stack([periods, pos_items], axis=1))

            neg_item_embed = self.item_embed(neg_items)
            neg_score = tf.reduce_sum(user_embed * neg_item_embed, axis=1) + self.intercept
            neg_score += tf.gather_nd(pop_bias, tf.stack([periods, neg_items], axis=1))

            outputs = HiddenPopMFOutput(
                users_embed=user_embed,
                pos_items_embed=pos_item_embed,
                pos_score=pos_score,
                neg_items_embed=neg_item_embed,
                neg_score=neg_score
            )
        else:
            users, items, periods, clicks = inputs
            pop_bias = self.pop_bias

            user_embed = self.user_embed(users)
            item_embed = self.item_embed(items)
            scores = tf.reduce_sum(user_embed * item_embed, axis=1) + self.intercept
            scores += tf.gather_nd(pop_bias, tf.stack([periods, items], axis=1))

            outputs = HiddenPopMFOutput(
                users_embed=user_embed,
                pos_items_embed=item_embed,
                pos_score=scores
            )
        return outputs


    def _recommend_batch(self, users, items=None, top_k: int=None):
        items = items if items is not None else tf.range(self.num_items)
        user_embed = self.user_embed(users)
        item_embed = self.item_embed(items)
        # scores shape (num_users, num_items)
        scores = tf.matmul(user_embed, item_embed, transpose_b=True) + self.intercept
        if self.adjust:
            scores += tf.gather(self.pop_bias, items)[None, :]

        # execute in CPU
        with tf.device('/CPU:0'):
            if top_k is not None:
                top_k = tf.math.top_k(scores, k=top_k).indices
                return top_k.numpy()
            else:
                return scores.numpy()
    
    def train_step(self, data):
        if self.pair_wise:
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
        else:
            with tf.GradientTape() as tape:
                outputs = self(data)
                scores = outputs.pos_score
                clicks = data[-1]
                # calculate the loss
                mf_loss = tf.keras.losses.binary_crossentropy(clicks, scores, from_logits=True)
                
                # regularization
                reg_loss = reg_loss_func(outputs.users_embed, outputs.pos_items_embed)
                reg_loss = self.l2_reg * reg_loss / tf.cast(tf.shape(scores)[0], tf.float32)
                loss = mf_loss + reg_loss

            # compute the gradients
            grads = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return {f'{self.loss_func}_loss': mf_loss, 'reg_loss': reg_loss}


class BetaUpdateCallback(Callback):
    def __init__(self, steps: int=10000, epsilon: float=1e-6):
        self.steps = steps
        self.epsilon = epsilon

    def on_epoch_end(self, epoch, logs=None):
        self.model.estimate_beta_map(steps=self.steps, epsilon=self.epsilon)
        print(f"\nEpoch {epoch}: beta={self.model.beta.numpy().flatten()}\n")