import tensorflow as tf
from keras import Model
from keras.layers import Embedding, Input, Dense
from dataclasses import dataclass
from core.model.bprmf import BPRMatrixFactorization, BPRMFOutput
from core.model.bprmf import bce_loss_func, reg_loss_func

@dataclass
class PDAMFVariationalOutput(BPRMFOutput):
    pos_pops: tf.Tensor = None
    neg_pops: tf.Tensor = None
    pos_pops_vi: tf.Tensor = None
    neg_pops_vi: tf.Tensor = None


class PDAMatrixFactorizationVariational(BPRMatrixFactorization):
    """Popularity-bias Deconfounding and Adjusting using Variational Inference Matrix Factorization"""
    def __init__(self, 
        num_users: int, 
        num_items: int, 
        embed_size: int = 32, 
        loss_func: str = 'BCE', 
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
        # init varitional parameters
        # popularity z ~ exp(lambda) = exp(1 / beta)
        self.q_param = tf.Variable(tf.zeros(shape=(self.num_items,)))
        self.beta_ = tf.math.exp(self.q_param)
        self.loss_func = "BCE" # force set BCE loss
        self.popularity = kwargs.get('popularity', None)
    
    def set_popularity(self, popularity):
        """
        set popularity and 
        """
        assert len(popularity) == self.num_items, 'Popularity length must be equal to the number of items'
        self.popularity = tf.constant(popularity, dtype=tf.float32)
    
    def set_prior_param(self, beta_):
        """use prior popularity to init beta"""
        # beta = exp(q) -> q = log(beta)
        beta_ = tf.constant(beta_, dtype=tf.float32)
        self.q_param.assign(tf.math.log(beta_))
        self.beta_ = tf.math.exp(self.q_param)

    def reparameterize(self, beta_):
        """
        reparameterize the variational distribution
        """
        # inv(CDF of exp) = -ln(1 - F) / lambda = -beta * ln(1 - F)
        batch_size = beta_.shape[0]
        eps = tf.random.uniform(shape=(batch_size,), minval=1e-10, maxval=1 - 1e-10)
        z = -1.0 * beta_ * tf.math.log(1 - eps)
        return z
    
    def call(self, inputs):
        # pos_pops, neg_pops are prior params of popularity p(z)
        users, pos_items, neg_items, pos_pops, neg_pops = inputs
        # get scores
        outputs = super().call(inputs=(users, pos_items, neg_items))

        # use ELU to make scores positive
        outputs.pos_score = tf.nn.sigmoid(outputs.pos_score)
        outputs.neg_score = tf.nn.sigmoid(outputs.neg_score)

        # set popularity = 1 - exp(-gamma * z)
        pos_pops_vi = self.reparameterize(tf.gather(self.beta_, pos_items))
        neg_pops_vi = self.reparameterize(tf.gather(self.beta_, neg_items))
        outputs.pos_score *= (1 - tf.math.exp(-self.gamma * pos_pops_vi))
        outputs.neg_score *= (1 - tf.math.exp(-self.gamma * neg_pops_vi))
    
        return PDAMFVariationalOutput(
            users_embed=outputs.users_embed,
            pos_items_embed=outputs.pos_items_embed,
            pos_score=outputs.pos_score,
            neg_items_embed=outputs.neg_items_embed,
            neg_score=outputs.neg_score,
            pos_pops=pos_pops,
            neg_pops=neg_pops,
            pos_pops_vi=pos_pops_vi,
            neg_pops_vi=neg_pops_vi
        )
    
    def _recommend_batch(self, users, items=None, top_k: int=None):
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
            return super()._recommend_batch(users, items, top_k)
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            outputs = self(data)
            pos_score, neg_score = outputs.pos_score, outputs.neg_score
             
            # calculate the loss
            if self.loss_func == 'BPR':
                ...
                # mf_loss = tf.reduce_mean(bpr_loss_func(pos_score, neg_score))
            elif self.loss_func == 'BCE':
                pos_loss, neg_loss = bce_loss_func(pos_score, neg_score)
                mf_loss = tf.reduce_mean(pos_loss + neg_loss)
            else:
                raise ValueError('Invalid loss function')
            
            # regularization for embedding
            reg_loss = reg_loss_func(outputs.users_embed, outputs.pos_items_embed, outputs.neg_items_embed)
            reg_loss = self.l2_reg * reg_loss / tf.cast(tf.shape(pos_score)[0], tf.float32)

            # variational regularization
            # KL = log(beta_q / beta_p) + (beta_p / beta_q - 1)
            kl_pos = tf.math.log(outputs.pos_pops_vi / outputs.pos_pops + 1e-10) + (outputs.pos_pops / outputs.pos_pops_vi - 1)
            kl_neg = tf.math.log(outputs.neg_pops_vi / outputs.neg_pops + 1e-10) + (outputs.neg_pops / outputs.neg_pops_vi - 1)
            kl_loss = tf.reduce_mean(kl_pos) + tf.reduce_mean(kl_neg)

            loss = mf_loss + reg_loss + kl_loss

        # compute the gradients
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        
        return {f'{self.loss_func}_loss': mf_loss, 'reg_loss': reg_loss, 'kl_loss': kl_loss}
    
    # gamma setter
    def set_gamma(self, gamma):
        self.gamma = gamma
