import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.callbacks import Callback
from . import variational as vai

class BetaMAPEstimator(Model):
    def __init__(self, 
        n_features, 
        n_pops: int, 
        fit_intercept: bool=True, 
        **kwargs
    ):
        super(BetaMAPEstimator, self).__init__(**kwargs)
        self.n_features = n_features
        self.n_pops = n_pops
        self.fit_intercept = fit_intercept
        self.beta_user = tf.Variable(
            initial_value=tf.random_normal_initializer()((n_features, )),
            name='beta_user',
        )
        self.beta_item = tf.Variable(
            initial_value=tf.random_normal_initializer()((n_features, )),
            name='beta_item',
        )
        self.intercept = tf.Variable(tf.zeros([1,]), trainable=fit_intercept, name='intercept')
        self.log_beta  = tf.Variable(tf.zeros([]), trainable=True, name='log_beta')
        # pop_bias_eps is used for reparametrization for exponential distribution
        # eps is sampled from uniform distribution
        self.log_pop_bias_eps   = tf.Variable(tf.zeros([n_pops, ]), trainable=True, name='log_pop_bias_eps')
    
    @property
    def beta(self):
        return tf.exp(self.log_beta)
    
    @property
    def pop_bias(self):
        """repameterize pop_bias from exponential distribution"""
        eps = tf.nn.sigmoid(self.log_pop_bias_eps) # transform to [0, 1]
        pops = -1.0 * self.beta * tf.math.log(1 - eps)
        return pops
    
    @property
    def coef(self):
        return tf.concat([self.beta_user, self.beta_item, self.pop_bias, self.intercept], axis=0)
    
    def call(self, inputs):
        # users, items with shape (batch_size, n_features)
        users, items, items_pop_idx = inputs
        covariate = tf.concat([
            users, items, tf.one_hot(items_pop_idx, self.n_pops), tf.ones([tf.shape(users)[0], 1])
        ], axis=1)
        logits = tf.reduce_sum(covariate * self.coef, axis=1)
        return covariate, logits
    
    def predict(self, 
        users, items, items_pop_idx, 
        unbias: bool=False):
        # get n_users and n_items
        n_users, n_items = users.shape[0], items.shape[0]
        p_users, p_items = users.shape[1], items.shape[1]
        logits = []
        H = tf.zeros((p_users + p_items + self.n_pops + 1, p_users + p_items + self.n_pops + 1))
        for i in range(n_users):
            users_ = tf.tile(users[i][None,:], [n_items, 1])
            if unbias:
                covariate = tf.concat([
                    users_, items, tf.zeros([n_items, self.n_pops]), tf.ones([n_items, 1])
                ], axis=1)
            else:
                covariate = tf.concat([
                    users_, items, tf.one_hot(items_pop_idx, self.n_pops), tf.ones([n_items, 1])
                ], axis=1)
            
            logits_ = tf.reduce_sum(covariate * self.coef, axis=1)
            logits.append(logits_)

            # calculate cov
            probs = tf.sigmoid(logits_)
            W = tf.linalg.diag(probs * (1 - probs))
            H_ = tf.transpose(covariate) @ W @ covariate
            H += H_

        logits = tf.concat(logits, axis=0)
        return logits, H


    def train_step(self, data):
        # update variables using Gradient Descent
        _, _, items_pop_idx, users, items, clicks = data
        with tf.GradientTape() as tape:
            _, logits = self((users, items, items_pop_idx))
            bce_loss = tf.keras.losses.binary_crossentropy(clicks, logits, from_logits=True)
            loss = bce_loss
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return {'loss': loss}

    def train_step_newton(self, data, lr=1.0, l2_reg=0.0):
        # update variables using Newton-Raphson method
        _, _, items_pop_idx, users, items, clicks = data
        covariate, logits = self((users, items, items_pop_idx))
        loss = tf.keras.losses.binary_crossentropy(clicks, logits, from_logits=True)
        # calculate gradients
        probs = tf.sigmoid(logits)
        grads = tf.reduce_sum(tf.transpose(covariate) * (probs - clicks), axis=1)

        # calculate Hessian
        W = tf.linalg.diag(probs * (1 - probs))
        H = tf.transpose(covariate) @ W @ covariate
        if l2_reg > 0:
            reg = tf.eye(H.shape[0])
            # set the diag of the top n_features x n_features to 0
            reg = tf.linalg.set_diag(reg, 
                tf.concat([tf.zeros([self.n_features, ]), 
                           tf.zeros([self.n_features, ]), 
                           tf.ones([self.n_pops + 1, ])], axis=0)
            )
            H += l2_reg * reg

        
        H_inv = tf.linalg.pinv(H)
        beta_step = tf.reduce_sum(H_inv * grads, axis=1)
        
        # get beta_step using linear solver
        # beta_step = tf.linalg.solve(H, grads[:, None])[:, 0]
        
        # scale beta_step by norm
        beta_step = tf.clip_by_norm(beta_step, 1.0)
        beta_step = tf.split(beta_step, [self.n_features, self.n_features, self.n_pops, 1], axis=0)

        # update beta
        self.beta_user.assign_sub(lr * beta_step[0])
        self.beta_item.assign_sub(lr * beta_step[1])
        self.intercept.assign_sub(lr * beta_step[3]) if self.fit_intercept else None
        # update pop_bias_eps using inv_sigmoid
        updated_pop_bias = self.pop_bias - lr * beta_step[2]
        updated_pop_bias = tf.clip_by_value(updated_pop_bias, 1e-8, 100.0)
        # numerical stability
        inv_sigmoid = lambda x: tf.math.log(x / (1 - x))
        eps = 1 - tf.exp(-updated_pop_bias / self.beta)
        log_pop_bias_eps = inv_sigmoid(eps)
        log_pop_bias_eps = tf.clip_by_value(log_pop_bias_eps, -15, 15)
        self.log_pop_bias_eps.assign(log_pop_bias_eps)

        return {'loss': loss}


    def estimate_beta_map(self, steps: int=10000, epsilon: float=1e-6):
        pop_bias = tf.constant(self.pop_bias.numpy(), dtype=tf.float32)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
        eps, step = 1, 0
        while eps > epsilon and step < steps:
            beta_bef = self.beta.numpy()
            with tf.GradientTape() as tape:
                # calculate KL divergence between beta and pop_bias and do MAP estimation
                # we assume pop_bias ~ exp(1 / beta)
                # p(pop_bias) = (1 / beta) * exp(-pop_bias / beta)
                # minimize neg-loglike => -log(p(pop_bias)) = log(beta) + pop_bias / beta
                map_loss = tf.reduce_mean(self.log_beta + pop_bias / self.beta)
            trainable_vars = [self.log_beta]
            grads = tape.gradient(map_loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            step += 1
            beta_aft = self.beta.numpy()
            eps = np.abs(beta_bef - beta_aft)
        
        # update pop_bias using new beta
        # pop_bias = -beta * log(1 - eps), eps = sigmoid(log_pop_bias_eps)
        # => eps = 1 - exp(-pop_bias / beta)
        # => log_pop_bias = inv_sigmoid(eps)
        inv_sigmoid = lambda x: tf.math.log(x / (1 - x))
        eps = 1 - tf.exp(-pop_bias / self.beta)
        self.log_pop_bias_eps.assign(inv_sigmoid(eps))


class BPREstimator(Model):
    def __init__(self,
        n_features,
        embed_size: int=10,
        **kwargs
    ):
        super(BPREstimator, self).__init__(**kwargs)
        self.n_features = n_features
        self.embed_size = embed_size
        self.beta_user = tf.keras.layers.Dense(1, activation='sigmoid', name='beta_user', use_bias=False)
        self.beta_item = tf.keras.layers.Dense(1, activation='sigmoid', name='beta_item', use_bias=False)
        self.intercept = tf.Variable(tf.zeros([]), name='intercept')

    def call(self, inputs):
        # users, items with shape (batch_size, n_features)
        users, items = inputs
        beta_user = self.beta_user(users)
        beta_item = self.beta_item(items)
        logits = tf.reduce_sum(beta_user * beta_item, axis=1) + self.intercept
        return logits
    
    def predict(self, 
        users, items, items_pop_idx, 
        unbias: bool=False):
        # get n_users and n_items
        n_users, n_items = users.shape[0], items.shape[0]
        p_users, p_items = users.shape[1], items.shape[1]
        logits = []

        for i in range(n_users):
            users_ = tf.tile(users[i][None,:], [n_items, 1])
            beta_user = self.beta_user(users_)
            beta_item = self.beta_item(items)
            logits_ = tf.reduce_sum(beta_user * beta_item, axis=1) + self.intercept
            logits.append(logits_)
        
        logits = tf.concat(logits, axis=0)
        return logits, None
    
    def train_step(self, data):
        # update variables using Gradient Descent
        _, _, _, users, items, clicks = data
        with tf.GradientTape() as tape:
            logits = self((users, items))
            bce_loss = tf.keras.losses.binary_crossentropy(clicks, logits, from_logits=True)
            loss = bce_loss
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return {'loss': loss}


class IPSEstimator(Model):
    def __init__(self,
        n_features,
        loss_func: str='clip',
        tau: float=1000.0,
        **kwargs
    ):
        super(IPSEstimator, self).__init__(**kwargs)
        self.n_features = n_features
        self.loss_func = loss_func.lower()
        self.tau = tau
        self.beta_user = tf.keras.layers.Dense(1, name='beta_user', use_bias=False)
        self.beta_item = tf.keras.layers.Dense(1, name='beta_item', use_bias=False)
        self.intercept = tf.Variable(tf.zeros([]), name='intercept')
        self.popularity = None
    
    def set_popularity(self, popularity, eps: float=1e-5):
        popularity = popularity.astype(np.float32)
        popularity = tf.clip_by_value(popularity, eps, 1.0)
        self.popularity = tf.constant(popularity, dtype=tf.float32)
    
    def call(self, inputs):
        # users, items with shape (batch_size, n_features)
        users, items = inputs
        beta_user = self.beta_user(users)[: ,0]
        beta_item = self.beta_item(items)[: ,0]
        logits = beta_user + beta_item + self.intercept
        return logits
    
    def predict(self, 
        users, items, items_pop_idx, 
        unbias: bool=False):
        # get n_users and n_items
        n_users, n_items = users.shape[0], items.shape[0]
        p_users, p_items = users.shape[1], items.shape[1]
        logits = []

        for i in range(n_users):
            users_ = tf.tile(users[i][None,:], [n_items, 1])
            beta_user = self.beta_user(users_)[:, 0]
            beta_item = self.beta_item(items)[: ,0]
            logits_ = beta_user + beta_item + self.intercept
            logits.append(logits_)
        
        logits = tf.concat(logits, axis=0)
        return logits, None
    
    def train_step(self, data):
        # update variables using Gradient Descent
        _, _, items_pop_idx, users, items, clicks = data
        with tf.GradientTape() as tape:
            logits = self((users, items))
            clicks = tf.cast(clicks, tf.float32)
            bce_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=clicks, logits=logits)
            # calculate IPS loss
            pops = tf.gather(self.popularity, items_pop_idx)
            weight = 1.0 / pops
            if self.loss_func == 'clip':
                weight = tf.clip_by_value(weight, 0, self.tau)
                loss = tf.reduce_mean(weight * bce_loss)
            elif self.loss_func == 'norm':
                weight = weight / tf.reduce_sum(weight)
                loss = tf.reduce_sum(weight * bce_loss)
        
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return {'loss': loss}
    
class PDAEstimator(Model):
    def __init__(self,
        n_features,
        embed_size: int=10,
        tau: float=0.5):
        super(PDAEstimator, self).__init__()
        self.n_features = n_features
        self.embed_size = embed_size
        self.tau = tau
        self.beta_user = tf.keras.layers.Dense(1, activation='sigmoid', name='beta_user', use_bias=False)
        self.beta_item = tf.keras.layers.Dense(1, activation='sigmoid', name='beta_item', use_bias=False)
        self.intercept = tf.Variable(tf.zeros([]), name='intercept')
    
    def set_popularity(self, popularity, eps: float=1e-5):
        popularity = popularity.astype(np.float32)
        # normalize popularity
        popularity = (popularity - np.min(popularity)) / (np.max(popularity) - np.min(popularity))
        popularity[popularity < eps] = eps
        popularity = tf.clip_by_value(popularity, eps, 1.0)
        self.popularity = tf.constant(popularity, dtype=tf.float32)
    
    def call(self, inputs):
        # users, items with shape (batch_size, n_features)
        users, items, item_pop_idx = inputs
        beta_user = self.beta_user(users)
        beta_item = self.beta_item(items)
        scores = tf.nn.elu(tf.reduce_sum(beta_user * beta_item, axis=1)) + 1.0
        # add popularity bias
        pops = tf.gather(self.popularity, item_pop_idx)
        logits = scores * tf.pow(pops, self.tau) + self.intercept
        return logits
    
    def predict(self,
        users, items, item_pop_idx,
        unbias: bool=False):
        # get n_users and n_items
        n_users, n_items = users.shape[0], items.shape[0]
        p_users, p_items = users.shape[1], items.shape[1]
        logits = []

        for i in range(n_users):
            users_ = tf.tile(users[i][None,:], [n_items, 1])
            beta_user = self.beta_user(users_)
            beta_item = self.beta_item(items)
            scores = tf.nn.elu(tf.reduce_sum(beta_user * beta_item, axis=1)) + 1.0
            if not unbias:
                pops = tf.gather(self.popularity, item_pop_idx)
                logits_ = scores * tf.pow(pops, self.tau)
            else:
                logits_ = scores
            logits.append(logits_)
        
        logits = tf.concat(logits, axis=0)
        return logits, None
    
    def train_step(self, data):
        # update variables using Gradient Descent
        _, _, items_pop_idx, users, items, clicks = data
        with tf.GradientTape() as tape:
            logits = self((users, items, items_pop_idx))
            bce_loss = tf.keras.losses.binary_crossentropy(clicks, logits, from_logits=True)
            loss = bce_loss
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return {'loss': loss}


class BetaNoDebiasEstimator(BetaMAPEstimator):
    def __init__(self, n_features, **kwargs):
        super(BetaNoDebiasEstimator, self).__init__(n_features, 1, fit_intercept=True, **kwargs)
        # reset pop_bias
        self.log_pop_bias_eps = tf.Variable(tf.zeros([1, ]), trainable=False, name='log_pop_bias_eps')
    
    def call(self, inputs):
        # users, items with shape (batch_size, n_features)
        users, items, items_pop_idx = inputs
        covariate = tf.concat([
            users, items, tf.zeros([tf.shape(users)[0], 1]), tf.ones([tf.shape(users)[0], 1])
        ], axis=1)
        logits = tf.reduce_sum(covariate * self.coef, axis=1)
        return covariate, logits
    
    def estimate_beta_map(self, steps: int = 10000, epsilon: float = 0.000001):
        pass


class BetaVariationalEstimator(Model):
    def __init__(
        self, 
        n_features, 
        n_pops,
        heteroscedasticity: bool=False, 
        fit_intercept: bool=True,
        pop_bias_dist: str='lognormal',
        **kwargs
    ):
        super(BetaVariationalEstimator, self).__init__()
        self.n_features = n_features
        self.n_pops = n_pops
        self.heteroscedasticity = heteroscedasticity
        self.fit_intercept = fit_intercept
        # GLM parameters
        self.beta_user = tf.Variable(tf.random.normal([n_features, ]), name='beta_user')
        self.beta_item = tf.Variable(tf.random.normal([n_features, ]), name='beta_item')
        self.intercept = tf.Variable(tf.zeros([1, ]), name='intercept', trainable=fit_intercept)
        # variational parameters for pop(i) ~ lognormal(mu, sigma)
        if pop_bias_dist == 'lognormal':
            self.pop_bias_var = vai.LogNormal(
                mu=tf.zeros([n_pops, ]),
                log_sigma=tf.zeros([n_pops, ]) if heteroscedasticity else 0.0,
                size=n_pops,
                heteroscedasticity=heteroscedasticity,
                name='pop_bias_var',
                **kwargs
            )
        elif pop_bias_dist == 'normal':
            self.pop_bias_var = vai.Gaussian(
                mu=tf.zeros([n_pops, ]),
                log_sigma=tf.zeros([n_pops, ]) if heteroscedasticity else 0.0,
                size=n_pops,
                heteroscedasticity=heteroscedasticity,
                name='pop_bias_var',
                **kwargs
            )

    @property
    def pop_bias_mu(self):
        """repameterize pop_bias from lognormal distribution"""
        return tf.exp(self.pop_bias_var.mu)
    
    @property
    def pop_bias_mean_field(self):
        # mean field approximation for pop_bias
        return tf.exp(self.pop_bias_var.mu + 0.5 * self.pop_bias_var.sigma**2)
    
    @property
    def coef(self):
        # sample pop_bias from variational distribution
        return tf.concat([self.beta_user, self.beta_item], axis=0)
    
    def call(self, inputs, L: int=1):
        # users, items with shape (batch_size, n_features)
        users, items, items_pop_idx = inputs
        covariate = tf.concat([users, items], axis=1)
        logits_base = tf.reduce_sum(covariate * self.coef, axis=1) + self.intercept

        # sample L times from variational distribution
        # pop_bias with shape (L, N)
        pop_bias = self.pop_bias_var.sample_from_indices(
            indices=items_pop_idx, n_samples=L)
        
        logits = []
        for i in range(L):
            logits_ = logits_base + pop_bias[i]
            logits.append(logits_)

        logits = tf.concat(logits, axis=0)
        return logits

    def predict(self, 
        users, items, items_pop_idx, 
        unbias: bool=False):
        # get n_users and n_items
        n_users, n_items = users.shape[0], items.shape[0]
        p_users, p_items = users.shape[1], items.shape[1]
        # add pop_bias using mean field approximation
        pop_bias = tf.gather(self.pop_bias_mean_field, items_pop_idx)
        
        logits = []
        for i in range(n_users):
            users_ = tf.tile(users[i][None,:], [n_items, 1])
            covariate = tf.concat([users_, items], axis=1)
            logits_ = tf.reduce_sum(covariate * self.coef, axis=1) + self.intercept
            if not unbias:
                logits_ += pop_bias
                
            logits.append(logits_)
        logits = tf.concat(logits, axis=0)
        return logits, None
    

    def train_variational_step(self, data, kl_reg: float=1e-4, L: int=1):
        # update variables using Gradient Descent
        _, _, items_pop_idx, users, items, clicks = data
        with tf.GradientTape() as tape:
            logits = self((users, items, items_pop_idx), L=L)
            clicks = tf.tile(clicks, [L])
            bce_loss = tf.keras.losses.binary_crossentropy(clicks, logits, from_logits=True)
            # KL divergence between q(z) and p(z)
            kl_div = kl_reg * self.pop_bias_var.kl_divergence(reduction='mean')
            loss = bce_loss + kl_div
        trainable_vars = self.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(grads, trainable_vars))

        return {'loss': loss, 'bce_loss': bce_loss, 'kl_div': kl_div}

    def train_variational_newton(self, data, kl_reg: float=1e-4, L: int=1):
        ...

def train_estimator(
    model, 
    dataloader, 
    optimizer=None,
    epochs=1, 
    max_steps: int=100000,
    learning_rate=1e-4,
    verbose: int=1, 
    estimate_beta_freq: int=-1,
    epsilon: float=1e-8,
    use_newton: bool=False,
    l2_reg: float=0.0
):
    # use learning rate scheduler
    model.converged = False
    if optimizer is None:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=2000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer)

    step, norms, max_norm = 0, [], 1
    # use loss stats
    loss_total = tf.keras.metrics.Mean(name='loss_total')

    for epoch in range(epochs):
        if epoch > 0:
            vars_bef = [var.numpy() for var in model.trainable_variables if 'log_beta' not in var.name]
        # reset loss
        loss_total.reset_states()
        for _, data in enumerate(dataloader):
            # update model
            if use_newton:
                # get learning rate from scheduler
                lr = optimizer.learning_rate(step)
                loss = model.train_step_newton(data, lr, l2_reg)['loss']
            else:
                loss = model.train_step(data)['loss']
            # update loss
            step += 1
            loss_total.update_state(loss)

            if verbose > 0 and step % verbose == 0:
                print(f"Epoch {epoch + 1}, Step {step}, BCE Loss: {loss_total.result().numpy():.4f}, max_norm: {norms}")
            if estimate_beta_freq > 0 and step % estimate_beta_freq == 0:
                model.estimate_beta_map()
        
        # check model variables convergence
        if epoch > 0:
            vars_aft = [var.numpy() for var in model.trainable_variables if 'log_beta' not in var.name]
            norms = [np.sqrt(np.sum(np.square(a - b))) / a.size for a, b in zip(vars_aft, vars_bef)]
            max_norm = np.max(norms)

            if max_norm < epsilon:
                model.converged = True
                break
        
        if step > max_steps:
            break

    # estimate beta MAP
    if hasattr(model, 'estimate_beta_map'):
        model.estimate_beta_map()
    return model


def train_variational_estimator(
    model,
    dataloader,
    optimizer=None,
    epochs=10000,
    max_steps: int=50000,
    learning_rate=1e-3,
    verbose: int=-1,
    kl_reg: float=0.0,
    L: int=1,
    update_pop_bias_prior_freq: int=-1,
    epsilon: float=1e-6,
    use_newton: bool=False,
):
    model.converged = False
    if optimizer is None:
        lr = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=2000,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer)
    model._lbfgs_step_results = None
    step, norms, max_norm = 0, [], 1
    loss_total = tf.keras.metrics.Mean(name='loss_total')

    for epoch in range(epochs):
        vars_bef = [var.numpy() for var in model.trainable_variables]
        for _, data in enumerate(dataloader):
            # update model
            if use_newton:
                loss_dict = model.train_variational_newton(data, kl_reg, L)
            else:
                loss_dict = model.train_variational_step(data, kl_reg, L)

            # update loss
            step += 1
            loss_total.update_state(loss_dict['loss'])

            if verbose > 0 and step % verbose == 0:
                print(f"Epoch {epoch + 1}, Step {step}, BCE Loss: {loss_total.result().numpy():.4f}, max_norm: {norms}")
            if update_pop_bias_prior_freq > 0 and step % update_pop_bias_prior_freq == 0:
                model.pop_bias_var.prior_mu = \
                    tf.constant(model.pop_bias_var.mu.numpy(), dtype=tf.float32)

        # check model variables convergence
        if epoch > 0:
            vars_aft = [var.numpy() for var in model.trainable_variables]
            norms = [np.sqrt(np.sum(np.square(a - b))) / a.size for a, b in zip(vars_aft, vars_bef)]
            max_norm = np.max(norms)

            if max_norm < epsilon:
                model.converged = True
                break
        
        if step > max_steps:
            break
    
    return model