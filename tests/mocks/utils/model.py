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

    def train_step_newton(self, data, lr=1e-3, l2_reg=0.0):
        # update variables using Newton-Raphson method
        _, _, items_pop_idx, users, items, clicks = data
        covariate, logits = self((users, items, items_pop_idx))
        bce_loss = tf.keras.losses.binary_crossentropy(clicks, logits, from_logits=True)
        loss = bce_loss
        
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
    

class BetaVariationalEstimator(Model):
    def __init__(
        self, n_features, n_users, n_items, 
        heteroscedasticity: bool=False, 
        pop_bias_dist: str='lognormal',
        **kwargs
    ):
        super(BetaVariationalEstimator, self).__init__()
        self.n_features = n_features
        self.n_users = n_users
        self.n_items = n_items
        self.beta_user = tf.keras.layers.Dense(1, activation=None, use_bias=False)
        self.beta_item = tf.keras.layers.Dense(1, activation=None, use_bias=False)
        # variational parameters for pop(i) ~ lognormal(mu, sigma)
        if pop_bias_dist == 'lognormal':
            self.pop_bias_var = vai.LogNormal(
                mu=tf.zeros([n_items, ]),
                log_sigma=tf.zeros([n_items, ]) if heteroscedasticity else 0.0,
                size=n_items,
                heteroscedasticity=heteroscedasticity,
                name='pop_bias_var',
                **kwargs
            )
        elif pop_bias_dist == 'normal':
            self.pop_bias_var = vai.Gaussian(
                mu=tf.zeros([n_items, ]),
                log_sigma=tf.zeros([n_items, ]) if heteroscedasticity else 0.0,
                size=n_items,
                heteroscedasticity=heteroscedasticity,
                name='pop_bias_var',
                **kwargs
            )
        # pop_bias_eps is used for reparametrization for exponential distribution
        # eps is sampled from uniform distribution
        # self.log_pop_bias_eps = tf.Variable(tf.zeros([n_items, ]), trainable=True, name='log_pop_bias_eps')
        # pops ~ exp(1 / beta)
        self.log_beta  = tf.Variable(tf.zeros([]), trainable=False, name='log_beta')
    
    @property
    def beta(self):
        return tf.exp(self.log_beta)

    @property
    def pop_bias(self):
        """repameterize pop_bias from lognormal distribution"""
        return tf.exp(self.pop_bias_var.mu)
    
    @property
    def pop_bias_mean_field(self):
        # mean field approximation for pop_bias
        return tf.exp(self.pop_bias_var.mu + 0.5 * self.pop_bias_var.sigma**2)
    
    
    def call(self, inputs):
        # users, items with shape (batch_size, n_features)
        users, items, items_idx = inputs
        logits = self.beta_user(users) + self.beta_item(items)
        pop_bias = self.pop_bias_var.sample_from_indices(items_idx)[0, :]
        logits = tf.reshape(logits, (-1, )) + pop_bias
        return logits
    
    def estimate_beta_map(self, steps: int=10000, epsilon: float=1e-8):
        pop_bias = tf.constant(self.pop_bias.numpy(), dtype=tf.float32)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
        eps, step = 1, 0
        log_beta = tf.Variable(tf.constant(self.log_beta.numpy()), trainable=True, name='log_beta')
        while eps > epsilon and step < steps:
            beta_bef = tf.exp(log_beta).numpy()
            with tf.GradientTape() as tape:
                # calculate KL divergence between beta and pop_bias and do MAP estimation
                # we assume pop_bias ~ exp(1 / beta)
                # p(pop_bias) = (1 / beta) * exp(-pop_bias / beta)
                # minimize neg-loglike => -log(p(pop_bias)) = log(beta) + pop_bias / beta
                map_loss = tf.reduce_mean(log_beta + pop_bias / tf.exp(log_beta))
            trainable_vars = [log_beta]
            grads = tape.gradient(map_loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
            step += 1
            beta_aft = tf.exp(log_beta).numpy()
            eps = np.abs(beta_bef - beta_aft)
        # assign new beta
        self.log_beta.assign(log_beta)
    

def train_estimator(model, 
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
    model.estimate_beta_map()
    return model


def train_variational_estimator(
    model,
    dataloader,
    epochs=10,
    learning_rate=1e-4,
    verbose: int=1,
    L: int=1,
    kl_reg: float=1e-4,
    estimate_beta_freq: int=-1,
    update_pop_bias_prior_freq: int=1
):
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
    for epoch in range(epochs):
        for step, data in enumerate(dataloader):
            _, items_idx, users, items, clicks, items_pop = data
            with tf.GradientTape() as tape:
                logits_, clicks_ = [], []
                for _ in range(L):
                    logits = model((users, items, items_idx))
                    logits_ = tf.concat([logits_, logits], axis=0)
                    clicks_ = tf.concat([clicks_, clicks], axis=0)
                # log-likelihood of clicks p(c|u, i, pop_bias)
                bce_loss = tf.keras.losses.binary_crossentropy(clicks, tf.nn.sigmoid(logits))
                # KL divergence between q(pop_bias) and p(pop_bias)
                kl_div = kl_reg * model.pop_bias_var.kl_divergence(reduction='mean')
            
                loss = bce_loss + kl_div
            trainable_vars = model.trainable_variables
            grads = tape.gradient(loss, trainable_vars)
            optimizer.apply_gradients(zip(grads, trainable_vars))
        if (epoch + 1) % verbose == 0:
            print(f"Epoch {epoch + 1}, BCE Loss: {bce_loss.numpy():.4f}, KL Div: {kl_div.numpy():.6f}")
        # update beta MAP estimation
        if estimate_beta_freq > 0 and (epoch + 1) % estimate_beta_freq == 0:
            model.estimate_beta_map()
            print(f"Beta MAP estimation done, beta: {model.beta.numpy():.4f}")
        # update pop_bias prior of mu
        if update_pop_bias_prior_freq > 0 and (epoch + 1) % update_pop_bias_prior_freq == 0:
            model.pop_bias_var.prior_mu = \
                tf.constant(model.pop_bias_var.mu.numpy(), dtype=tf.float32)
            print(f"Pop Bias Prior updated!")
    
    return model