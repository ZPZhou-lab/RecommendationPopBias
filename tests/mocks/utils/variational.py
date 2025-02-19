import tensorflow as tf
import numpy as np
from scipy import stats
from keras.models import Model
from keras.layers import Layer
from typing import Any, Dict, List, Tuple, Union

class VariationalParameter(Model):
    def __init__(self, size: int=1, name: str=None):
        super().__init__(name=name)
        self.size = size
    
    def log_prob(self, x: Union[float, tf.Tensor]):
        raise NotImplementedError("log_prob not implemented")
    
    def sample(self, n_samples: int=1):
        raise NotImplementedError("sample not implemented")
    
    def kl_divergence(self, reduction: str="mean"):
        """calculate KL divergence between approximate posterior and prior"""
        kl = self._cal_kl_divergence()
        if reduction == "mean":
            return tf.reduce_mean(kl)
        elif reduction == "sum":
            return tf.reduce_sum(kl)
        else:
            return kl

    def _cal_kl_divergence(self):
        raise NotImplementedError("_cal_kl_divergence not implemented")
    
    def set_prior(self, prior: Dict[str, Any]):
        for key, value in prior.items():
            if hasattr(self, key):
                if isinstance(value, (float, int)):
                    value = tf.constant([value] * self.size, dtype=tf.float32)
                else:
                    value = tf.constant(value, dtype=tf.float32)
                var = getattr(self, key)
                var.assign(value)
                key = f"prior_{key}"
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid prior key: {key}")
    
class Gaussian(VariationalParameter):
    def __init__(self, 
        mu: Union[float, tf.Tensor, np.ndarray]=0.0,
        log_sigma: Union[float, tf.Tensor, np.ndarray]=0.0,
        heteroscedasticity: bool=False,
        sigma_trainable: bool=True,
        size: int=1,
        name: str=None
    ):
        super().__init__(size=size, name=name)
        # init parameters
        if isinstance(mu, (float, int)):
            mu = tf.constant([mu] * size, dtype=tf.float32)
        if isinstance(log_sigma, (float, int)):
            if heteroscedasticity:
                log_sigma = tf.constant([log_sigma] * size, dtype=tf.float32)
            else:
                log_sigma = tf.constant(log_sigma, dtype=tf.float32)
        else:
            if heteroscedasticity:
                log_sigma = tf.constant(log_sigma, dtype=tf.float32)
            else:
                raise ValueError("log_sigma should be a scalar when heteroscedasticity is False")

        assert mu.shape == (size, ), f"mu should have shape ({size}, )"
        if heteroscedasticity:
            assert log_sigma.shape == (size, ), f"log_sigma should have shape ({size}, )"
        else:
            assert log_sigma.shape == (), f"log_sigma should be a scalar when heteroscedasticity is False"

        # variational parameters
        self.mu        = tf.Variable(initial_value=mu, name="mu")
        self.log_sigma = tf.Variable(initial_value=log_sigma, name="log_sigma", trainable=sigma_trainable)
        # set prior
        self.prior_mu        = tf.constant(mu, dtype=tf.float32)
        self.prior_log_sigma = tf.constant(log_sigma, dtype=tf.float32)
        self.heteroscedasticity = heteroscedasticity

    @property
    def sigma(self):
        return tf.exp(self.log_sigma)
    
    @property
    def prior_sigma(self):
        return tf.exp(self.prior_log_sigma)
    
    @property
    def mean_field(self):
        return self.mu
    
    def interval_estimate(self, alpha: float=0.05, bounds: str="two-sided"):
        # calculate interval estimate
        if bounds == "two-sided":
            z = stats.norm.ppf(1 - alpha / 2)
            lower = self.mu - z * self.sigma
            upper = self.mu + z * self.sigma
        elif bounds == "lower":
            z = stats.norm.ppf(alpha)
            lower = self.mu - z * self.sigma
            upper = np.inf
        elif bounds == "upper":
            z = stats.norm.ppf(1 - alpha)
            lower = -np.inf
            upper = self.mu + z * self.sigma
        else:
            raise ValueError(f"Invalid bounds: {bounds}")
        return lower, upper
    
    def log_prob(self, x: Union[float, tf.Tensor]):
        # log-prob of gaussian
        # p(x) = 1 / (sigma * sqrt(2 * pi)) * exp(-0.5 * ((x - mu) / sigma) ** 2)
        # log(p(x)) = -log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((x - mu) / sigma) ** 2
        log_prob = -self.log_sigma - 0.5 * tf.math.log(2 * np.pi) - 0.5 * ((x - self.mu) / self.sigma) ** 2
        return log_prob
    
    def sample(self, n_samples: int=1):
        # sample using reparametrization trick
        eps = tf.random.normal(shape=(n_samples, self.size), dtype=tf.float32)
        return self.mu + self.sigma * eps
    
    def sample_from_indices(self, indices: tf.Tensor, n_samples: int=1):
        # sample from given indices using reparametrization trick
        eps     = tf.random.normal(shape=(n_samples, indices.shape[0]), dtype=tf.float32)
        mu      = tf.gather(self.mu, indices)
        sigma   = tf.gather(self.sigma, indices) if self.heteroscedasticity else self.sigma
        return mu + sigma * eps
    
    def _cal_kl_divergence(self):
        # KL(q || p) = log(sigma_p / sigma_q) + (sigma_q ^ 2 + (mu_q - mu_p) ^ 2) / (2 * sigma_p ^ 2) - 0.5
        kl = tf.math.log(self.prior_sigma / self.sigma) + (self.sigma ** 2 + (self.mu - self.prior_mu) ** 2) / (2 * self.prior_sigma ** 2) - 0.5
        return kl
    
    def set_prior(self, prior: Dict[str, Any]):
        for key, value in prior.items():
            if hasattr(self, key):
                if isinstance(value, (float, int)):
                    if key == 'log_sigma' and not self.heteroscedasticity:
                        value = tf.constant(value, dtype=tf.float32)
                    else:
                        value = tf.constant([value] * self.size, dtype=tf.float32)
                else:
                    value = tf.constant(value, dtype=tf.float32)
                var = getattr(self, key)
                var.assign(value)
                key = f"prior_{key}"
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid prior key: {key}")
        

class LogNormal(Gaussian):
    def __init__(self, 
        mu: Union[float, tf.Tensor, np.ndarray]=0.0,
        log_sigma: Union[float, tf.Tensor, np.ndarray]=0.0,
        heteroscedasticity: bool=False,
        sigma_trainable: bool=True,
        size: int=1,
        name: str=None
    ):
        super().__init__(mu, log_sigma, heteroscedasticity, sigma_trainable, size, name)

    @property
    def mean_field(self):
        return tf.exp(self.mu + 0.5 * self.sigma ** 2)
    
    def log_prob(self, x: Union[float, tf.Tensor]):
        # log-prob of lognormal
        # p(x) = 1 / (x * sigma * sqrt(2 * pi)) * exp(-0.5 * ((log(x) - mu) / sigma) ** 2)
        # log(p(x)) = -log(x) - log(sigma) - 0.5 * log(2 * pi) - 0.5 * ((log(x) - mu) / sigma) ** 2
        log_prob = -tf.math.log(x) - self.log_sigma - 0.5 * tf.math.log(2 * np.pi) - 0.5 * ((tf.math.log(x) - self.mu) / self.sigma) ** 2
        return log_prob
    
    def sample(self, n_samples: int=1):
        return tf.exp(super().sample(n_samples))
    
    def sample_from_indices(self, indices: tf.Tensor, n_samples: int=1):
        return tf.exp(super().sample_from_indices(indices, n_samples))

    def interval_estimate(self, alpha: float = 0.05, bounds: str = "two-sided"):
        lower, upper = super().interval_estimate(alpha, bounds)
        return tf.exp(lower), tf.exp(upper)