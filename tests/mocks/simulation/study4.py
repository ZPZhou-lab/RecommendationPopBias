from utils import utils
import tensorflow as tf
import numpy as np
import pandas as pd
import random
import itertools
import os
import sys
from sklearn.metrics import roc_auc_score
import time
from tqdm import tqdm
import multiprocessing as mp
import gc

from tests.mocks.utils.dataset import (
    generate_bayesian_recommend_mockdata,
    create_dataloader
)
from tests.mocks.utils.model import (
    BetaVariationalEstimator, 
    BetaMAPEstimator,
    train_estimator,
    train_variational_estimator
)
from evaluator.metrics import (
    calculate_recall,
    calculate_precision,
    calculate_ndcg
)
# set GPU memory limit
utils.set_gpu_memory_limitation(memory=10)

# number of trials
NUM_TRIALS = 10
N_FEATURES = 5
N_ITEMS = 1000
N_POPS = 200
POP_BIAS_SCALE = 0.1
PROB = 0.05

def mock_unbias_evaluation(
    model_vi, model_mle,
    beta_user, beta_item, intercept, pop_bias_mu, 
    n_users: int=5000, unbias=True
):
    def _evaluate(model, dataset):
        metrics = {}
        # make prediction and evaluation
        logits, _ = model.predict(
            tf.constant(dataset.users, dtype=tf.float32),
            tf.constant(dataset.items, dtype=tf.float32),
            tf.constant(dataset.items_pop_idx, dtype=tf.int32),
            unbias=unbias
        )
        score = tf.reshape(logits, (n_users, -1))

        # construct labels
        # labels: Dict[int, List[int]] - user_id to list of item_id
        labels = {}
        for i, row in enumerate(dataset.clicks):
            user_items = np.where(row == 1)[0].tolist()
            # avoid empty labels
            if len(user_items) > 0:
                labels[i] = user_items
        
        for topk in [10, 20, 50]:
            rec_items = tf.math.top_k(score, k=topk).indices
            rec_items = tf.gather(rec_items, list(labels.keys())).numpy()
            metrics[f"Recall@{topk}"]    = calculate_recall(labels, rec_items)
            metrics[f"Precision@{topk}"] = calculate_precision(labels, rec_items)
            metrics[f"NDCG@{topk}"]      = calculate_ndcg(labels, rec_items)
        
        return metrics

    # generate unbiase mock data
    dataset = generate_bayesian_recommend_mockdata(
        n_users=n_users,
        n_items=N_POPS * 30,
        n_pops=N_POPS,
        unbias=unbias,
        pop_dist='lognormal',
        pop_bias_scale=POP_BIAS_SCALE,
        params={
            'beta_user': beta_user,
            'beta_item': beta_item,
            'intercept': intercept,
            'pop_bias_mu':  pop_bias_mu
        }
    )

    metrics_vi = _evaluate(model_vi, dataset)
    metrics_mle = _evaluate(model_mle, dataset)
    return metrics_vi, metrics_mle


def main(seed: int, param: dict, path: str,
         beta_user, beta_item, intercept, pop_bias_mu):
    # set GPU memory limit
    utils.set_gpu_memory_limitation(memory=10)
    # set tensorflow global random seed
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    results = pd.DataFrame(
        columns=["seed", "param", "model", "err", "unbias_metrics", "bias_metrics", "time"]
    )

    # generate mock data
    s_time = time.time()
    dataset = generate_bayesian_recommend_mockdata(
        n_features=N_FEATURES,
        n_users=param['n_users'],
        n_items=N_ITEMS,
        n_pops=N_POPS,
        seed=seed,
        pop_dist='lognormal',
        pop_bias_scale=POP_BIAS_SCALE,
        params={
            'beta_user': beta_user,
            'beta_item': beta_item,
            'intercept': intercept,
            'pop_bias_mu':  pop_bias_mu
        }
    )
    dataloader = create_dataloader(dataset, batch_size=20480)

    # create model
    vi = BetaVariationalEstimator(
        n_features=N_FEATURES,
        n_pops=N_POPS,
        fit_intercept=True,
        heteroscedasticity=False,
        pop_bias_dist='lognormal',
        sigma_trainable=False
    )
    vi.pop_bias_var.set_prior({
        'mu': 0.0,
        'log_sigma': np.log(POP_BIAS_SCALE)
    })
    # train model using Newton's method
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=1000,
        decay_rate=0.9,
        staircase=True
    )
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=lr, global_clipnorm=1.0
    )
    vi = train_variational_estimator(
        model=vi,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=10000,
        max_steps=50000,
        verbose=-1,
        L=param['L'],
        epsilon=1e-5
    )

    # create model
    mle = BetaMAPEstimator(
        n_features=N_FEATURES,
        n_pops=N_POPS,
        fit_intercept=True
    )
    # train model
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1.0,
        decay_steps=50,
        decay_rate=0.5,
        staircase=True
    )
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr)
    mle = train_estimator(
        model=mle,
        dataloader=dataloader,
        optimizer=optimizer,
        max_steps=20000,
        verbose=-1,
        estimate_beta_freq=200,
        epsilon=1e-8,
        use_newton=True,
        l2_reg=0.0
    )

    # evaluate unbias auc
    unbias_metrics = mock_unbias_evaluation(
        vi, mle, dataset.beta_user, dataset.beta_item, dataset.intercept, dataset.pop_bias_mu,
        n_users=5000, unbias=True
    )
    bias_metrics = mock_unbias_evaluation(
        vi, mle, dataset.beta_user, dataset.beta_item, dataset.intercept, dataset.pop_bias_mu,
        n_users=5000, unbias=False
    )

    # consider estimation error
    vi_err = vi.pop_bias_mu.numpy() - dataset.pop_bias_mu + vi.intercept.numpy() - dataset.intercept
    mle_err = mle.pop_bias.numpy() - dataset.pop_bias_mu + mle.intercept.numpy() - dataset.intercept

    # logging progress
    e_time = time.time()

    # save results
    if not os.path.exists(path):
        results.loc[0] = [seed, param, "var", vi_err,  unbias_metrics[0], bias_metrics[0], e_time - s_time]
        results.loc[1] = [seed, param, "mle", mle_err, unbias_metrics[1], bias_metrics[1], e_time - s_time]
    else:
        results = pd.read_pickle(path)
        results.loc[results.shape[0]] = [seed, param, "var", vi_err,  unbias_metrics[0], bias_metrics[0], e_time - s_time]
        results.loc[results.shape[0]] = [seed, param, "mle", mle_err, unbias_metrics[1], bias_metrics[1], e_time - s_time]

    results.to_pickle(path)


if __name__ == "__main__":
    PATH = "./tests/mocks/simulation/results/study4.pkl"
    if os.path.exists(PATH):
        df = pd.read_pickle(PATH)
        start_idx = len(df) // 2
    else:
        start_idx = 0

    # define the grids
    PARAM_GRID_TASK1 = {
        'n_users': [500, 1000, 2000, 5000],
        'L': [1],
        'pop_beta': [0.2, 0.5, 1.0]
    }
    PARAM_GRID_TASK2 = {
        'n_users': [5000],
        'L': [1, 5],
        'pop_beta': [0.5]
    }

    create_param_grid = lambda param_grid: list(dict(zip(param_grid.keys(), val)) for val in itertools.product(*param_grid.values()))
    PARAM_GRIDS = create_param_grid(PARAM_GRID_TASK1) + create_param_grid(PARAM_GRID_TASK2)
    SEED = 1234
    COOL_DOWN_ROUND = 10

    random_state = np.random.RandomState(SEED)
    # generate parameters
    beta_user = random_state.normal(0, 1, N_FEATURES)
    beta_item = random_state.normal(0, 1, N_FEATURES)
    intercept = np.log(PROB / (1 - PROB))

    params = []
    for param in PARAM_GRIDS:
        pop_beta = param['pop_beta']
        mu = np.log(pop_beta)
        pop_bias_mu = random_state.lognormal(mean=mu, sigma=POP_BIAS_SCALE, size=(N_POPS,))
        for i in range(NUM_TRIALS):
            seed = random_state.randint(0, 2**16)
            params.append((
                seed, param, PATH, 
                beta_user, beta_item, intercept, pop_bias_mu))
    
    # run main
    params = params[start_idx:]
    print("Total jobs: ", len(params))
    mp.set_start_method('spawn')
    idx = 0
    for param in tqdm(params):
        # clear session
        gc.collect()
        tf.keras.backend.clear_session()
        time.sleep(1)
        
        while True:
            job = mp.Process(target=main, args=param)
            job.start()
            job.join()
            if job.exitcode == 0:
                break

        idx += 1
        # cool down
        if idx % COOL_DOWN_ROUND == 0:
            time.sleep(60)