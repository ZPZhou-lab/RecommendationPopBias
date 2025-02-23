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
    generate_map_recommend_mockdata,
    create_dataloader
)
from tests.mocks.utils.model import (
    BetaMAPEstimator,
    train_estimator
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
N_FEATURES = 10
PROB = 0.05
N_POPS = 200
N_ITEMS = 1000
N_USERS = 5000
POP_BETA = 0.5
# define the grids
PARAM_GRID = {
    'n_features': [N_FEATURES, ],
    'n_users': [N_USERS],
    'n_items': [N_ITEMS],
    'n_pops': [N_POPS, ],
    'pop_beta': [POP_BETA, ],
    'p': [PROB, ],
    'n_corr':  [0, 1, 2, 3, 4],
    'rho': [0.9, ]
}

def main(seed: int, param: dict, path: str,
         beta_user, beta_item, intercept, pop_bias):
    # set GPU memory limit
    utils.set_gpu_memory_limitation(memory=10)
    # set tensorflow global random seed
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    results = pd.DataFrame(
        columns=["seed", "param", "converged",
                 "beta_user", "beta_item", "intercept", "pop_beta", "pop_bias", "cost"]
    )

    # generate mock data
    s_time = time.time()
    dataset = generate_map_recommend_mockdata(
        n_features=param['n_features'],
        n_users=param['n_users'],
        n_items=param['n_items'],
        n_pops=param['n_pops'],
        seed=seed,
        params={
            'beta_user': beta_user,
            'beta_item': beta_item,
            'intercept': intercept,
            'pop_bias':  pop_bias
        },
        sick={
            'n_corr': param['n_corr'],
            'rho': param['rho']
        }
    )

    # create model
    model = BetaMAPEstimator(
        n_features=param['n_features'],
        n_pops=param['n_pops'],
        fit_intercept=True
    )
    dataloader = create_dataloader(dataset, batch_size=20480)

    # train model using Newton's method
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1.0,
        decay_steps=50,
        decay_rate=0.50
    )
    optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=lr)
    model = train_estimator(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=10000,
        max_steps=10000,
        verbose=-1,
        estimate_beta_freq=200,
        epsilon=1e-5,
        use_newton=True,
        l2_reg=0.0
    )

    # logging progress
    e_time = time.time()

    # create StudyResult
    res = [
        seed, param, model.converged,
        (dataset.beta_user, model.beta_user.numpy()),
        (dataset.beta_item, model.beta_item.numpy()),
        (dataset.intercept, model.intercept.numpy()[0]),
        (dataset.pop_beta,  model.beta.numpy()),
        (dataset.pop_bias,  model.pop_bias.numpy()),
        e_time - s_time
    ]

    # save results
    if not os.path.exists(path):
        results.loc[0] = res
    else:
        results = pd.read_pickle(path)
        results.loc[results.shape[0]] = res
    results.to_pickle(path)


if __name__ == "__main__":
    PATH = "./tests/mocks/simulation/results/study3.pkl"
    if os.path.exists(PATH):
        df = pd.read_pickle(PATH)
        start_idx = len(df)
    else:
        start_idx = 0

    PARAM_GRIDS = list(dict(zip(PARAM_GRID.keys(), val)) for val in itertools.product(*PARAM_GRID.values()))
    SEED = 1234
    COOL_DOWN_ROUND = 10

    # generate params
    random_state = np.random.RandomState(SEED)
    # generate beta
    beta_user = random_state.normal(0, 1, size=N_FEATURES)
    beta_item = random_state.normal(0, 1, size=N_FEATURES)
    intercept = np.log(PROB / (1 - PROB))
    pop_bias  = random_state.exponential(POP_BETA, size=(N_POPS,))

    params = []
    for param in PARAM_GRIDS:
        for i in range(NUM_TRIALS):
            seed = random_state.randint(0, 2**16)
            params.append((seed, param, PATH, beta_user, beta_item, intercept, pop_bias))

    # run main
    params = params[start_idx:]
    mp.set_start_method('spawn')
    idx = 0
    for param in tqdm(params, ncols=100):
        while True:
            job = mp.Process(target=main, args=param)
            job.start()
            job.join()
            # clear session
            gc.collect()
            tf.keras.backend.clear_session()            
            time.sleep(1)

            # check if the job is successful
            if job.exitcode == 0:
                break
            
        idx += 1
        # cool down
        if idx % COOL_DOWN_ROUND == 0:
            time.sleep(60)