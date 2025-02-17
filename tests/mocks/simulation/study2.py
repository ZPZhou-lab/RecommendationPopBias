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
    BetaNoDebiasEstimator,
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
# define the grids
PARAM_GRID = {
    'n_features': [10, 50],
    'n_users': [5000],
    'n_items': [1000],
    'n_pops': [200, ],
    'p': [0.01, 0.05],
    'pop_beta':  [0.2, 0.5, 1.0, 1.5, 2.0],
}

def mock_unbias_evaluation(
    model, model_nodebis, param,
    beta_user, beta_item, intercept, pop_bias, 
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
    dataset = generate_map_recommend_mockdata(
        n_features=param['n_features'],
        n_users=n_users,
        n_items=param['n_pops'] * 30,
        unbias=unbias,
        params={
            'beta_user': beta_user,
            'beta_item': beta_item,
            'intercept': intercept,
            'pop_bias':  pop_bias
        }
    )

    metrics = _evaluate(model, dataset)
    metrics_nodebias = _evaluate(model_nodebis, dataset)
    return metrics, metrics_nodebias


def main(seed: int, param: dict, path: str):
    # set GPU memory limit
    utils.set_gpu_memory_limitation(memory=10)
    # set tensorflow global random seed
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

    results = pd.DataFrame(
        columns=["seed", "param", "model", "unbias_metrics", "bias_metrics", "time"]
    )

    # generate mock data
    s_time = time.time()
    dataset = generate_map_recommend_mockdata(
        n_features=param['n_features'],
        n_users=param['n_users'],
        n_items=param['n_items'],
        n_pops=param['n_pops'],
        p=param['p'],
        pop_beta=param['pop_beta'],
        seed=seed
    )
    dataloader = create_dataloader(dataset, batch_size=20480)

    # create model
    model = BetaMAPEstimator(
        n_features=param['n_features'],
        n_pops=param['n_pops'],
        fit_intercept=True
    )
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
        max_steps=20000,
        verbose=-1,
        estimate_beta_freq=200,
        epsilon=1e-5,
        use_newton=True,
        l2_reg=0.0
    )

    # train model without debiasing
    model_nodebias = BetaNoDebiasEstimator(
        n_features=param['n_features']
    )
    lr = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1.0,
        decay_steps=50,
        decay_rate=0.50
    )
    optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=lr)
    model_nodebias = train_estimator(
        model=model_nodebias,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=10000,
        max_steps=20000,
        verbose=-1,
        estimate_beta_freq=200,
        epsilon=1e-5,
        use_newton=True,
        l2_reg=0.0
    )

    # evaluate unbias auc
    unbias_metrics = mock_unbias_evaluation(
        model, model_nodebias, param,
        dataset.beta_user, dataset.beta_item, dataset.intercept, dataset.pop_bias,
        n_users=5000, unbias=True
    )
    bias_metrics = mock_unbias_evaluation(
        model, model_nodebias, param,
        dataset.beta_user, dataset.beta_item, dataset.intercept, dataset.pop_bias,
        n_users=5000, unbias=False
    )

    # logging progress
    e_time = time.time()

    # save results
    if not os.path.exists(path):
        results.loc[0] = [seed, param, "debias", unbias_metrics[0], bias_metrics[0], e_time - s_time]
        results.loc[1] = [seed, param, "no-debias", unbias_metrics[1], bias_metrics[1], e_time - s_time]
    else:
        results = pd.read_pickle(path)
        results.loc[results.shape[0]] = [seed, param, "debias", unbias_metrics[0], bias_metrics[0], e_time - s_time]
        results.loc[results.shape[0]] = [seed, param, "no-debias", unbias_metrics[1], bias_metrics[1], e_time - s_time]

    results.to_pickle(path)


if __name__ == "__main__":
    PATH = "./tests/mocks/simulation/results/study2.pkl"
    if os.path.exists(PATH):
        df = pd.read_pickle(PATH)
        start_idx = len(df) // 2
    else:
        start_idx = 0

    PARAM_GRIDS = list(dict(zip(PARAM_GRID.keys(), val)) for val in itertools.product(*PARAM_GRID.values()))
    SEED = 1234
    COOL_DOWN_ROUND = 10

    # generate params
    random_state = np.random.RandomState(SEED)
    params = []
    for param in PARAM_GRIDS:
        for i in range(NUM_TRIALS):
            seed = random_state.randint(0, 2**16)
            params.append((seed, param))

    # run main
    params = params[start_idx:]
    print("Total jobs: ", len(params))
    mp.set_start_method('spawn')
    idx = 0
    for seed, param in tqdm(params, ncols=100):
        # clear session
        gc.collect()
        tf.keras.backend.clear_session()
        time.sleep(1)
        
        while True:
            job = mp.Process(target=main, args=(seed, param, PATH))
            job.start()
            job.join()
            if job.exitcode == 0:
                break

        idx += 1
        # cool down
        if idx % COOL_DOWN_ROUND == 0:
            time.sleep(60)