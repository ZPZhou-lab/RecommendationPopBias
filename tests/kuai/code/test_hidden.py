from utils import utils
# set GPU memory limit
utils.set_gpu_memory_limitation()
import tensorflow as tf
from core.dataset import HiddenDataset, HiddenBatch
from core.model import HiddenPopMatrixFactorization
from core.model.hidden import BetaUpdateCallback
from evaluator.popularity import (
    calculate_recommendation_rate,
    calculate_popularity_vector
)
from evaluator.evaluate import evaluate_recommender
from evaluator.metrics import (
    RecallCallback, 
    PrecisionCallback, 
    HitRateCallback, 
    NDCGCallback,
    RestoreBestCallback
)
from datetime import datetime
import numpy as np
import random
import os

def main(params):
    seed = 1234
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    dataset = HiddenDataset(
        train_path='~/datasets/KuaiRec 2.0/kuai_filter/train_cross_table.csv',
        valid_path='~/datasets/KuaiRec 2.0/kuai_filter/test_valid_cross_table.csv',
        test_path='~/datasets/KuaiRec 2.0/kuai_filter/test_test_cross_table.csv',
        batch_size=params['batch_size'],
        neg_sample=params['neg_sample'],
        global_pop=params['global_pop'],
        pair_wise=params['pair_wise'],
        pop_eps=params['pop_eps']
    )
    dataset.add_unbias_data('~/datasets/KuaiRec 2.0/kuai_filter/unbias_cross_table.csv', periods=[6, 7])

    # BCEMF
    model = HiddenPopMatrixFactorization(
        num_users=dataset.NUM_USERS,
        num_items=dataset.NUM_ITEMS,
        embed_size=params['embed_size'],
        loss_func=params['loss_func'],
        adjust=True,
        pair_wise=params['pair_wise'],
        global_pop=params['global_pop'],
        num_periods=dataset.NUM_PERIODS,
        pop_eps=params['pop_eps'],
        l2_reg=params['l2_reg'],
        penalize_bias=params['penalize_bias']
    )
    # prob = 0.0617
    # model.intercept = np.log(prob / (1 - prob))
    # learning rate scheduler
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer)
    history = model.fit(
        dataset.create_tf_dataset(max_workers=4, max_steps=1000),
        epochs=20,
        callbacks=[
            RecallCallback(dataset, top_k=20),
            PrecisionCallback(dataset, top_k=20),
            HitRateCallback(dataset, top_k=20),
            NDCGCallback(dataset, top_k=20),
            RestoreBestCallback(metric='Recall@20 valid', maximize=True),
            BetaUpdateCallback()
        ]
    )

    # calculate recommendation rate
    metrics = evaluate_recommender(model, dataset, top_k=[20, 50], unbias_top_k=[10, 20])
    print(f"====== Metrics ======\n{metrics}")

    popularity = calculate_popularity_vector(
        data=dataset.train.data, num_items=dataset.NUM_ITEMS
    )
    rec_items = model.recommend(dataset.train.users, top_k=20)
    rate = calculate_recommendation_rate(popularity, rec_items)
    print(f"====== Recommendation Rate ======\n{rate}")

    # save
    metrics['rate'] = str(rate.values)
    metrics['model'] = params['model_name']
    metrics['params'] = str(params)

    return metrics, model

if __name__ == "__main__":
    params = {
        'batch_size': 2048,
        'neg_sample': 1,
        'global_pop': False,
        'pair_wise': True,
        'loss_func': 'BCE', # 'BCE' or 'BPR'
        'pop_eps': 1e-6,
        'embed_size': 64,
        'l2_reg': 1e-3,
        'penalize_bias': 0.0
    }
    params['model_name'] = 'Hidden-global' if params['global_pop'] else 'Hidden-local'
    metrics, model = main(params)
    time_suffix = datetime.now().strftime("%m%d%H%M")
    # metrics.to_excel(f"./tests/kuai/results/{params.get('model_name')}_{time_suffix}_metrics.xlsx", index=False)
    model.save_weights(f"./tests/kuai/{params.get('model_name')}_model.h5")