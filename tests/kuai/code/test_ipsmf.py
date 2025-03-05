from utils import utils
# set GPU memory limit
utils.set_gpu_memory_limitation()
import tensorflow as tf
from core.dataset import PairwiseDataset
from core.model import IPSMMatrixFactorization, MostPopModel
from evaluator.metrics import (
    RecallCallback, 
    PrecisionCallback, 
    HitRateCallback, 
    NDCGCallback,
    RestoreBestCallback
)
from evaluator.evaluate import evaluate_recommender
from evaluator.popularity import calculate_popularity_vector, calculate_recommendation_rate
from datetime import datetime
import numpy as np
import random
import os


def main(params):
    seed = 1234
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)        
    dataset = PairwiseDataset(
        train_path='~/datasets/KuaiRec 2.0/kuai_filter/train_cross_table.csv',
        valid_path='~/datasets/KuaiRec 2.0/kuai_filter/test_valid_cross_table.csv',
        test_path='~/datasets/KuaiRec 2.0/kuai_filter/test_test_cross_table.csv',
        batch_size=params['batch_size'],
    )
    dataset.add_unbias_data('~/datasets/KuaiRec 2.0/kuai_filter/unbias_cross_table.csv', periods=[7])

    # BPRMF
    model = IPSMMatrixFactorization(
        num_users=dataset.NUM_USERS,
        num_items=dataset.NUM_ITEMS,
        embed_size=params['embed_size'],
        loss_func=params['loss_func'], # 'clip' or 'normalized'
        add_bias=False,
        l2_reg=params['l2_reg'],
        tau=params['tau'],
    )
    # get popularity vector
    popularity = calculate_popularity_vector(
        dataset.train.data,
        num_items=dataset.NUM_ITEMS)
    model.set_popularity(popularity, eps=params['pop_eps'])

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer)
    history = model.fit(
        dataset.create_tf_dataset(max_workers=4),
        epochs=10,
        callbacks=[
            RecallCallback(dataset, top_k=20),
            PrecisionCallback(dataset, top_k=20),
            HitRateCallback(dataset, top_k=20),
            NDCGCallback(dataset, top_k=20),
            RestoreBestCallback(metric='Recall@20 valid', maximize=True)
        ]
    )

    # evaluate
    metrics = evaluate_recommender(model, dataset, top_k=[20, 50], unbias_top_k=[10, 20])
    print(f"====== Metrics ======\n{metrics}")

    # calculate recommendation rate
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
        'batch_size': 1024,
        'neg_sample': 1,
        'embed_size': 64,
        'loss_func': 'normalized', # 'clip' or 'normalized'
        'tau': 1e3,
        'pop_eps': 1e-5,
        'l2_reg': 1e-3
    }
    params['model_name'] = 'IPS-norm' if params['loss_func'] == 'normalized' else 'IPS-clip'
    metrics, model = main(params)
    time_suffix = datetime.now().strftime("%m%d%H%M")
    metrics.to_excel(f"./tests/kuai/results/{params['model_name']}_{time_suffix}_metrics.xlsx", index=False)