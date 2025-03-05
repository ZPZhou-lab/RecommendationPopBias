from utils import utils
# set GPU memory limit
utils.set_gpu_memory_limitation()
import tensorflow as tf
from core.dataset import PairwiseDataset
from core.model import BPRMatrixFactorization
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
    dataset = PairwiseDataset(
        train_path='~/datasets/Douban/movie_filter/train_cross_table.csv',
        valid_path='~/datasets/Douban/movie_filter/test_valid_cross_table.csv',
        test_path='~/datasets/Douban/movie_filter/test_test_cross_table.csv',
        batch_size=params['batch_size'],
        neg_sample=params['neg_sample']
    )

    # BCEMF
    model = BPRMatrixFactorization(
        num_users=dataset.NUM_USERS,
        num_items=dataset.NUM_ITEMS,
        embed_size=params['embed_size'],
        loss_func=params['loss_func'], # 'clip' or 'normalized'
        add_bias=params['add_bias'],
        l2_reg=params['l2_reg'],
    )
    # learning rate scheduler
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer)
    history = model.fit(
        dataset.create_tf_dataset(max_workers=4, max_steps=2000),
        epochs=20,
        callbacks=[
            RecallCallback(dataset, top_k=20),
            PrecisionCallback(dataset, top_k=20),
            HitRateCallback(dataset, top_k=20),
            NDCGCallback(dataset, top_k=20),
            RestoreBestCallback(metric='Recall@20 valid', maximize=True)
        ]
    )

    # calculate recommendation rate
    metrics = evaluate_recommender(model, dataset, top_k=[20, 50])
    print(f"====== Metrics ======\n{metrics}")
    
    popularity = calculate_popularity_vector(data=dataset.train.data, num_items=dataset.NUM_ITEMS)
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
        'embed_size': 128,
        'loss_func': 'BCE', # BCE or BPR
        'add_bias': True,
        'l2_reg': 1e-3
    }
    params['model_name'] = 'BCE' if params['loss_func'] == 'BCE' else 'BPR'
    metrics, model = main(params)
    time_suffix = datetime.now().strftime("%m%d%H%M")
    metrics.to_excel(f"./tests/douban/results/{params['model_name']}_{time_suffix}_metrics.xlsx", index=False)