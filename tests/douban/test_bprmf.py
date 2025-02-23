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
import numpy as np
import random
import os



if __name__ == "__main__":
    seed = 1234
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    dataset = PairwiseDataset(
        train_path='~/datasets/Douban/movie_filter/train_cross_table.csv',
        valid_path='~/datasets/Douban/movie_filter/test_valid_cross_table.csv',
        test_path='~/datasets/Douban/movie_filter/test_test_cross_table.csv',
        batch_size=1024,
        neg_sample=1
    )

    # BCEMF
    model = BPRMatrixFactorization(
        num_users=dataset.NUM_USERS,
        num_items=dataset.NUM_ITEMS,
        embed_size=64,
        loss_func='BCE',
        add_bias=False,
        l2_reg=1e-3
    )
    # learning rate scheduler
    # lr = tf.keras.optimizers.schedules.CosineDecay(
    #     initial_learning_rate=1e-4,
    #     decay_steps=5000,
    #     alpha=0.1
    # )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=optimizer)
    history = model.fit(
        dataset.create_tf_dataset(max_workers=4),
        epochs=15,
        callbacks=[
            RecallCallback(dataset, top_k=20),
            PrecisionCallback(dataset, top_k=20),
            HitRateCallback(dataset, top_k=20),
            NDCGCallback(dataset, top_k=20),
            RestoreBestCallback(metric='Recall@20 valid', maximize=True)
        ]
    )

    # calculate recommendation rate
    model_name = f"BCE"
    metrics = evaluate_recommender(model, dataset, top_k=[20, 50])
    print(f"====== Metrics ======\n{metrics}")
    metrics['model'] = model_name
    metrics.to_csv('./tests/douban/results/BCE_metrics.csv')
    
    popularity = calculate_popularity_vector(data=dataset.train.data, num_items=dataset.NUM_ITEMS)
    rec_items = model.recommend(dataset.train.users, top_k=20)
    rate = calculate_recommendation_rate(popularity, rec_items)
    print(f"====== Recommendation Rate ======\n{rate}")

    # save
    rate['model'] = model_name
    rate.to_csv('./tests/douban/results/BCE_rr.csv')