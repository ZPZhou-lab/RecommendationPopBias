# set GPU memory limit
from utils import utils
utils.set_gpu_memory_limitation()
import tensorflow as tf
import numpy as np
from core.dataset import PairwiseDataset
from core.model import MostPopModel
from evaluator.popularity import (
    calculate_recommendation_rate,
    calculate_popularity_vector
)
from evaluator.evaluate import evaluate_recommender
from datetime import datetime

def main(params):
    dataset = PairwiseDataset(
        train_path='~/datasets/Douban/movie_filter/train_cross_table.csv',
        valid_path='~/datasets/Douban/movie_filter/test_valid_cross_table.csv',
        test_path='~/datasets/Douban/movie_filter/test_test_cross_table.csv',
    )

    # MostPopModel
    model = MostPopModel(num_users=dataset.NUM_USERS, num_items=dataset.NUM_ITEMS)
    model.fit(dataset, period=8)

    # evaluate
    metrics = evaluate_recommender(model, dataset, top_k=[20, 50])
    print(f"====== Metrics ======\n{metrics}")
    
    # calculate recommendation rate
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
    params = {}
    params['model_name'] = 'MostPop'
    metrics, model = main(params)
    time_suffix = datetime.now().strftime("%m%d%H%M")
    metrics.to_excel(f"./tests/douban/results/{params['model_name']}_{time_suffix}_metrics.xlsx", index=False)