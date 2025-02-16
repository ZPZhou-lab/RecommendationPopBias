import tensorflow as tf
from core.dataset import PairwiseDataset
from core.model import MostPopModel
from evaluator.popularity import (
    calculate_recommendation_rate,
    calculate_popularity_vector
)
from evaluator.metrics import RecallCallback, PrecisionCallback, HitRateCallback, NDCGCallback
from utils import utils
# set GPU memory limit
utils.set_gpu_memory_limitation()

if __name__ == "__main__":
    dataset = PairwiseDataset(
        train_path='~/datasets/Douban/movie_filter/train_cross_table.csv',
        valid_path='~/datasets/Douban/movie_filter/test_valid_cross_table.csv',
        test_path='~/datasets/Douban/movie_filter/test_test_cross_table.csv',
    )

    # MostPopModel
    model = MostPopModel(num_users=dataset.NUM_USERS, num_items=dataset.NUM_ITEMS)
    model.fit(dataset, period=None)

    # evaluate
    recall      = RecallCallback.evaluate(dataset.valid, model, top_k=20)
    precision   = PrecisionCallback.evaluate(dataset.valid, model, top_k=20)
    hit_rate    = HitRateCallback.evaluate(dataset.valid, model, top_k=20)
    ndcg        = NDCGCallback.evaluate(dataset.valid, model, top_k=20)

    print(f"====== MostPopModel Evaluation ======")
    print(f"Recall@20:      {recall:.4f}")
    print(f"Precision@20:   {precision:.4f}")
    print(f"HitRate@20:     {hit_rate:.4f}")
    print(f"NDCG@20:        {ndcg:.4f}")

    # calculate recommendation rate
    popularity = calculate_popularity_vector(data=dataset.train.data, num_items=dataset.NUM_ITEMS)
    rec_items = model.recommend(dataset.train.users, top_k=20)
    rate = calculate_recommendation_rate(popularity, rec_items)
    print(f"====== Recommendation Rate ======\n{rate}")

    # save
    rate.to_csv('./tests/douban/results/mostpop_rr.csv')