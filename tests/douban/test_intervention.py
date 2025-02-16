import tensorflow as tf
from core.dataset import PairwiseDataset
from evaluator.popularity import (
    calculate_popularity_vector,
    calculate_entropy,
    intervention_sampling_on_popularity
)
from utils import utils
# set GPU memory limit
utils.set_gpu_memory_limitation()

if __name__ == "__main__":
    dataset = PairwiseDataset(
        train_path='~/datasets/Douban/movie_filter/train_cross_table.csv',
        valid_path='~/datasets/Douban/movie_filter/test_valid_cross_table.csv',
        test_path='~/datasets/Douban/movie_filter/test_test_cross_table.csv',
        batch_size=1024
    )
    # get the popularity vector
    entropy_train = calculate_entropy(calculate_popularity_vector(
        data=dataset.train.data, num_items=dataset.NUM_ITEMS
    ))
    # do intervention sampling
    train_intervention, indices = intervention_sampling_on_popularity(
        data=dataset.train.data, size=0.3, max_cap=0.9
    )
    entropy_intervention = calculate_entropy(calculate_popularity_vector(
        data=train_intervention, num_items=dataset.NUM_ITEMS
    ))
    indices = dataset.train.data.index.isin(indices)
    entropy_remained = calculate_entropy(calculate_popularity_vector(
        data=dataset.train.data.iloc[~indices], num_items=dataset.NUM_ITEMS
    ))

    print(f"====== Popularity Entropy ======")
    print(f"Entropy on train:        {entropy_train:.4f}")
    print(f"Entropy on intervention: {entropy_intervention:.4f}")
    print(f"Entropy on remained:     {entropy_remained:.4f}")