import tensorflow as tf
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from typing import Union

def calculate_popularity_vector(
    data: pd.DataFrame, 
    num_items: int,
    item_col: str='item_id', 
    rating_col: str=None,
    normalize: bool=False,
    min_popularity: float=0.0
):
    """
    Calculate the popularity vector of items in the dataset.

    Args:
    - data: pandas DataFrame, the dataset with columns `'item_id'` and `'rating'` (if rating_col is not None)
    - num_items: int, the number of items in the dataset
    - item_col: str, the column name of item ID, default `'item_id'`
    - rating_col: str, the column name of rating, default `None`\
        if provided, the popularity will be calculated as the sum of ratings.
    - normalize: bool, whether to normalize the popularity vector to [0, 1], default `False`
    - min_popularity: float, the minimum popularity value, default `0.0`
    """

    if rating_col is None:
        # get normalized popularity
        popularity = data[item_col].value_counts(normalize=True)
    else:
        # sum ratings as popularity
        assert rating_col in data.columns, f"Column {rating_col} not found in the dataset"
        popularity = data.groupby(item_col)[rating_col].sum()
        popularity = popularity / popularity.sum()
    
    # create a base popularity vector
    base_val = [min_popularity] * num_items
    base = pd.Series(base_val, index=range(0, num_items))
    # merge with base vector to include all items
    popularity = base.add(popularity, fill_value=0)

    if normalize:
        popularity = (popularity - popularity.min()) / (popularity.max() - popularity.min())
    
    return popularity

def calculate_entropy(dist):
    """
    Calculate entropy of a distribution.
    """
    # convert to numpy array
    dist = np.array(dist)
    
    # calculate entropy
    eps = 1e-10
    entropy = -np.sum(dist * np.log2(dist + eps))

    return entropy

def calculate_js_divergence(dist1, dist2):
    """
    Calculate Jensen-Shannon divergence between two distributions.
    """
    # convert to numpy array
    dist1 = np.array(dist1)
    dist2 = np.array(dist2)
    
    # calculate mean distribution
    mean_dist = 0.5 * (dist1 + dist2)
    
    # calculate KL divergence
    eps = 0.0
    kl1 = dist1 * np.log2(dist1 / (mean_dist + eps) + eps)
    kl2 = dist2 * np.log2(dist2 / (mean_dist + eps) + eps)
    # remove 0 values
    kl1[dist1 == 0] = 0
    kl2[dist2 == 0] = 0
    
    # calculate JS divergence
    js_divergence = 0.5 * (np.sum(kl1) + np.sum(kl2))

    return js_divergence


def calculate_recommendation_rate(
    popularity: pd.Series,
    rec_items: np.ndarray,
    n_groups: int=10
):
    """
    calculate the recommendation rate of the top_k recommendations
    
    Args:
    - popularity: pd.Series, the popularity vector of items
    - rec_items: np.ndarray, the recommendation array of items, shape (num_users, num_items)
    - n_groups: int, the number of groups to be divided, default `10`
    """
    # sort popularity
    popularity = popularity.sort_values(ascending=False)
    # divide into groups
    pop_per_group = popularity.sum() / n_groups
    groups = []

    # counts for rec_items
    rec_items = rec_items.flatten().tolist()
    rec_cnts = Counter(rec_items)

    # calculate recommendation rate
    start = 0
    pop_cum = popularity.cumsum()
    for i in range(n_groups):
        pop = pop_per_group * (i + 1)
        end = len(pop_cum[pop_cum <= pop]) if i < n_groups - 1 else len(pop_cum)

        grp_items = popularity.index[start:end]
        # counts of recommended items in the group
        rec_cnt = 0
        for item in grp_items:
            rec_cnt += rec_cnts.get(item, 0)
        groups.append(rec_cnt)
        start = end
    
    groups = pd.Series(groups, index=range(1, n_groups + 1))
    groups = groups / groups.sum()
    return groups

def intervention_sampling_on_popularity(
    data: pd.DataFrame,
    size: float=0.5,
    max_cap: float=0.9
):
    """
    running intervention sampling on popularity.

    Args:
    - data: pandas DataFrame, the dataset with columns `'item_id'` and `'rating'`
    - size: float, the size of the sampled dataset, default `0.5`
    - max_cap: float, the maximum ratio of each items to be sampled, default `0.9`
    """
    # step 1: get the each items-cnt and the max cap for each item
    items_cnt = data['item_id'].value_counts().sort_index()
    items_cap = items_cnt.map(lambda x: int(x * max_cap))

    # step 2: calculate inverse popularity weights
    inv_pop = items_cnt.sum() / items_cnt 
    weights = inv_pop / inv_pop.sum()

    # step 3: determine total number of samples to be drawn
    size = int(size * len(data))
    indices = []

    # step 4: draw samples in terms of items
    print(f"Desired sample size:            {size}")
    items_to_sample = np.random.choice(items_cnt.index, size=size, p=weights, replace=True)
    items_sample_counts = pd.Series(items_to_sample).value_counts().sort_index()
    items_sample_counts = items_sample_counts.clip(upper=items_cap)
    total = items_sample_counts.sum()
    print(f"Actual sample size after cap:   {total}")

    # sample data
    pbar = tqdm(total=len(items_sample_counts), ncols=100)
    for item, counts in items_sample_counts.items():
        item_indices = data[data['item_id'] == item].index.tolist()
        indices.extend(np.random.choice(item_indices, size=counts, replace=False))
        pbar.set_description(f"Sampling item {item}, total {len(indices)} samples")
        pbar.update(1)
    pbar.close()

    print(f"max ind: {max(indices)}")
    print(f"min ind: {min(indices)}")
    return data.loc[indices], indices