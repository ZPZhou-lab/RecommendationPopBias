from .metrics import RecallCallback, PrecisionCallback, HitRateCallback, NDCGCallback
from typing import List
import pandas as pd

def evaluate_recommender(
    model,
    dataset,
    top_k: List[int] = [20, 50],
    unbias_top_k: List[int] = None
):
    # iter on each top_k
    metrics = pd.DataFrame(
        columns=['top_k', 'dataset', 'recall', 'precision', 'hit_rate', 'ndcg']
    )
    for top in top_k:
        for set_ in ['train', 'valid', 'test']:
            # evaluate on testset
            recall    = RecallCallback.evaluate(getattr(dataset, set_), model, top_k=top)
            precision = PrecisionCallback.evaluate(getattr(dataset, set_), model, top_k=top)
            hit_rate  = HitRateCallback.evaluate(getattr(dataset, set_), model, top_k=top)
            ndcg      = NDCGCallback.evaluate(getattr(dataset, set_), model, top_k=top)

            # add to metrics
            metrics.loc[len(metrics)] = [top, set_, recall, precision, hit_rate, ndcg]
    if unbias_top_k is not None:
        for top in unbias_top_k:
            # evaluate on testset
            recall    = RecallCallback.evaluate(dataset.unbias, model, top_k=top, unbias=True)
            precision = PrecisionCallback.evaluate(dataset.unbias, model, top_k=top, unbias=True)
            hit_rate  = HitRateCallback.evaluate(dataset.unbias, model, top_k=top, unbias=True)
            ndcg      = NDCGCallback.evaluate(dataset.unbias, model, top_k=top, unbias=True)
            # add to metrics
            metrics.loc[len(metrics)] = [top, 'unbias', recall, precision, hit_rate, ndcg]

    return metrics