import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple
from core.dataset import PairwiseDataset, UserItemDataset
from keras.callbacks import Callback

def calculate_recall(
    labels: Dict[int, List[int]],
    rec_items: np.ndarray,
    avg_user: bool=True
):
    """
    Compute Recall@k

    Args:
        labels: Dict[int, List[int]] - user_id to list of item_id, with `num_users` keys
        rec_items: np.ndarray - recommended items, shape (num_users, top_k)
        avg_user: bool - whether to average recall over users or calculate recall on total
    """
    if avg_user:
        recall = []
        for i, (user, items) in enumerate(labels.items()):
            top_k_items = rec_items[i].tolist()
            hits = len(set(items) & set(top_k_items))
            recall.append(hits / len(items))
        return np.mean(recall)
    else:
        hits, total = 0, 0
        for i, (user, items) in enumerate(labels.items()):
            top_k_items = rec_items[i].tolist()
            hits += len(set(items) & set(top_k_items))
            total += len(items)
        return hits / total

def calculate_precision(
    labels: Dict[int, List[int]],
    rec_items: np.ndarray,
    avg_user: bool=True
):
    """
    Compute Precision@k

    Args:
        labels: Dict[int, List[int]] - user_id to list of item_id, with `num_users` keys
        rec_items: np.ndarray - recommended items, shape (num_users, top_k)
        avg_user: bool - whether to average precision over users or calculate precision on total
    """
    if avg_user:
        precision = []
        for i, (user, items) in enumerate(labels.items()):
            top_k_items = rec_items[i].tolist()
            hits = len(set(items) & set(top_k_items))
            precision.append(hits / len(top_k_items))
        return np.mean(precision)
    else:
        hits, total = 0, 0
        for i, (user, items) in enumerate(labels.items()):
            top_k_items = rec_items[i].tolist()
            hits += len(set(items) & set(top_k_items))
            total += len(top_k_items)
        return hits / total

def calculate_hit_rate(
    labels: Dict[int, List[int]],
    rec_items: np.ndarray
):
    """
    Compute Hit Rate@k

    Args:
        labels: Dict[int, List[int]] - user_id to list of item_id, with `num_users` keys
        rec_items: np.ndarray - recommended items, shape (num_users, top_k)
    """
    hits, total = 0, 0
    for i, (user, items) in enumerate(labels.items()):
        top_k_items = rec_items[i].tolist()
        hits += 1 if len(set(items) & set(top_k_items)) > 0 else 0
        total += 1
    return hits / total

def calculate_ndcg(
    labels: Dict[int, List[int]],
    rec_items: np.ndarray,
):
    """
    Compute NDCG@k

    Args:
        labels: Dict[int, List[int]] - user_id to list of item_id, with `num_users` keys
        rec_items: np.ndarray - recommended items, shape (num_users, top_k)
    """
    ndcg = []
    top_k = rec_items.shape[1]
    for i, (user, items) in enumerate(labels.items()):
        top_k_items = rec_items[i].tolist()
        hits = len(set(items) & set(top_k_items))
        if hits == 0:
            ndcg.append(0)
        else:
            dcg = sum([1 / np.log2(i + 2) if top_k_items[i] in items else 0 for i in range(top_k)])
            idcg = sum([1 / np.log2(i + 2) for i in range(min(len(items), top_k))])
            ndcg.append(dcg / idcg)
    return np.mean(ndcg)


class RecallCallback(Callback):
    def __init__(self, 
        dataset: PairwiseDataset, 
        top_k: int=20, 
        eval_per_epoch: int=1, 
        avg_user: bool=True
    ):
        super(RecallCallback, self).__init__()
        self.dataset = dataset
        self.top_k = top_k
        self.eval_per_epoch = eval_per_epoch
        self.avg_user = avg_user
        
    def on_epoch_end(self, epoch, logs=None):
        """
        evaluate recall
        """
        logs = logs or {}
        if epoch % self.eval_per_epoch == 0:
            # eval on valid set
            recall = self.evaluate(self.dataset.valid, self.model, self.top_k, self.avg_user)
            logs[f'Recall@{self.top_k} valid'] = recall
    
    @staticmethod
    def evaluate(dataset: UserItemDataset, model, top_k: int=20, avg_user: bool=True):
        """
        evaluate recall on test set
        """
        users = dataset.users
        rec_items = model.recommend(users, top_k=top_k)
        recall = calculate_recall(
            labels=dataset.user_attr_item,
            rec_items=rec_items, 
            avg_user=avg_user
        )
        return recall


class PrecisionCallback(Callback):
    def __init__(self, 
        dataset: PairwiseDataset, 
        top_k: int=20, 
        eval_per_epoch: int=1, 
        avg_user: bool=True
    ):
        super(PrecisionCallback, self).__init__()
        self.dataset = dataset
        self.top_k = top_k
        self.eval_per_epoch = eval_per_epoch
        self.avg_user = avg_user
        
    def on_epoch_end(self, epoch, logs=None):
        """
        evaluate precision
        """
        logs = logs or {}
        if epoch % self.eval_per_epoch == 0:
            precision = self.evaluate(self.dataset.valid, self.model, self.top_k, self.avg_user)
            logs[f'Precision@{self.top_k} valid'] = precision

    @staticmethod
    def evaluate(dataset: UserItemDataset, model, top_k: int=20, avg_user: bool=True):
        """
        evaluate precision on test set
        """
        users = dataset.users
        rec_items = model.recommend(users, top_k=top_k)
        precision = calculate_precision(
            labels=dataset.user_attr_item,
            rec_items=rec_items, 
            avg_user=avg_user
        )
        return precision
    
class HitRateCallback(Callback):
    def __init__(self, 
        dataset: PairwiseDataset, 
        top_k: int=20, 
        eval_per_epoch: int=1
    ):
        super(HitRateCallback, self).__init__()
        self.dataset = dataset
        self.top_k = top_k
        self.eval_per_epoch = eval_per_epoch
        
    def on_epoch_end(self, epoch, logs=None):
        """
        evaluate hit rate
        """
        logs = logs or {}
        if epoch % self.eval_per_epoch == 0:
            hit_rate = self.evaluate(self.dataset.valid, self.model, self.top_k)
            logs[f'HitRate@{self.top_k} valid'] = hit_rate

    @staticmethod
    def evaluate(dataset: UserItemDataset, model, top_k: int=20):
        """
        evaluate hit rate on test set
        """
        users = dataset.users
        rec_items = model.recommend(users, top_k=top_k)
        hit_rate = calculate_hit_rate(
            labels=dataset.user_attr_item,
            rec_items=rec_items
        )
        return hit_rate

class NDCGCallback(Callback):
    def __init__(self, 
        dataset: PairwiseDataset, 
        top_k: int=20, 
        eval_per_epoch: int=1
    ):
        super(NDCGCallback, self).__init__()
        self.dataset = dataset
        self.top_k = top_k
        self.eval_per_epoch = eval_per_epoch
        
    def on_epoch_end(self, epoch, logs=None):
        """
        evaluate ndcg
        """
        logs = logs or {}
        if epoch % self.eval_per_epoch == 0:
            ndcg = self.evaluate(self.dataset.valid, self.model, self.top_k)
            logs[f'NDCG@{self.top_k} valid'] = ndcg

    @staticmethod
    def evaluate(dataset: UserItemDataset, model, top_k: int=20):
        """
        evaluate ndcg on test set
        """
        users = dataset.users
        rec_items = model.recommend(users, top_k=top_k)
        ndcg = calculate_ndcg(
            labels=dataset.user_attr_item,
            rec_items=rec_items
        )
        return ndcg

class RestoreBestCallback(Callback):
    def __init__(self, 
        metric: str,
        maximize: bool=True
    ):
        super(RestoreBestCallback, self).__init__()
        self.metric = metric
        self.maximize = maximize
        self.best = float('-inf') if maximize else float('inf')
        self._restore_weights = None
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.maximize:
            if logs[self.metric] > self.best:
                self.best = logs[self.metric]
                self._restore_weights = self.model.get_weights()
        else:
            if logs[self.metric] < self.best:
                self.best = logs[self.metric]
                self._restore_weights = self.model.get_weights()
        # if model weights are nan or inf, restore the best model
        if any([np.isnan(w).any() for w in self.model.get_weights()]):
            if self._restore_weights is not None:
                self.model.set_weights(self._restore_weights)
                print(f"Restoring best model with {self.metric}={self.best}")
        
    def on_train_end(self, logs=None):
        if self._restore_weights is not None:
            self.model.set_weights(self._restore_weights)
            print(f"Restoring best model with {self.metric}={self.best}")
        else:
            print("No best model found")