import numpy as np
from typing import List, Dict, Tuple
from math import log2


def rmse(preds: List[float], labels: List[float]) -> float:
    preds = np.array(preds)
    labels = np.array(labels)
    return float(np.sqrt(np.mean((preds-labels)**2)))


def mae(preds: List[float], labels: List[float]) -> float:
    preds = np.array(preds)
    labels = np.array(labels)
    return float(np.mean(np.abs(preds-labels)))


def precision_recall_at_k(recommended: Dict[int, List[int]], ground_truth: Dict[int, set], k:int):
    precisions = []
    recalls = []
    for u, recs in recommended.items():
        rec_k = recs[:k]
        if u not in ground_truth or len(ground_truth[u]) == 0:
            continue
        hits = len(set(rec_k) & ground_truth[u])
        precisions.append(hits / k)
        recalls.append(hits / len(ground_truth[u]))
    return np.mean(precisions) if precisions else 0.0, np.mean(recalls) if recalls else 0.0


def ndcg_at_k(recommended: Dict[int, List[int]], ground_truth: Dict[int, set], k:int):
    ndcgs = []
    for u, recs in recommended.items():
        rel = [1 if r in ground_truth.get(u, set()) else 0 for r in recs[:k]]
        dcg = 0.0
        for idx, val in enumerate(rel):
            if idx == 0:
                dcg += val
            else:
                dcg += val / log2(idx+1+0)
        # ideal dcg
        ideal_rel = sorted([1]*min(k, len(ground_truth.get(u,[]))) + [0]*k, reverse=True)[:k]
        idcg = 0.0
        for idx, val in enumerate(ideal_rel):
            if idx == 0:
                idcg += val
            else:
                idcg += val / log2(idx+1+0)
        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)
    return np.mean(ndcgs) if ndcgs else 0.0
