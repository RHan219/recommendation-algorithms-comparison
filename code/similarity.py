import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import issparse


def compute_item_similarity_matrix(ratings_csr, k=40):
    if issparse(ratings_csr):
        item_matrix = ratings_csr.T.tocsr()  # items x users
        # convert to dense for cosine (OK for moderate item counts e.g., <= 10k)
        X = item_matrix.toarray()
    else:
        X = np.asarray(ratings_csr).T  # ensure items x users

    # compute cosine similarity between item vectors (items x items)
    sim = cosine_similarity(X)  # returns dense matrix

    n_items = sim.shape[0]
    if k <= 0 or k >= n_items:
        # keep full matrix
        np.fill_diagonal(sim, 0.0)
        return sim

    # For each row keep only top-k neighbors (by similarity), zero out others
    for i in range(n_items):
        row = sim[i]
        # get indices of top-k (exclude self)
        # argpartition is O(n) per row
        if k < n_items - 1:
            topk_idx = np.argpartition(row, -k-1)[-k-1:]  # include potential self
            # ensure we don't accidentally include self; filter it out
            topk_idx = topk_idx[topk_idx != i]
            # if we have more than k because self removed, keep top k by sorting
            if topk_idx.size > k:
                topk_idx = topk_idx[np.argsort(row[topk_idx])[-k:]]
        else:
            topk_idx = np.argsort(row)[-k:]
            topk_idx = topk_idx[topk_idx != i]
        mask = np.ones(n_items, dtype=bool)
        mask[topk_idx] = False
        # zero out all except topk_idx
        sim[i][mask] = 0.0

    # optional: zero diagonal explicitly
    np.fill_diagonal(sim, 0.0)
    return sim
