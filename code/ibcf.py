import numpy as np
from scipy.sparse import csr_matrix
from typing import Optional
from tqdm import tqdm


def build_user_item_matrix(n_users: int, n_items: int, rows, cols, data):
    return csr_matrix((data, (rows, cols)), shape=(n_users, n_items))


def cosine_similarity_matrix(mat: csr_matrix):
    # mat: users x items (sparse)
    from scipy.sparse import csr_matrix
    X = mat.T.tocsr()  # items x users
    # compute norms
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    # dense dot product may be large; compute pairwise in blocks if needed
    # For simplicity and ML-100k size, compute dense
    Xd = X.toarray()
    sim = Xd.dot(Xd.T)
    denom = np.outer(norms, norms)
    denom[denom==0] = 1e-9
    sim = sim / denom
    np.fill_diagonal(sim, 0.0)
    return sim


def pearson_similarity_matrix(mat: csr_matrix):
    # center by item mean across users who rated
    X = mat.T.tocsr().toarray()  # items x users
    # subtract mean per item (only considering non-zero entries)
    means = np.zeros(X.shape[0])
    Xc = X.copy()
    for i in range(X.shape[0]):
        row = X[i]
        nz = row != 0
        if np.any(nz):
            means[i] = row[nz].mean()
            Xc[i, nz] = row[nz] - means[i]
    norms = np.sqrt((Xc**2).sum(axis=1))
    denom = np.outer(norms, norms)
    denom[denom==0] = 1e-9
    sim = Xc.dot(Xc.T) / denom
    np.fill_diagonal(sim, 0.0)
    return sim


class ItemCF:
    def __init__(self, n_users:int, n_items:int, similarity: str = 'cosine', k:int = 20):
        assert similarity in ['cosine','pearson']
        self.n_users = n_users
        self.n_items = n_items
        self.similarity = similarity
        self.k = k
        self.sim_matrix = None
        self.user_item = None

    def fit(self, user_item_matrix: csr_matrix):
        self.user_item = user_item_matrix.tocsr()
        if self.similarity == 'cosine':
            self.sim_matrix = cosine_similarity_matrix(self.user_item)
        else:
            self.sim_matrix = pearson_similarity_matrix(self.user_item)

    def predict(self, user:int, item:int) -> Optional[float]:
        # get items user has rated
        row = self.user_item[user].toarray().ravel()
        rated_idx = np.where(row != 0)[0]
        if rated_idx.size == 0:
            return None
        sims = self.sim_matrix[item, rated_idx]
        if self.k is not None and self.k < rated_idx.size:
            topk_idx = np.argsort(np.abs(sims))[-self.k:]
            rated_idx = rated_idx[topk_idx]
            sims = sims[topk_idx]
        ratings = row[rated_idx]
        denom = np.sum(np.abs(sims))
        if denom == 0:
            return None
        return float(np.dot(sims, ratings) / denom)

    def recommend_top_k(self, user:int, k:int=10):
        # score all items not rated by user
        row = self.user_item[user].toarray().ravel()
        candidates = np.where(row == 0)[0]
        scores = []
        for it in candidates:
            pred = self.predict(user, int(it))
            if pred is None:
                pred = 0.0
            scores.append((it, pred))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [i for i,_ in scores[:k]]
