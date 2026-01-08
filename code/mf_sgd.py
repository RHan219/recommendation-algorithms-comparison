import numpy as np
from typing import List, Tuple, Optional
from tqdm import trange

class MF_SGD:
    def __init__(self,
                 n_users:int,
                 n_items:int,
                 n_factors:int=20,
                 lr:float=0.01,
                 reg:float=0.02,
                 n_epochs:int=20,
                 seed:int=42,
                 item_sim:Optional[np.ndarray]=None,
                 sim_reg:float=0.0):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

        # similarity matrix (n_items x n_items) and sim_reg weight
        self.item_sim = item_sim
        self.sim_reg = sim_reg

        # RNG for deterministic init and shuffling
        self.rng = np.random.RandomState(self.seed)

        # parameters
        self.P = self.rng.normal(scale=0.1, size=(n_users, n_factors))
        self.Q = self.rng.normal(scale=0.1, size=(n_items, n_factors))
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.mu = 0.0

        # Precompute neighbor lists from item_sim for efficiency (if provided)
        self._item_neighbors = None
        if self.item_sim is not None:
            self._build_neighbor_struct()

    def _build_neighbor_struct(self):
        """
        From self.item_sim (dense matrix), build a list of neighbor indices and weights per item
        to speed up gradient computation.
        """
        sim = self.item_sim
        n_items = sim.shape[0]
        neighbors = []
        weights = []
        for i in range(n_items):
            row = sim[i]
            nz_idx = np.where(row > 0)[0]
            if nz_idx.size == 0:
                neighbors.append(np.array([], dtype=int))
                weights.append(np.array([], dtype=float))
            else:
                neighbors.append(nz_idx)
                weights.append(row[nz_idx])
        self._item_neighbors = neighbors
        self._item_weights = weights

    def fit(self, interactions: List[Tuple[int,int,float]], verbose=True):
        """
        interactions: list or array-like of (u, i, r)
        """
        interactions = np.asarray(interactions)
        if interactions.ndim != 2 or interactions.shape[1] != 3:
            raise ValueError("interactions must be an array-like of shape (n, 3)")

        self.mu = np.mean(interactions[:,2].astype(float))
        n = interactions.shape[0]

        for epoch in trange(self.n_epochs, desc='MF epochs', disable=not verbose):
            # deterministic shuffle per epoch
            perm = self.rng.permutation(n)
            for idx in perm:
                u = int(interactions[idx,0])
                i = int(interactions[idx,1])
                r = float(interactions[idx,2])

                # prediction and error
                pred = self.mu + self.bu[u] + self.bi[i] + self.P[u].dot(self.Q[i])
                err = r - pred  # residual

                # gradients for biases and latent factors (standard)
                # update biases
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])

                # store copies for update
                Pu = self.P[u].copy()
                Qi = self.Q[i].copy()

                # gradient step for P[u] and Q[i] (w/ reg)
                self.P[u] += self.lr * (err * Qi - self.reg * Pu)
                self.Q[i] += self.lr * (err * Pu - self.reg * Qi)

                # similarity-based regularization (graph smoothness)
                # Adds gradient: 2 * sim_reg * sum_j w_ij * (Q_i - Q_j)
                if (self.sim_reg is not None and self.sim_reg > 0.0
                        and self._item_neighbors is not None
                        and self._item_neighbors[i].size > 0):

                    neigh_idx = self._item_neighbors[i]
                    neigh_w = self._item_weights[i]  # same length as neigh_idx

                    # compute weighted diff sum_j w_ij * (Q_i - Q_j)
                    Qj = self.Q[neigh_idx]  # shape (m, f)
                    diff = Qi - Qj  # broadcasted
                    # weight each row by w_ij and sum
                    sim_grad = np.sum(diff * neigh_w.reshape(-1,1), axis=0)  # shape (f,)

                    # the gradient for Q_i is 2 * sim_reg * sim_grad
                    # perform a gradient descent step on Q[i]
                    self.Q[i] -= self.lr * (2.0 * self.sim_reg) * sim_grad

                    # OPTIONAL: update neighbors with opposite gradient (symmetric)
                    # For simplicity and efficiency we *do not* update neighbors here.
                    # If desired, you could also apply a small symmetric push on neighbors.

    def predict(self, u:int, i:int) -> float:
        return float(self.mu + self.bu[u] + self.bi[i] + self.P[u].dot(self.Q[i]))

    def recommend_top_k(self, u:int, k:int=10, train_items:set=None):
        scores = (self.mu + self.bi + self.bu[u] + self.Q.dot(self.P[u])).tolist()
        # optionally filter out training items
        if train_items:
            for it in train_items:
                scores[it] = -1e9
        topk = np.argsort(scores)[::-1][:k]
        return topk.tolist()
