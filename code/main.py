import argparse
import os
import numpy as np
from data_loader import load_ratings, remap_ids
from ibcf import ItemCF
from mf_sgd import MF_SGD
from evaluate import rmse, mae, precision_recall_at_k, ndcg_at_k
from scipy.sparse import csr_matrix
from collections import defaultdict
import random
from tqdm import tqdm
from similarity import compute_item_similarity_matrix


def train_test_split_per_user(df, test_ratio=0.2, seed=42):
    random.seed(seed)
    train = []
    test = []
    grouped = df.groupby('user')
    for u, group in grouped:
        rows = group.index.tolist()
        n_test = max(1, int(len(rows) * test_ratio))
        test_idx = set(random.sample(rows, n_test))
        for idx in rows:
            row = df.loc[idx]
            if idx in test_idx:
                test.append((row['user'], row['item'], row['rating']))
            else:
                train.append((row['user'], row['item'], row['rating']))
    return train, test


def build_sparse(n_users, n_items, interactions):
    rows = [u for u,i,r in interactions]
    cols = [i for u,i,r in interactions]
    data = [r for u,i,r in interactions]
    return csr_matrix((data,(rows,cols)), shape=(n_users,n_items))


def ground_truth_from_test(test):
    gt = defaultdict(set)
    for u,i,r in test:
        gt[u].add(i)
    return gt


def rating_eval(model, test):
    preds = []
    labels = []
    for u,i,r in test:
        try:
            p = model.predict(u,i)
        except Exception:
            p = None
        if p is None:
            # use global mean or zero
            p = 0.0
        preds.append(p)
        labels.append(r)
    return rmse(preds, labels), mae(preds, labels)


def topk_recommendations(model, train_interactions, users, k=10):
    train_items = defaultdict(set)
    for u,i,r in train_interactions:
        train_items[u].add(i)
    recs = {}
    for u in users:
        if hasattr(model, 'recommend_top_k'):
            recs[u] = model.recommend_top_k(u, k=k, train_items=train_items.get
            (u, set())) if 'mf_sgd' in model.__class__.__module__ else model.recommend_top_k(u, k=k)
        else:
            recs[u] = []
    return recs


def main(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    os.environ['OMP_NUM_THREADS'] = '1'        # limit OpenMP threads
    os.environ['MKL_NUM_THREADS'] = '1'       # limit MKL threads
    os.environ['NUMEXPR_NUM_THREADS'] = '1'

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    df_raw = load_ratings(args.ratings)

    # Filter out 0 values
    df_raw = df_raw[df_raw["rating"] > 0]

    df, user_map, item_map = remap_ids(df_raw)
    n_users = df['user'].nunique()
    n_items = df['item'].nunique()
    print(f"Loaded {len(df)} ratings, {n_users} users, {n_items} items.")

    train, test = train_test_split_per_user(df, test_ratio=args.test_ratio)
    print(f"Train interactions: {len(train)}, Test interactions: {len(test)}")

    # build sparse train matrix
    train_mat = build_sparse(n_users, n_items, train)

    # Optionally compute item-item similarity matrix for graph regularization
    item_sim = None
    if args.sim_reg > 0.0:
        # compute item similarity using similarity.py
        from similarity import compute_item_similarity_matrix
        print("Computing item-item similarity matrix (this may take some time)...")
        item_sim = compute_item_similarity_matrix(train_mat, k=args.sim_k)
        print("Item similarity computed. Top-k neighbors retained per item.")

    # choose method
    if args.method == 'ibcf':
        model = ItemCF(n_users=n_users, n_items=n_items, similarity=args.similarity, k=args.k)
        model.fit(train_mat)
    elif args.method == 'mf':
        model = MF_SGD(n_users=n_users, n_items=n_items, n_factors=args.n_factors, lr=args.lr, reg=args.reg, n_epochs=args.epochs)
        model.fit(train)
    if args.method == 'mf2':
        model = MF_SGD(n_users=n_users, n_items=n_items, n_factors=args.n_factors, lr=args.lr, reg=args.reg, n_epochs=args.epochs,
                       item_sim=item_sim, sim_reg=args.sim_reg)
        model.fit(train)
    else:
        raise ValueError("method must be 'ibcf' or 'mf'")

    # rating prediction evaluation (on test interactions)
    rmse_val, mae_val = rating_eval(model, test)
    print(f"Rating prediction: RMSE={rmse_val:.4f}, MAE={mae_val:.4f}")

    # top-k evaluation
    users = df['user'].unique().tolist()
    recs = topk_recommendations(model, train, users, k=args.k_eval)
    gt = ground_truth_from_test(test)
    prec, rec = precision_recall_at_k(recs, gt, args.k_eval)
    ndcg = ndcg_at_k(recs, gt, args.k_eval)
    print(f"Top-{args.k_eval}: Precision={prec:.4f}, Recall={rec:.4f}, NDCG={ndcg:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ratings', type=str, required=True, help='Path to ratings file (CSV or u.data)')
    parser.add_argument('--sep', type=str, default=None, help='Separator for ratings file; default auto')
    parser.add_argument('--method', type=str, choices=['ibcf','mf','mf2'], default='ibcf')
    parser.add_argument('--similarity', type=str, choices=['cosine','pearson'], default='cosine')
    parser.add_argument('--k', type=int, default=20, help='k for IBCF neighbors')
    parser.add_argument('--n_factors', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--reg', type=float, default=0.02)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--k_eval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--sim_reg", type=float, default=0.0, help="Weight for similarity-based graph regularization")
    parser.add_argument("--sim_k", type=int, default=40, help="Number of neighbors for similarity matrix")

    args = parser.parse_args()
    main(args)
