import pandas as pd
import numpy as np

def load_ratings(path: str):
    # MovieLens 100K: u.data file (tab-separated)
    if path.endswith(".data") or "u.data" in path:
        df = pd.read_csv(
            path,
            sep="\t",
            names=["userId", "itemId", "rating", "timestamp"],
            header=None,
            engine="python"
        )
        return df[["userId", "itemId", "rating"]]

    # MovieLens 1M: ratings.dat ( "::" separated )
    if path.endswith("ratings.dat"):
        df = pd.read_csv(
            path,
            sep="::",
            engine="python",
            header=None,
            names=["userId", "itemId", "rating", "timestamp"]
        )
        return df[["userId", "itemId", "rating"]]

    raise ValueError("Unsupported file. Only MovieLens 100K and 1M are supported.")


def remap_ids(df: pd.DataFrame):
    """Maps userId and itemId to contiguous 0..n-1 integers."""
    users = np.sort(df["userId"].unique())
    items = np.sort(df["itemId"].unique())

    user_map = {u: i for i, u in enumerate(users)}
    item_map = {i: j for j, i in enumerate(items)}

    df2 = df.copy()
    df2["user"] = df2["userId"].map(user_map)
    df2["item"] = df2["itemId"].map(item_map)

    return df2[["user", "item", "rating"]], user_map, item_map
