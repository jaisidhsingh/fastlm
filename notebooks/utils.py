import os
import json
import math
from copy import deepcopy
import numpy as np
import pandas as pd
from collections import defaultdict


def create_store_csv():
    folder = "./data"
    data = defaultdict(list)
    
    for arch_id in os.listdir(folder):
        for n in os.listdir(os.path.join(folder, arch_id)):
            for gbs_str in os.listdir(os.path.join(folder, arch_id, n)):
                gbs = int(gbs_str.split("_")[-1])
                for lr_str in os.listdir(os.path.join(folder, arch_id, n, gbs_str)):
                    lr = float(lr_str.split("_")[-1].replace("p", "."))
                    for m_str in os.listdir(os.path.join(folder, arch_id, n, gbs_str, lr_str)):
                        d = m_str.split("_")[-1].split(".json")[0].replace("p", ".")
                        
                        mpath = os.path.join(folder, arch_id, n, gbs_str, lr_str, m_str)
                        with open(mpath) as f:
                            mdata = json.load(f)
                        
                        data["arch_id"].append(arch_id)
                        data["n"].append(n)
                        data["d"].append(d)
                        data["gbs"].append(gbs)
                        data["lr"].append(lr)
                        data['train_loss'].append(mdata['train/loss'][-1])
                        data['val_loss'].append(mdata['valid/loss'][-1])
    
    df = pd.DataFrame(data)
    df.to_csv("./store.csv") 
    


class Collector:
    def __init__(self, arch_id, path="./store.csv", limits=None):
        self.arch_id = arch_id

        self.df = pd.read_csv(path)
        self.df = self.df[self.df["arch_id"] == arch_id].copy()

        self._lr_fit_cache = {}
        self._gbs_fit_cache = {}

        # Optional filtering by limits
        if limits is not None:
            for col, (lo, hi) in limits.items():
                self.df = self.df[(self.df[col] >= lo) & (self.df[col] <= hi)]
    
    def index(self, n=None, d=None, gbs=None, lr=None):
        df = deepcopy(self.df)
        if n is not None:
            df = df[df["n"] == n]
        if d is not None:
            df = df[df["d"] == d]
        if gbs is not None:
            df = df[df["gbs"] == gbs]
        if lr is not None:
            df = df[df["lr"] == lr]
        return df

    def _best(self, filters, group_cols, loss_type):
        assert loss_type in ("train", "val")

        loss_col = f"{loss_type}_loss"

        df = self.df
        for col, val in filters.items():
            df = df[df[col] == val]

        if len(df) == 0:
            return None

        best = (
            df.groupby(group_cols, as_index=False)[loss_col]
            .mean()  # or .min() if each config only appears once
            .sort_values(loss_col)
            .iloc[0]
        )

        params = tuple(best[c] for c in group_cols)
        loss = best[loss_col]

        return params, loss

    def best_gbs_and_lr(self, n, d, loss_type="val"):
        return self._best(
            filters={"n": n, "d": d},
            group_cols=["gbs", "lr"],
            loss_type=loss_type,
        )

    def best_lr(self, n, d, gbs, loss_type="val"):
        return self._best(
            filters={"n": n, "d": d, "gbs": gbs},
            group_cols=["lr"],
            loss_type=loss_type,
        )

    def best_gbs(self, n, d, lr, loss_type="val"):
        return self._best(
            filters={"n": n, "d": d, "lr": lr},
            group_cols=["gbs"],
            loss_type=loss_type,
        )

    def _filter(self, n=None, d=None, gbs=None, lr=None):
        df = self.df

        if n is not None:
            df = df[df["n"] == n]
        if d is not None:
            df = df[df["d"] == d]
        if gbs is not None:
            df = df[df["gbs"] == gbs]
        if lr is not None:
            df = df[df["lr"] == lr]

        return df

    def _quad_fit_log2(self, x, y):
        # fit y = ax^2 + bx + c
        coeffs = np.polyfit(x, y, 2)
        return coeffs  # a, b, c

    def _quad_min(self, coeffs):
        a, b, c = coeffs
        if a == 0:
            return None, None

        x_star = -b / (2 * a)
        y_star = a * x_star**2 + b * x_star + c
        return x_star, y_star

    def lr_sweep_fit(self, n, d, gbs, loss_type="val"):
        key = (n, d, gbs, loss_type)
        if key in self._lr_fit_cache:
            return self._lr_fit_cache[key]

        df = self._filter(n=n, d=d, gbs=gbs)

        loss_col = f"{loss_type}_loss"

        grouped = df.groupby("lr")[loss_col].mean().reset_index()
        grouped = grouped.dropna()

        x = np.log2(grouped["lr"].values)
        y = grouped[loss_col].values

        coeffs = self._quad_fit_log2(x, y)
        self._lr_fit_cache[key] = (grouped, coeffs)

        return grouped, coeffs

    def optimal_lr(self, n, d, gbs, loss_type="val"):
        grouped, coeffs = self.lr_sweep_fit(n, d, gbs, loss_type)

        x_star, y_star = self._quad_min(coeffs)
        lr_star = float(2**x_star) if x_star is not None else None

        return lr_star, y_star

    def gbs_sweep_fit(self, n, d, lr, loss_type="val"):
        key = (n, d, lr, loss_type)
        if key in self._gbs_fit_cache:
            return self._gbs_fit_cache[key]

        df = self._filter(n=n, d=d, lr=lr)

        loss_col = f"{loss_type}_loss"

        grouped = df.groupby("gbs")[loss_col].mean().reset_index()
        grouped = grouped.dropna()

        x = np.log2(grouped["gbs"].values)
        y = grouped[loss_col].values

        coeffs = self._quad_fit_log2(x, y)
        self._gbs_fit_cache[key] = (grouped, coeffs)

        return grouped, coeffs

    def optimal_gbs(self, n, d, lr, loss_type="val"):
        grouped, coeffs = self.gbs_sweep_fit(n, d, lr, loss_type)

        x_star, y_star = self._quad_min(coeffs)
        gbs_star = float(2**x_star) if x_star is not None else None

        return gbs_star, y_star


def quad_log2_fit(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        raise ValueError("Need at least 3 points for quadratic fit")

    # transform to log2 space
    x_log = np.log2(x)

    # quadratic fit
    coeffs = np.polyfit(x_log, y, 2)
    a, b, c = coeffs

    # smooth curve
    x_fit_log = np.linspace(x_log.min(), x_log.max(), 200)
    y_fit = a * x_fit_log**2 + b * x_fit_log + c
    x_fit = 2 ** x_fit_log
    return x_fit, y_fit



if __name__ == "__main__":
    create_store_csv()
    
