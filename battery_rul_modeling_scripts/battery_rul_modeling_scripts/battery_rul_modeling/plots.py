from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_pred_vs_true(pred_df: pd.DataFrame, path: Path, title: str):
    y = pd.to_numeric(pred_df["y_true"], errors="coerce")
    yp = pd.to_numeric(pred_df["y_pred"], errors="coerce")
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(y, yp, s=8, alpha=0.35)
    finite = np.isfinite(y) & np.isfinite(yp)
    if finite.any():
        mn = float(min(y[finite].min(), yp[finite].min()))
        mx = float(max(y[finite].max(), yp[finite].max()))
        plt.plot([mn, mx], [mn, mx], linestyle="--", linewidth=1.2)
    plt.xlabel("RUL real (ciclos)")
    plt.ylabel("RUL predito (ciclos)")
    plt.title(title)
    _savefig(path)


def plot_residuals(pred_df: pd.DataFrame, path: Path, title: str):
    y = pd.to_numeric(pred_df["y_true"], errors="coerce")
    err = pd.to_numeric(pred_df["error"], errors="coerce")
    plt.figure(figsize=(6.5, 5.0))
    plt.scatter(y, err, s=8, alpha=0.35)
    plt.axhline(0, linestyle="--", linewidth=1.2)
    plt.xlabel("RUL real (ciclos)")
    plt.ylabel("Erro assinado (predito - real)")
    plt.title(title)
    _savefig(path)


def plot_bar_metric(df: pd.DataFrame, label_col: str, metric_col: str, path: Path, title: str, top_n: int = 30):
    if df.empty or label_col not in df.columns or metric_col not in df.columns:
        return
    data = df[[label_col, metric_col]].copy()
    data[metric_col] = pd.to_numeric(data[metric_col], errors="coerce")
    data = data.dropna().sort_values(metric_col, ascending=False).head(top_n)
    if data.empty:
        return
    plt.figure(figsize=(max(7, min(14, 0.35 * len(data))), 5.5))
    plt.bar(data[label_col].astype(str), data[metric_col])
    plt.xticks(rotation=75, ha="right")
    plt.ylabel(metric_col)
    plt.title(title)
    _savefig(path)
