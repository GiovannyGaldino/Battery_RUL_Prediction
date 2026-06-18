from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class NumericFeatureCleaner(BaseEstimator, TransformerMixin):
    """Select numeric columns using training data only and drop high-missing/constant features."""

    def __init__(self, max_missing_frac: float = 0.60, min_variance: float = 1e-12):
        self.max_missing_frac = max_missing_frac
        self.min_variance = min_variance
        self.feature_names_in_: List[str] = []
        self.feature_names_out_: List[str] = []
        self.drop_report_: pd.DataFrame | None = None

    def fit(self, X, y=None):
        X_df = self._to_dataframe(X)
        self.feature_names_in_ = list(X_df.columns)
        rows = []
        selected = []
        for col in X_df.columns:
            s = pd.to_numeric(X_df[col], errors="coerce")
            missing_frac = float(s.isna().mean())
            nunique = int(s.nunique(dropna=True))
            var = float(s.var(skipna=True)) if nunique > 1 else 0.0
            reason = "kept"
            keep = True
            if missing_frac > self.max_missing_frac:
                keep = False
                reason = "too_many_missing"
            elif nunique <= 1:
                keep = False
                reason = "constant_or_all_nan"
            elif not np.isfinite(var) or var <= self.min_variance:
                keep = False
                reason = "near_zero_variance"
            if keep:
                selected.append(col)
            rows.append({
                "feature": col,
                "missing_frac_train": missing_frac,
                "nunique_train": nunique,
                "variance_train": var,
                "status": reason,
            })
        if not selected:
            raise ValueError("Nenhuma feature numérica permaneceu após limpeza por missing/constância.")
        self.feature_names_out_ = selected
        self.drop_report_ = pd.DataFrame(rows)
        return self

    def transform(self, X):
        X_df = self._to_dataframe(X)
        for col in self.feature_names_out_:
            if col not in X_df.columns:
                X_df[col] = np.nan
        out = X_df[self.feature_names_out_].apply(pd.to_numeric, errors="coerce")
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_, dtype=object)

    @staticmethod
    def _to_dataframe(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)
