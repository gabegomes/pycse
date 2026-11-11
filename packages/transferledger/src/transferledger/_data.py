"""Turning a table into the numeric design the models use, the same way at fit and predict."""

from __future__ import annotations

import numpy as np
import pandas as pd


class Design:
    """Column bookkeeping for a table or array.

    Numeric and boolean columns pass through. A text or categorical column becomes one 0/1
    column per level, named ``column=level``. Missing values are an error that names the
    columns: an imputed value would enter the equation as if it had been measured.

    Attributes:
        names_: the names of the numeric design's columns.
        source_: for each design column, the name of the input column it came from.
        feature_names_in_: the input column names.
    """

    def __init__(self, feature_names=None):
        self.feature_names = feature_names

    def _frame(self, X):
        if isinstance(X, pd.DataFrame):
            return X.reset_index(drop=True)
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be two-dimensional, got shape {X.shape}")
        names = (
            self.feature_names
            if self.feature_names is not None
            else [f"x{i}" for i in range(X.shape[1])]
        )
        if len(names) != X.shape[1]:
            raise ValueError(f"{len(names)} feature names for {X.shape[1]} columns")
        return pd.DataFrame(X, columns=list(names))

    @staticmethod
    def _check_missing(df):
        bad = df.isna().sum()
        bad = bad[bad > 0]
        if len(bad):
            listing = ", ".join(f"{c} ({n})" for c, n in bad.items())
            raise ValueError(
                f"missing values in: {listing}. Drop those rows or impute them explicitly before "
                "fitting; a filled-in value would enter the equation as if it had been measured."
            )

    def fit(self, X):
        df = self._frame(X)
        self._check_missing(df)
        self.feature_names_in_ = [str(c) for c in df.columns]
        self.levels_ = {}
        self.names_, self.source_ = [], []
        for c in df.columns:
            col = df[c]
            if pd.api.types.is_bool_dtype(col) or pd.api.types.is_numeric_dtype(col):
                self.names_.append(str(c))
                self.source_.append(str(c))
            else:
                levels = [v for v in pd.unique(col.astype(str))]
                self.levels_[str(c)] = sorted(levels)
                for v in self.levels_[str(c)]:
                    self.names_.append(f"{c}={v}")
                    self.source_.append(str(c))
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, X):
        df = self._frame(X)
        if [str(c) for c in df.columns] != self.feature_names_in_:
            missing = [c for c in self.feature_names_in_ if c not in {str(k) for k in df.columns}]
            if missing or len(df.columns) != len(self.feature_names_in_):
                raise ValueError(f"columns differ from those at fit; missing: {missing}")
            df = df[[c for c in self.feature_names_in_]]
        self._check_missing(df)
        cols, unseen = [], np.zeros(len(df), bool)
        for c in self.feature_names_in_:
            col = df[c]
            if c in self.levels_:
                s = col.astype(str).to_numpy()
                unseen |= ~np.isin(s, self.levels_[c])
                for v in self.levels_[c]:
                    cols.append((s == v).astype(float))
            else:
                cols.append(col.to_numpy(dtype=float))
        self.unseen_level_ = unseen
        return np.column_stack(cols) if cols else np.zeros((len(df), 0))

    def fit_transform(self, X):
        return self.fit(X).transform(X)
