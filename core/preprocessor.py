"""
Auto Preprocessor — detects column types and applies smart transformations.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


class AutoPreprocessor:
    def __init__(self):
        self.report = {}
        self.transformations = []
        self.encoders = {}
        self.scalers = {}

    def analyze_column_types(self, df: pd.DataFrame) -> dict:
        """Classify each column into: numeric, categorical, datetime, text, boolean."""
        col_types = {}
        for col in df.columns:
            series = df[col].dropna()
            if series.empty:
                col_types[col] = "empty"
                continue

            # Boolean
            if series.dtype == bool or set(series.unique()).issubset({0, 1, True, False, "True", "False", "true", "false", "yes", "no", "Yes", "No"}):
                col_types[col] = "boolean"
                continue

            # Datetime
            if series.dtype == "datetime64[ns]":
                col_types[col] = "datetime"
                continue
            if series.dtype == object:
                try:
                    pd.to_datetime(series.head(20), infer_datetime_format=True)
                    col_types[col] = "datetime"
                    continue
                except Exception:
                    pass

            # Numeric
            if pd.api.types.is_numeric_dtype(series):
                col_types[col] = "numeric"
                continue

            # Categorical vs Text
            if series.dtype == object:
                unique_ratio = series.nunique() / len(series)
                avg_len = series.astype(str).str.len().mean()
                if unique_ratio < 0.5 or series.nunique() <= 20:
                    col_types[col] = "categorical"
                else:
                    col_types[col] = "text" if avg_len > 30 else "categorical"
                continue

            col_types[col] = "unknown"

        return col_types

    def detect_outliers(self, series: pd.Series) -> dict:
        """Detect outliers using IQR and Z-score methods."""
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        iqr_outliers = ((series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))).sum()

        z_scores = np.abs(stats.zscore(series.dropna()))
        z_outliers = (z_scores > 3).sum()

        return {
            "iqr_count": int(iqr_outliers),
            "z_score_count": int(z_outliers),
            "iqr_lower": round(q1 - 1.5 * iqr, 4),
            "iqr_upper": round(q3 + 1.5 * iqr, 4),
        }

    def compute_missing_strategy(self, series: pd.Series, col_type: str) -> str:
        """Decide best imputation strategy per column."""
        missing_pct = series.isna().mean()
        if missing_pct == 0:
            return "none"
        if missing_pct > 0.6:
            return "drop_column"
        if col_type == "numeric":
            skew = series.skew()
            return "median" if abs(skew) > 1 else "mean"
        return "mode"

    def preprocess(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Full auto-preprocessing pipeline. Returns cleaned df + report."""
        original_shape = df.shape
        report = {
            "original_shape": original_shape,
            "steps": [],
            "column_types": {},
            "missing_info": {},
            "outlier_info": {},
            "dropped_columns": [],
            "encoded_columns": [],
            "scaled_columns": [],
            "new_features": [],
        }

        df = df.copy()

        # ── Step 1: Duplicate rows ──────────────────────────────────────────
        dupes = df.duplicated().sum()
        if dupes > 0:
            df = df.drop_duplicates()
            report["steps"].append(f"Removed {dupes} duplicate rows.")

        # ── Step 2: Detect column types ────────────────────────────────────
        col_types = self.analyze_column_types(df)
        report["column_types"] = col_types

        # ── Step 3: Handle missing values ──────────────────────────────────
        for col in df.columns:
            missing = df[col].isna().sum()
            missing_pct = round(df[col].isna().mean() * 100, 2)
            report["missing_info"][col] = {"count": int(missing), "pct": missing_pct}

            if missing == 0:
                continue

            ctype = col_types.get(col, "unknown")
            strategy = self.compute_missing_strategy(df[col], ctype)

            if strategy == "drop_column":
                df.drop(columns=[col], inplace=True)
                report["dropped_columns"].append(col)
                report["steps"].append(f"Dropped '{col}' — {missing_pct}% missing.")
                continue

            if strategy == "mean":
                df[col].fillna(df[col].mean(), inplace=True)
                report["steps"].append(f"'{col}': filled {missing} NaNs with mean.")
            elif strategy == "median":
                df[col].fillna(df[col].median(), inplace=True)
                report["steps"].append(f"'{col}': filled {missing} NaNs with median (skewed).")
            elif strategy == "mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
                report["steps"].append(f"'{col}': filled {missing} NaNs with mode.")

        # Refresh col_types after drops
        col_types = self.analyze_column_types(df)
        report["column_types"] = col_types

        # ── Step 4: Datetime feature extraction ────────────────────────────
        for col, ctype in col_types.items():
            if ctype == "datetime" and col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                    df.drop(columns=[col], inplace=True)
                    new_feats = [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"]
                    report["new_features"].extend(new_feats)
                    report["steps"].append(f"Extracted year/month/day/dayofweek from '{col}'.")
                except Exception:
                    pass

        # Refresh col_types
        col_types = self.analyze_column_types(df)

        # ── Step 5: Outlier detection (report only, no removal) ────────────
        for col, ctype in col_types.items():
            if ctype == "numeric" and col in df.columns:
                outlier_info = self.detect_outliers(df[col].dropna())
                report["outlier_info"][col] = outlier_info

        # ── Step 6: Encode categoricals ────────────────────────────────────
        for col, ctype in col_types.items():
            if ctype in ("categorical", "boolean") and col in df.columns:
                n_unique = df[col].nunique()
                if n_unique <= 10:
                    # One-hot encode low-cardinality
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
                    report["encoded_columns"].append(f"{col} (one-hot, {n_unique} cats)")
                    report["steps"].append(f"One-hot encoded '{col}' ({n_unique} unique).")
                else:
                    # Label encode high-cardinality
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                    report["encoded_columns"].append(f"{col} (label, {n_unique} cats)")
                    report["steps"].append(f"Label encoded '{col}' (high cardinality: {n_unique}).")

        # ── Step 7: Scale numerics ─────────────────────────────────────────
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() > 2]
        for col in numeric_cols:
            skew = df[col].skew()
            if abs(skew) > 1:
                scaler = MinMaxScaler()
                method = "minmax"
            else:
                scaler = StandardScaler()
                method = "standard"
            df[col] = scaler.fit_transform(df[[col]])
            self.scalers[col] = scaler
            report["scaled_columns"].append(f"{col} ({method})")

        if numeric_cols:
            report["steps"].append(f"Scaled {len(numeric_cols)} numeric columns (StandardScaler or MinMaxScaler based on skew).")

        report["final_shape"] = df.shape
        report["columns_final"] = list(df.columns)
        self.report = report
        return df, report
