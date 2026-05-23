"""
Auto Visualizer — generates all standard EDA plots as Plotly JSON for the frontend.
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy import stats
import json
import warnings
warnings.filterwarnings("ignore")


PALETTE = px.colors.qualitative.Bold
BG = "#0d1117"
PAPER = "#161b22"
GRID = "#21262d"
TEXT = "#e6edf3"
ACCENT = "#58a6ff"


def _base_layout(title="", height=420):
    return dict(
        title=dict(text=title, font=dict(color=TEXT, size=15, family="JetBrains Mono, monospace")),
        paper_bgcolor=PAPER,
        plot_bgcolor=BG,
        font=dict(color=TEXT, family="JetBrains Mono, monospace", size=11),
        height=height,
        margin=dict(l=50, r=30, t=50, b=50),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    )


def fig_to_json(fig) -> dict:
    return json.loads(fig.to_json())


class AutoVisualizer:
    def __init__(self, original_df: pd.DataFrame, processed_df: pd.DataFrame, col_types: dict):
        self.orig = original_df
        self.proc = processed_df
        self.col_types = col_types
        self.numeric_cols = [c for c in original_df.columns if pd.api.types.is_numeric_dtype(original_df[c])]
        self.cat_cols = [c for c, t in col_types.items() if t in ("categorical", "boolean") and c in original_df.columns]

    def dataset_overview(self) -> dict:
        df = self.orig
        missing = df.isna().sum()
        missing_pct = (df.isna().mean() * 100).round(2)
        return {
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "missing_cells": int(missing.sum()),
            "missing_pct": round(df.isna().mean().mean() * 100, 2),
            "duplicates": int(df.duplicated().sum()),
            "numeric_cols": len(self.numeric_cols),
            "cat_cols": len(self.cat_cols),
            "col_missing": {c: {"count": int(missing[c]), "pct": float(missing_pct[c])} for c in df.columns},
        }

    def missing_heatmap(self) -> dict:
        df = self.orig
        cols_with_missing = [c for c in df.columns if df[c].isna().any()]
        if not cols_with_missing:
            return None

        sample = df[cols_with_missing].head(200)
        z = sample.isna().astype(int).T.values.tolist()
        fig = go.Figure(go.Heatmap(
            z=z,
            x=list(range(len(sample))),
            y=cols_with_missing,
            colorscale=[[0, BG], [1, "#f85149"]],
            showscale=False,
        ))
        fig.update_layout(**_base_layout("Missing Value Map (red = missing)", height=max(300, len(cols_with_missing) * 28)))
        return fig_to_json(fig)

    def distributions(self) -> list[dict]:
        """Histogram + KDE for each numeric column."""
        plots = []
        for col in self.numeric_cols[:12]:  # cap at 12
            series = self.orig[col].dropna()
            if len(series) < 3:
                continue

            fig = make_subplots(rows=1, cols=2, subplot_titles=["Distribution", "Box Plot"])

            # Histogram
            fig.add_trace(go.Histogram(
                x=series, name=col,
                marker_color=ACCENT, opacity=0.75,
                nbinsx=min(50, max(10, len(series) // 10)),
            ), row=1, col=1)

            # Box
            fig.add_trace(go.Box(
                y=series, name=col,
                marker_color=ACCENT,
                boxmean='sd',
                line_color=ACCENT,
            ), row=1, col=2)

            layout = _base_layout(f"{col} — Distribution & Box", height=360)
            layout.pop("xaxis", None)
            layout.pop("yaxis", None)
            layout["showlegend"] = False
            layout["xaxis"] = dict(gridcolor=GRID)
            layout["yaxis"] = dict(gridcolor=GRID)
            layout["xaxis2"] = dict(gridcolor=GRID)
            layout["yaxis2"] = dict(gridcolor=GRID)
            fig.update_layout(**layout)
            plots.append({"col": col, "fig": fig_to_json(fig)})
        return plots

    def correlation_heatmap(self) -> dict:
        num_df = self.orig[self.numeric_cols].dropna()
        if len(self.numeric_cols) < 2:
            return None

        corr = num_df.corr().round(3)
        mask = np.triu(np.ones_like(corr, dtype=bool))
        corr_masked = corr.mask(mask)

        cols = list(corr.columns)
        fig = go.Figure(go.Heatmap(
            z=corr_masked.values,
            x=cols, y=cols,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=corr_masked.round(2).values,
            texttemplate="%{text}",
            textfont=dict(size=9),
        ))
        h = max(400, len(cols) * 35)
        fig.update_layout(**_base_layout("Correlation Heatmap", height=h))
        return fig_to_json(fig)

    def pairplot_sample(self) -> dict:
        """Scatter matrix for up to 5 numeric columns."""
        cols = self.numeric_cols[:5]
        if len(cols) < 2:
            return None

        sub = self.orig[cols].dropna()
        df_sample = sub.sample(min(500, len(sub)), random_state=42)
        fig = px.scatter_matrix(df_sample, dimensions=cols,
                                color_discrete_sequence=[ACCENT])
        fig.update_traces(diagonal_visible=True, showupperhalf=False, marker=dict(size=3, opacity=0.5))
        fig.update_layout(**_base_layout("Pair Plot (sample 500)", height=600))
        return fig_to_json(fig)

    def categorical_plots(self) -> list[dict]:
        """Bar charts for categorical columns."""
        plots = []
        for col in self.cat_cols[:8]:
            vc = self.orig[col].value_counts().head(20)
            fig = go.Figure(go.Bar(
                x=vc.index.astype(str), y=vc.values,
                marker_color=PALETTE[:len(vc)],
                text=vc.values,
                textposition="outside",
            ))
            fig.update_layout(**_base_layout(f"{col} — Value Counts", height=360))
            plots.append({"col": col, "fig": fig_to_json(fig)})
        return plots

    def outlier_plot(self) -> dict:
        """Combined box plots for all numeric columns (normalized)."""
        cols = self.numeric_cols[:10]
        if not cols:
            return None

        df_norm = self.orig[cols].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
        fig = go.Figure()
        for i, col in enumerate(cols):
            fig.add_trace(go.Box(
                y=df_norm[col].dropna(),
                name=col,
                marker_color=PALETTE[i % len(PALETTE)],
                boxmean=True,
            ))
        fig.update_layout(**_base_layout("Outlier Detection (Z-normalized Box Plots)", height=420))
        return fig_to_json(fig)

    def skewness_plot(self) -> dict:
        """Horizontal bar chart of skewness per numeric column."""
        if not self.numeric_cols:
            return None

        skews = {c: round(self.orig[c].skew(), 3) for c in self.numeric_cols if self.orig[c].notna().sum() > 2}
        skews = dict(sorted(skews.items(), key=lambda x: abs(x[1]), reverse=True))

        colors = ["#f85149" if abs(v) > 1 else "#3fb950" if abs(v) < 0.5 else "#d29922" for v in skews.values()]

        fig = go.Figure(go.Bar(
            x=list(skews.values()),
            y=list(skews.keys()),
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in skews.values()],
            textposition="outside",
        ))
        fig.add_vline(x=0, line_color=TEXT, line_width=1)
        fig.add_vline(x=1, line_color="#f85149", line_dash="dash", line_width=1)
        fig.add_vline(x=-1, line_color="#f85149", line_dash="dash", line_width=1)
        h = max(350, len(skews) * 30)
        fig.update_layout(**_base_layout("Skewness per Column  (|skew|>1 = red)", height=h))
        return fig_to_json(fig)

    def target_analysis(self, target_col: str) -> list[dict]:
        """If target provided: show distribution + relationship with top features."""
        plots = []
        if target_col not in self.orig.columns:
            return plots

        series = self.orig[target_col].dropna()

        # Target distribution
        if pd.api.types.is_numeric_dtype(series):
            fig = go.Figure(go.Histogram(x=series, marker_color="#bc8cff", nbinsx=40))
            fig.update_layout(**_base_layout(f"Target: {target_col} Distribution", height=360))
        else:
            vc = series.value_counts()
            fig = go.Figure(go.Bar(x=vc.index.astype(str), y=vc.values, marker_color=PALETTE))
            fig.update_layout(**_base_layout(f"Target: {target_col} Distribution", height=360))
        plots.append({"col": target_col, "label": "Target Distribution", "fig": fig_to_json(fig)})

        # Scatter vs top numeric features (correlation)
        if pd.api.types.is_numeric_dtype(series):
            other_num = [c for c in self.numeric_cols if c != target_col]
            corrs = {c: abs(self.orig[[c, target_col]].dropna().corr().iloc[0, 1]) for c in other_num[:10]}
            top_feats = sorted(corrs, key=corrs.get, reverse=True)[:4]

            for feat in top_feats:
                sub = self.orig[[feat, target_col]].dropna()
                sub = sub.sample(min(500, len(sub)), random_state=42)
                fig = px.scatter(sub, x=feat, y=target_col, trendline="ols",
                                 color_discrete_sequence=[ACCENT],
                                 trendline_color_override="#f85149")
                fig.update_layout(**_base_layout(f"{feat} vs {target_col}  (r={corrs[feat]:.3f})", height=360))
                plots.append({"col": feat, "label": f"vs Target", "fig": fig_to_json(fig)})

        return plots

    def generate_all(self, target_col: str = None) -> dict:
        return {
            "overview": self.dataset_overview(),
            "missing_heatmap": self.missing_heatmap(),
            "distributions": self.distributions(),
            "correlation": self.correlation_heatmap(),
            "pairplot": self.pairplot_sample(),
            "categorical": self.categorical_plots(),
            "outliers": self.outlier_plot(),
            "skewness": self.skewness_plot(),
            "target": self.target_analysis(target_col) if target_col else [],
        }
