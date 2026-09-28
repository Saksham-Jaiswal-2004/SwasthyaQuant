"""Per-subgroup metrics -- our substitute for cross-hospital external validation.

The Kaggle 918-row merge dropped the source-hospital column, so genuine external
validation is off the table (see docs/DECISIONS.md). Slicing by sex and age band
recovers part of that: it answers "does this model work equally well for everyone
in the data", which is the clinically important half of the generalization
question, and it is cheap.

Report it because the dataset is strongly male-skewed. A model can post a
respectable pooled sensitivity while missing a much larger share of female
patients, and pooled numbers hide exactly that.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qheart.eval.metrics import HEADLINE, summary

__all__ = ["subgroup_table", "worst_subgroup", "disparity"]

MIN_N = 30      # below this, subgroup metrics are too noisy to report as findings


def subgroup_table(df: pd.DataFrame, y_true, y_prob, keys: list[str] | None = None,
                   threshold: float = 0.5, min_n: int = MIN_N) -> pd.DataFrame:
    """Metrics computed within each level of each subgroup key.

    ``df`` must be the test-set rows, aligned with ``y_true`` and ``y_prob``.
    Groups smaller than ``min_n`` are still returned but flagged ``reportable=False``
    so a noisy 12-patient slice never becomes a headline claim.
    """
    from qheart.data.audit import age_band

    y = np.asarray(y_true).ravel()
    p = np.asarray(y_prob, dtype=float).ravel()
    if not (len(df) == y.size == p.size):
        raise ValueError(f"length mismatch: df={len(df)}, y={y.size}, p={p.size}")

    work = df.reset_index(drop=True).copy()
    if "age_band" not in work.columns and "age" in work.columns:
        work["age_band"] = age_band(work["age"])

    keys = keys or [k for k in ("sex", "age_band") if k in work.columns]
    rows = []
    overall = summary(y, p, threshold)
    rows.append({"key": "ALL", "level": "ALL", "n": int(y.size),
                 "prevalence": float(y.mean()), "reportable": True,
                 **{m: overall[m] for m in ("sensitivity", "specificity", "ppv",
                                            "pr_auc", "roc_auc", "brier")}})

    for key in keys:
        for level, idx in work.groupby(key, observed=True).groups.items():
            i = np.asarray(idx, dtype=int)
            if i.size == 0:
                continue
            if np.unique(y[i]).size < 2:
                rows.append({"key": key, "level": str(level), "n": int(i.size),
                             "prevalence": float(y[i].mean()), "reportable": False,
                             "sensitivity": np.nan, "specificity": np.nan,
                             "ppv": np.nan, "pr_auc": np.nan, "roc_auc": np.nan,
                             "brier": np.nan})
                continue
            s = summary(y[i], p[i], threshold)
            rows.append({"key": key, "level": str(level), "n": int(i.size),
                         "prevalence": float(y[i].mean()),
                         "reportable": bool(i.size >= min_n),
                         **{m: s[m] for m in ("sensitivity", "specificity", "ppv",
                                              "pr_auc", "roc_auc", "brier")}})
    return pd.DataFrame(rows)


def worst_subgroup(table: pd.DataFrame, metric: str = "sensitivity",
                   reportable_only: bool = True) -> pd.Series:
    """The slice the model serves worst. Put this in the deck next to the mean."""
    t = table[table["key"] != "ALL"]
    if reportable_only:
        t = t[t["reportable"]]
    t = t.dropna(subset=[metric])
    if t.empty:
        raise ValueError("no reportable subgroups")
    return t.loc[t[metric].idxmin()]


def disparity(table: pd.DataFrame, metric: str = "sensitivity") -> dict[str, float]:
    """Max-minus-min gap in ``metric`` within each subgroup key.

    A gap wider than the fold-to-fold standard deviation of the pooled metric is a
    real finding and belongs in the report, not a footnote.
    """
    out: dict[str, float] = {}
    for key, grp in table[table["key"] != "ALL"].groupby("key"):
        g = grp[grp["reportable"]].dropna(subset=[metric])
        if len(g) >= 2:
            out[key] = float(g[metric].max() - g[metric].min())
    return out


def format_table(table: pd.DataFrame) -> str:
    cols = ["key", "level", "n", "prevalence", *HEADLINE, "ppv", "pr_auc", "brier"]
    t = table[cols].copy()
    for c in ("prevalence", *HEADLINE, "ppv", "pr_auc", "brier"):
        t[c] = t[c].map(lambda v: "-" if pd.isna(v) else f"{v:.3f}")
    flag = np.where(table["reportable"], "", "  (n too small)")
    t["level"] = t["level"].astype(str) + flag
    return t.to_markdown(index=False)
