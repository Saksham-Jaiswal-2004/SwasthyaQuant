"""Report tables, with a gate that refuses to render a misleading one.

``refuse_if_incomplete`` raises if the main table omits sensitivity or specificity,
or if it contains a quantum model with no Control-C row beside it. That is deliberate
enforcement rather than documentation: under deadline pressure at 2 a.m. the
temptation is to ship the accuracy column and drop the control, and a rule that lives
only in a README does not survive that moment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from qheart.eval.metrics import HEADLINE

__all__ = ["main_table", "param_table", "refuse_if_incomplete", "to_markdown",
           "MAIN_METRICS"]

# Order matters: sensitivity first because in screening a missed case is the costly
# error, PR-AUC before ROC-AUC because ROC-AUC flatters a model under class
# imbalance, and accuracy last because it is the least informative number on the row.
MAIN_METRICS = ["sensitivity", "specificity", "ppv", "npv", "pr_auc", "roc_auc",
                "balanced_accuracy", "mcc", "brier", "ece", "accuracy"]


def refuse_if_incomplete(df: pd.DataFrame) -> None:
    """Raise unless the table meets the project's reporting rules."""
    cols = set(df.columns)
    missing = [m for m in HEADLINE if not any(c.startswith(m) for c in cols)]
    if missing:
        raise ValueError(
            f"main table is missing {missing}. Sensitivity and specificity are not "
            f"optional in a screening context: a model at 0.96 accuracy that misses a "
            f"third of cases is worthless, and the accuracy column hides that."
        )
    models = set(df["model"]) if "model" in df else set(df.index)
    quantum = {m for m in models
               if any(k in str(m) for k in ("qkernel", "vqc", "hybrid", "quantum"))}
    if quantum and not any("control" in str(m) for m in models):
        raise ValueError(
            f"table reports quantum models {sorted(quantum)} with no control_c row. "
            f"Without the parameter-matched control the comparison is unfalsifiable -- "
            f"which is exactly the gap in both reference papers. Run "
            f"`make control` and re-render."
        )
    if "majority_class_accuracy" not in df.attrs:
        print("NOTE: no majority-class baseline recorded in df.attrs. Quote accuracy "
              "against that floor or it reads as better than it is.")


def main_table(ledger: pd.DataFrame, metrics: list[str] | None = None,
               decimals: int = 3) -> pd.DataFrame:
    """Aggregate a fold-level ledger into one row per model, ``mean +/- sd``.

    Standard deviation across folds, with the caveat recorded in
    ``qheart.eval.metrics.aggregate``: folds share training data, so this understates
    true uncertainty. It is the honest available number, not a confidence interval,
    and must not be presented as one.
    """
    metrics = metrics or MAIN_METRICS
    have = [m for m in metrics if m in ledger.columns]
    g = ledger.groupby("model")
    out = pd.DataFrame(index=sorted(g.groups))
    for m in have:
        mu, sd = g[m].mean(), g[m].std(ddof=1)
        out[m] = [f"{mu[i]:.{decimals}f} +/- {sd[i]:.{decimals}f}" for i in out.index]
        out[f"{m}__mean"] = mu.reindex(out.index).values
    if "n_trainable_params" in ledger.columns:
        out["params"] = g["n_trainable_params"].first().reindex(out.index).values
    out["folds"] = g.size().reindex(out.index).values
    out.insert(0, "model", out.index)
    return out.reset_index(drop=True)


def param_table(rows: list[dict]) -> pd.DataFrame:
    """The parameter-efficiency table -- the one figure the whole project is built on.

    Two columns, side by side, labelled. Not a sentence buried in the discussion.
    """
    df = pd.DataFrame(rows)
    if "params" in df:
        df["bytes_fp32"] = df["params"] * 4
        ref = df["params"].max()
        df["x_smaller_than_largest"] = (ref / df["params"]).round(1)
    return df


def to_markdown(df: pd.DataFrame, drop_helpers: bool = True) -> str:
    d = df.drop(columns=[c for c in df.columns if c.endswith("__mean")]) if drop_helpers else df
    return d.to_markdown(index=False)
