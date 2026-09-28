"""Data audit for the 70,000-row cardiovascular dataset.

Run this before model training.

The audit checks:
- class balance and majority-class baseline
- exact and feature-identical duplicates
- contradictory labels
- missingness
- numerical ranges
- categorical distributions
- clinically meaningful subgroups
- a label-noise / duplicate-based accuracy ceiling

No model fitting or learned preprocessing happens here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from qheart import schema as S

__all__ = ["audit", "audit_report", "age_group"]


def age_group(age_years: pd.Series) -> pd.Series:
    """Bucket age in years for subgroup reporting."""
    bins = [-np.inf, 40, 50, 60, 70, np.inf]
    labels = ["young", "middle", "older", "senior", "elderly"]

    out = pd.cut(
        age_years,
        bins=bins,
        labels=labels,
        right=False,
    )

    return out.cat.add_categories(["unknown"]).fillna("unknown")


def audit(df: pd.DataFrame) -> dict[str, Any]:
    """Return a dataset audit report for a schema-conformant frame."""
    S.validate(df, require_target=True)

    n = len(df)
    y = df[S.TARGET].to_numpy()

    dup_all = int(df.duplicated().sum())

    feat_only = df[S.FEATURES]
    dup_feat = int(feat_only.duplicated().sum())

    # Feature-identical rows whose labels disagree.
    grp = df.groupby(
        list(S.FEATURES),
        observed=True,
        dropna=False,
    )[S.TARGET]

    contradictory_groups = int((grp.nunique() > 1).sum())
    contradictory_rows = int(
        grp.transform("nunique").gt(1).sum()
    )

    missing = {
        c: int(df[c].isna().sum())
        for c in S.FEATURES
    }

    numeric_stats: dict[str, dict[str, Any]] = {}

    for c in S.NUMERIC:
        s = df[c].dropna()

        numeric_stats[c] = {
            "n": int(s.size),
            "min": float(s.min()) if s.size else None,
            "p50": float(s.median()) if s.size else None,
            "max": float(s.max()) if s.size else None,
            "mean": float(s.mean()) if s.size else None,
            "std": float(s.std(ddof=1)) if s.size > 1 else None,
        }

    cats = {
        c: {
            str(k): int(v)
            for k, v in df[c].value_counts(
                dropna=False
            ).items()
        }
        for c in S.CATEGORICAL
    }

    # Age subgroup is based on the raw age converted to years.
    age_years = pd.to_numeric(df["age"], errors="coerce") / 365.25
    band = age_group(age_years)

    subgroups = {
        "gender": {
            str(k): int(v)
            for k, v in df["gender"]
            .value_counts(dropna=False)
            .items()
        },
        "age_group": {
            str(k): int(v)
            for k, v in band.value_counts().items()
        },
    }

    prevalence_by = {
        "gender": {
            str(k): float(v)
            for k, v in df.groupby(
                "gender",
                observed=True,
            )[S.TARGET].mean().items()
        },
        "age_group": {
            str(k): float(v)
            for k, v in df.groupby(
                band,
                observed=True,
            )[S.TARGET].mean().items()
        },
    }

    pos = int(y.sum())

    report: dict[str, Any] = {
        "source": df.attrs.get("source", "unknown"),
        "n_rows": n,
        "n_features": len(S.FEATURES),

        "class_balance": {
            "positive": pos,
            "negative": n - pos,
            "prevalence": float(pos / n) if n else None,
            "majority_class_accuracy": (
                float(max(pos, n - pos) / n)
                if n
                else None
            ),
        },

        "sentinel_zeros_nulled": (
            df.attrs.get("sentinel_zeros_nulled", {})
        ),

        "impossible_values_nulled": (
            df.attrs.get("impossible_values_nulled", {})
        ),

        "missing_after_cleaning": missing,

        "missing_pct": {
            k: (round(100 * v / n, 2) if n else None)
            for k, v in missing.items()
        },

        "duplicates": {
            "exact_rows": dup_all,
            "feature_identical_rows": dup_feat,
            "contradictory_groups": contradictory_groups,
            "contradictory_rows": contradictory_rows,
            "label_noise_ceiling_hint": _ceiling(df),
        },

        "numeric": numeric_stats,
        "categorical": cats,
        "subgroups": subgroups,
        "prevalence_by_subgroup": prevalence_by,
    }

    report["warnings"] = _warnings(report)

    return report


def _ceiling(df: pd.DataFrame) -> float | None:
    """Estimate the accuracy ceiling from contradictory feature groups."""
    if df.empty:
        return None

    g = df.groupby(
        list(S.FEATURES),
        observed=True,
        dropna=False,
    )[S.TARGET]

    correct = g.agg(
        lambda s: max(
            int((s == 0).sum()),
            int((s == 1).sum()),
        )
    ).sum()

    return round(float(correct / len(df)), 4)


def _warnings(r: dict[str, Any]) -> list[str]:
    """Generate audit warnings."""
    warnings: list[str] = []

    for col, n in r["sentinel_zeros_nulled"].items():
        if n:
            pct = 100 * n / r["n_rows"]
            warnings.append(
                f"{col}: {n} rows ({pct:.1f}%) had a sentinel 0 "
                "and are now NaN. Impute inside the fold and "
                f"consider a '{col}_missing' indicator."
            )

    for col, pct in r["missing_pct"].items():
        if pct and pct > 20:
            warnings.append(
                f"{col}: {pct}% missing. Consider dropping "
                "the column rather than imputing most of it."
            )

    if r["duplicates"]["exact_rows"]:
        warnings.append(
            f"{r['duplicates']['exact_rows']} exact duplicate rows. "
            "Inspect duplicates before splitting to avoid "
            "potential train/test leakage."
        )

    if r["duplicates"]["contradictory_rows"]:
        warnings.append(
            f"{r['duplicates']['contradictory_rows']} rows are "
            "feature-identical with disagreeing labels; estimated "
            f"accuracy ceiling is "
            f"{r['duplicates']['label_noise_ceiling_hint']}."
        )

    gender = r["subgroups"].get("gender", {})

    if gender:
        smallest = min(gender.values())
        total = sum(gender.values())

        if total and smallest / total < 0.30:
            warnings.append(
                f"gender is imbalanced ({gender}). Report "
                "sensitivity and specificity by gender."
            )

    maj = r["class_balance"]["majority_class_accuracy"]

    if maj is not None:
        warnings.append(
            "Baseline to beat: predicting the majority class "
            f"always gives {maj:.1%} accuracy."
        )

    return warnings


def audit_report(r: dict[str, Any]) -> str:
    """Render the audit as Markdown."""
    lines = [
        f"# Data audit -- `{r['source']}`",
        "",
    ]

    cb = r["class_balance"]

    lines += [
        f"- rows: **{r['n_rows']}**, "
        f"features: **{r['n_features']}**",
        f"- prevalence: **{cb['prevalence']:.3f}** "
        f"({cb['positive']} positive / {cb['negative']} negative)",
        f"- majority-class accuracy: "
        f"**{cb['majority_class_accuracy']:.3f}**",
        "",
    ]

    lines += [
        "## Missingness after cleaning",
        "",
        "| column | missing | % |",
        "|---|---:|---:|",
    ]

    for c, n in r["missing_after_cleaning"].items():
        lines.append(
            f"| `{c}` | {n} | {r['missing_pct'][c]} |"
        )

    lines += [
        "",
        "## Numeric ranges",
        "",
        "| column | n | min | median | max | mean | sd |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for c, stats in r["numeric"].items():

        def fmt(value):
            return "-" if value is None else f"{value:.2f}"

        lines.append(
            f"| `{c}` | {stats['n']} | "
            f"{fmt(stats['min'])} | "
            f"{fmt(stats['p50'])} | "
            f"{fmt(stats['max'])} | "
            f"{fmt(stats['mean'])} | "
            f"{fmt(stats['std'])} |"
        )

    lines += [
        "",
        "## Duplicates and label noise",
        "",
    ]

    d = r["duplicates"]

    lines += [
        f"- exact duplicate rows: **{d['exact_rows']}**",
        f"- feature-identical rows: "
        f"**{d['feature_identical_rows']}**",
        f"- contradictory groups / rows: "
        f"**{d['contradictory_groups']}** / "
        f"**{d['contradictory_rows']}**",
        f"- accuracy ceiling from label noise: "
        f"**{d['label_noise_ceiling_hint']}**",
    ]

    lines += [
        "",
        "## Subgroups",
        "",
    ]

    for key, values in r["subgroups"].items():
        prev = r["prevalence_by_subgroup"].get(key, {})

        parts = [
            f"{group}={count} "
            f"(prev {prev.get(group, float('nan')):.2f})"
            for group, count in values.items()
        ]

        lines.append(
            f"- **{key}**: " + ", ".join(parts)
        )

    lines += [
        "",
        "## Warnings",
        "",
    ]

    lines += [
        f"{i}. {warning}"
        for i, warning in enumerate(
            r["warnings"],
            1,
        )
    ]

    return "\n".join(lines) + "\n"


def write(
    r: dict[str, Any],
    out_dir: str | Path = "results/runs",
) -> tuple[Path, Path]:
    """Write JSON and Markdown audit reports."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / "data_audit.json"
    markdown_path = out / "data_audit.md"

    json_path.write_text(
        json.dumps(r, indent=2, default=str)
    )

    markdown_path.write_text(
        audit_report(r)
    )

    return json_path, markdown_path