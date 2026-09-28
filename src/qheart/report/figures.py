"""Figures. Matplotlib only, no seaborn, no styles that need a network fetch.

Every figure here is required to be readable in greyscale and at the size a projector
renders it. That rules out the default categorical palette -- five lines that are
distinguishable on a laptop become five grey lines on a hall screen, so line style
varies alongside colour.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = ["FIGURES", "save_all", "roc_curves", "pr_curves", "calibration_plot",
           "param_vs_metric", "noise_degradation", "subgroup_bars"]

FIGURES = ("roc_curves", "pr_curves", "calibration", "param_vs_metric",
           "noise_degradation", "subgroup_bars")

STYLES = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]


def _ax(figsize=(5.2, 4.0)):
    import matplotlib
    matplotlib.use("Agg")            # no display in CI or on a headless box
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    return fig, ax


def _save(fig, path: str | Path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return p


def roc_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], path="results/figures/roc.png"):
    """``{name: (fpr, tpr)}``. Chance diagonal always drawn -- a ROC curve without it
    invites the reader to over-read a modest bulge."""
    fig, ax = _ax()
    for i, (name, (fpr, tpr)) in enumerate(curves.items()):
        ax.plot(fpr, tpr, STYLES[i % len(STYLES)], linewidth=1.7, label=name)
    ax.plot([0, 1], [0, 1], color="0.6", linewidth=0.9, label="chance")
    ax.set_xlabel("false positive rate (1 - specificity)")
    ax.set_ylabel("true positive rate (sensitivity)")
    ax.legend(fontsize=7.5, loc="lower right")
    return _save(fig, path)


def pr_curves(curves: dict[str, tuple[np.ndarray, np.ndarray]], prevalence: float,
              path="results/figures/pr.png"):
    """Precision-recall, with the prevalence line drawn as the true baseline.

    A PR curve without the prevalence line is unreadable: at 55% prevalence, precision
    0.6 is barely above chance, and nothing on the axes says so.
    """
    fig, ax = _ax()
    for i, (name, (rec, prec)) in enumerate(curves.items()):
        ax.plot(rec, prec, STYLES[i % len(STYLES)], linewidth=1.7, label=name)
    ax.axhline(prevalence, color="0.6", linewidth=0.9,
               label=f"prevalence = {prevalence:.3f}")
    ax.set_xlabel("recall (sensitivity)")
    ax.set_ylabel("precision (PPV)")
    ax.legend(fontsize=7.5, loc="lower left")
    return _save(fig, path)


def calibration_plot(bins: dict[str, tuple[np.ndarray, np.ndarray]],
                     path="results/figures/calibration.png"):
    """Predicted vs observed frequency. A model can rank well and still be badly
    calibrated, which matters the moment a probability is shown to a clinician."""
    fig, ax = _ax()
    for i, (name, (pred, obs)) in enumerate(bins.items()):
        ax.plot(pred, obs, STYLES[i % len(STYLES)], marker="o", markersize=3.2,
                linewidth=1.5, label=name)
    ax.plot([0, 1], [0, 1], color="0.6", linewidth=0.9, label="perfect")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.legend(fontsize=7.5, loc="upper left")
    return _save(fig, path)


def param_vs_metric(rows: list[dict], metric: str = "pr_auc",
                    path="results/figures/params_vs_metric.png"):
    """THE figure. Log-x parameter count against performance, Control-C annotated.

    Log scale because the range runs from ~96 to millions and a linear axis collapses
    everything interesting into the left-hand pixel column.
    """
    fig, ax = _ax(figsize=(5.6, 4.0))
    for r in rows:
        q = bool(r.get("quantum"))
        ctrl = "control" in str(r.get("model", ""))
        ax.scatter(r["params"], r[metric], s=64 if ctrl else 44,
                   marker="D" if ctrl else ("o" if q else "s"),
                   facecolor="none" if not q else None, edgecolor="black", linewidth=1.1,
                   zorder=3)
        ax.annotate(r["model"], (r["params"], r[metric]), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel(metric)
    ax.set_title("Performance against model size", fontsize=9)
    return _save(fig, path)


def noise_degradation(rows: list[dict], metric_label: str = "PR-AUC",
                      path="results/figures/noise.png"):
    """Metric against shot count with error bars -- the "would it survive hardware"
    plot. Log-x, because the interesting regime is the low-shot end."""
    fig, ax = _ax()
    shots = [r["shots"] for r in rows]
    mu = [r["mean"] for r in rows]
    sd = [r.get("sd", 0.0) for r in rows]
    ax.errorbar(shots, mu, yerr=sd, marker="o", markersize=3.5, linewidth=1.6,
                capsize=2.5, color="black")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("shots per circuit evaluation")
    ax.set_ylabel(metric_label)
    ax.set_title("Degradation under finite sampling", fontsize=9)
    return _save(fig, path)


def subgroup_bars(table, metric: str = "sensitivity",
                  path="results/figures/subgroups.png"):
    """Per-subgroup metric with unreportable groups hatched.

    Hatched rather than hidden: a group with 24 patients still belongs on the plot,
    marked as too small to conclude from. Dropping it silently is how a disparity
    disappears from a report.
    """
    fig, ax = _ax(figsize=(5.6, 3.6))
    rows = list(table)
    labels = [f"{r['key']}={r['group']}" for r in rows]
    vals = [r.get(metric, np.nan) for r in rows]
    ok = [bool(r.get("reportable", True)) for r in rows]
    ax.bar(range(len(rows)), vals, color="0.85", edgecolor="black", linewidth=0.9,
           hatch=["" if o else "///" for o in ok])
    for i, r in enumerate(rows):
        ax.text(i, 0.02, f"n={r.get('n', '?')}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.set_title(f"{metric} by subgroup (hatched = n < 30, not reportable)", fontsize=9)
    return _save(fig, path)


def save_all(payload: dict, outdir: str | Path = "results/figures") -> list[Path]:
    """Render whatever is present in ``payload``; skip the rest without complaint."""
    out: list[Path] = []
    d = Path(outdir)
    if "roc" in payload:
        out.append(roc_curves(payload["roc"], d / "roc.png"))
    if "pr" in payload:
        out.append(pr_curves(payload["pr"], payload.get("prevalence", 0.5), d / "pr.png"))
    if "calibration" in payload:
        out.append(calibration_plot(payload["calibration"], d / "calibration.png"))
    if "params" in payload:
        out.append(param_vs_metric(payload["params"], payload.get("metric", "pr_auc"),
                                   d / "params_vs_metric.png"))
    if "noise" in payload:
        out.append(noise_degradation(payload["noise"], path=d / "noise.png"))
    if "subgroups" in payload:
        out.append(subgroup_bars(payload["subgroups"], path=d / "subgroups.png"))
    return out
