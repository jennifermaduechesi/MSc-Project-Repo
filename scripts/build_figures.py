"""Regenerate the two diagrams in Chapter Three.

Kept in the repository because the originals were drawn in a scratch directory and could
not be reproduced after it was recycled, which is the same failure that cost the chapter
its Word builder once already. A figure nobody can regenerate is a figure nobody can check.

The block counts and window lengths are read from the run configuration rather than typed
in, so a diagram cannot drift away from the design it claims to show.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

OUT = Path("dissertation")
PROC = Path("data/processed")
DPI = 200
INK, LIGHT, MID, DARK = "#1A1A1A", "#E8E8E8", "#B4B4B4", "#3A3A3A"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def figure_two() -> None:
    """Feature windows against the forecast week."""
    fig, ax = plt.subplots(figsize=(10.8, 5.25))
    ax.set_xlim(0, 10.6); ax.set_ylim(0, 5.0); ax.axis("off")

    labels = [f"t-{i}" for i in range(8, 0, -1)] + ["t", "t+1"]
    x0, w, gap, ybox, hbox = 0.6, 0.62, 0.075, 1.35, 1.55
    centres = {}
    for i, lab in enumerate(labels):
        x = x0 + i * (w + gap)
        centres[lab] = x + w / 2
        fill = DARK if lab == "t" else ("white" if lab == "t+1" else LIGHT)
        ax.add_patch(Rectangle((x, ybox), w, hbox, facecolor=fill, edgecolor=MID, lw=1.4))
        ax.text(x + w / 2, ybox + hbox / 2, lab, ha="center", va="center",
                fontsize=13, color="white" if lab == "t" else INK,
                fontweight="bold" if lab == "t" else "normal")

    # The forecast is issued on the Monday of week t, so the boundary sits on its left edge.
    cut = x0 + 8 * (w + gap) - gap / 2
    ax.plot([cut, cut], [0.55, 4.35], ls=(0, (5, 4)), color=INK, lw=2.0)
    ax.text(cut + 0.12, 4.42, "forecast issued, Monday of week t",
            fontsize=13, fontweight="bold", va="bottom")

    # Both arrows stop at t-1's right edge. Labels are right-aligned to finish just short
    # of that rule: centring them on their own span pushes the text across it, because the
    # text is wider than the span it describes.
    for span_from, y, text in ((labels[0], 3.72, "features: longer windows, all ending at t-1"),
                               ("t-4",     2.98, "features: last 4 weeks, ending at t-1")):
        left = centres[span_from] - w / 2
        ax.annotate("", xy=(cut, y), xytext=(left, y),
                    arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
        ax.text(cut - 0.12, y + 0.16, text, ha="right", va="bottom", fontsize=13)

    ax.annotate("", xy=(centres["t"] + w / 2, 0.82), xytext=(cut, 0.82),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax.text(centres["t"] + w / 2 + 0.18, 0.82, "label: did an event occur in week t",
            fontsize=13, fontweight="bold", va="center")

    fig.savefig(OUT / "figure2_leakage_window.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def figure_three() -> None:
    """How the stack is fitted inside one rolling-origin fold."""
    cfg = json.load(open(PROC / "ensemble_results.json"))["config"]
    inner, test = cfg["inner_weeks"], cfg["test_weeks"]
    n_base = len(cfg["base"])
    words = {2: "two", 3: "three", 4: "four", 5: "five"}[n_base]

    fig, ax = plt.subplots(figsize=(10.8, 5.85))
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 5.6); ax.axis("off")

    xa, xb, xc, xd = 0.45, 4.15, 5.85, 7.55
    ax.add_patch(Rectangle((xa, 4.10), xb - xa, 0.95, facecolor=LIGHT, edgecolor=MID, lw=1.4))
    ax.add_patch(Rectangle((xb, 4.10), xc - xb, 0.95, facecolor=MID, edgecolor=MID, lw=1.4))
    ax.add_patch(Rectangle((xc, 4.10), xd - xc, 0.95, facecolor=DARK, edgecolor=MID, lw=1.4))
    ax.text((xa + xb) / 2, 4.575, "earlier weeks", ha="center", va="center", fontsize=13)
    ax.text((xb + xc) / 2, 4.575, f"inner block\n{inner} weeks", ha="center", va="center", fontsize=13)
    ax.text((xc + xd) / 2, 4.575, f"test block\n{test} weeks", ha="center", va="center",
            fontsize=13, color="white")
    ax.text(xd + 0.15, 4.575, "time →", fontsize=13, style="italic", va="center")

    ax.annotate("", xy=(xc, 5.40), xytext=(xa, 5.40),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    ax.text((xa + xc) / 2, 5.52, "training period for this fold", ha="center", va="bottom", fontsize=13)

    for y, fit_to, pred_to, fitted, note in (
            (2.55, xb, xc, f"{words} base learners fitted", "meta-learner fitted here"),
            (0.85, xc, xd, "the same four refitted",        "meta-learner applied here")):
        ax.add_patch(Rectangle((xa, y), fit_to - xa, 0.80, facecolor="white",
                               edgecolor=INK, lw=1.6, hatch="//"))
        ax.add_patch(Rectangle((fit_to, y), pred_to - fit_to, 0.80, facecolor="white",
                               edgecolor=INK, lw=1.6, ls=(0, (5, 4))))
        # White backing, or the hatching reads straight through the label.
        backing = dict(facecolor="white", edgecolor="none", pad=2.5)
        ax.text((xa + fit_to) / 2, y + 0.40, fitted, ha="center", va="center",
                fontsize=13, bbox=backing)
        ax.text((fit_to + pred_to) / 2, y + 0.40, "predict", ha="center", va="center",
                fontsize=13, bbox=backing)
        ax.text(pred_to + 0.15, y + 0.40, note, fontsize=13, va="center")

    for x in (xb, xc):
        ax.plot([x, x], [0.62, 5.05], ls=(0, (5, 4)), color=INK, lw=1.8, zorder=0)
    ax.annotate("", xy=(4.55, 2.45), xytext=(4.55, 1.80),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.8))
    ax.text(4.70, 2.12, "learned weights carried forward", fontsize=13, va="center",
            bbox=dict(facecolor="white", edgecolor="none", pad=2.5))

    ax.text(xa, 0.18, "Hatched: rows the model was fitted on.     "
                      "Dashed: rows it predicted but never saw the answers for.",
            fontsize=12.5, va="center", color="#333333")

    fig.savefig(OUT / "figure3_stacking_design.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def figure_four() -> None:
    """Average precision per fold, every model. Chapter Four, the model comparison."""
    import json
    res = json.load(open(PROC / "ensemble_results.json"))["results"]
    order = [("long_run", "long-run rate baseline"), ("recency", "recency baseline"),
             ("gradient_boosting", "gradient boosting"), ("logistic", "logistic, calibrated"),
             ("stgnn", "graph network"), ("random_forest", "random forest"),
             ("ensemble_unweighted", "stack, unweighted meta")]
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    marks = ["o", "s", "^", "D", "v", "P", "X"]
    greys = ["#B0B0B0", "#9A9A9A", "#848484", "#6E6E6E", "#585858", "#3C3C3C", INK]
    folds = [1, 2, 3, 4, 5]
    for (key, label), mk, col in zip(order, marks, greys):
        ap = [f["average_precision"] for f in res[key]]
        ax.plot(folds, ap, marker=mk, color=col, lw=2.0 if key == "ensemble_unweighted" else 1.2,
                ms=8 if key == "ensemble_unweighted" else 6, label=label,
                zorder=3 if key == "ensemble_unweighted" else 2)
    ax.set_xticks(folds)
    ax.set_xlabel("rolling-origin fold", fontsize=12)
    ax.set_ylabel("average precision", fontsize=12)
    ax.set_xlim(0.8, 5.2)
    ax.grid(axis="y", color="#E4E4E4", lw=0.9)
    ax.set_axisbelow(True)
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, ncol=2, loc="upper left")
    fig.savefig(OUT / "figure4_fold_performance.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def figure_five() -> None:
    """The horizon trade-off: ranking quality up, operational recall down."""
    import json
    import numpy as np
    prot = json.load(open(PROC / "protocol_results.json"))["horizon"]
    keys = ["7d", "14d", "28d"]
    ap = [np.mean([f["average_precision"] for f in prot[k]]) for k in keys]
    r20 = [np.mean([f["recall_at_20"] for f in prot[k]]) for k in keys]
    lift = [np.mean([f["lift_at_20"] for f in prot[k]]) for k in keys]
    x = range(3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.4))
    a1.plot(x, ap, marker="o", color=INK, lw=2.0, ms=8)
    a1.set_ylabel("average precision", fontsize=12)
    a1.set_title("ranking quality rises with the window", fontsize=12.5, pad=10)
    # Recall alone, on its own axis. An earlier draft overlaid lift divided by forty on
    # this axis, which put two different quantities on one scale and read as though the
    # two were comparable. They are not.
    a2.plot(x, r20, marker="s", color=INK, lw=2.0, ms=8)
    for xi, (r, l) in enumerate(zip(r20, lift)):
        a2.annotate(f"lift {l:.1f}x", (xi, r), textcoords="offset points",
                    xytext=(0, -20), ha="center", fontsize=10.5, color="#5A5A5A")
    a2.set_ylabel("recall at top 20", fontsize=12)
    a2.set_title("what a patrol list catches falls", fontsize=12.5, pad=10)
    a2.set_ylim(min(r20) * 0.86, max(r20) * 1.05)
    for a in (a1, a2):
        a.set_xticks(list(x)); a.set_xticklabels(["7 days", "14 days", "28 days"], fontsize=11)
        a.grid(axis="y", color="#E4E4E4", lw=0.9); a.set_axisbelow(True)
        for spine in ("top", "right"): a.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "figure5_horizon_tradeoff.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def figure_six() -> None:
    """Predicted against observed, by decile of predicted probability."""
    import pandas as pd
    s = pd.read_parquet(PROC / "ensemble_test_scores.parquet")
    s["dec"] = pd.qcut(s["ensemble_unweighted"], 10, labels=False, duplicates="drop")
    g = s.groupby("dec").agg(predicted=("ensemble_unweighted", "mean"),
                             observed=("occurred", "mean"))
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    top = float(max(g["predicted"].max(), g["observed"].max())) * 1.08
    ax.plot([0, top], [0, top], ls=(0, (5, 4)), color="#9A9A9A", lw=1.4,
            label="perfect calibration")
    ax.plot(g["predicted"], g["observed"], marker="o", color=INK, lw=1.8, ms=7,
            label="the stack, unweighted meta")
    ax.set_xlabel("mean predicted probability", fontsize=12)
    ax.set_ylabel("observed event rate", fontsize=12)
    ax.set_xlim(0, top); ax.set_ylim(0, top)
    ax.grid(color="#E8E8E8", lw=0.9); ax.set_axisbelow(True)
    for spine in ("top", "right"): ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    fig.savefig(OUT / "figure6_calibration.png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    figure_two()
    figure_three()
    figure_four()
    figure_five()
    figure_six()
    for f in ("figure2_leakage_window.png", "figure3_stacking_design.png",
              "figure4_fold_performance.png", "figure5_horizon_tradeoff.png",
              "figure6_calibration.png"):
        print(f"wrote {OUT / f} ({(OUT / f).stat().st_size:,} bytes)")
