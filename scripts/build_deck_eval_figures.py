"""ROC curves and a confusion matrix for the defence deck, from the held-out predictions.

Both are computed from data/processed/ensemble_test_scores.parquet, which holds every
prediction the models made on weeks they never trained on, with the true label beside it.
Nothing here is illustrative.

The confusion matrix is taken at the operating point the system is actually used at, which
is the twenty highest-ranked areas in each week, rather than at a probability of 0.5. At
0.5 a model facing a 3.6 per cent base rate predicts almost nothing and the matrix says
only that the outcome is rare.
"""
from __future__ import annotations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

SCORES = Path("data/processed/ensemble_test_scores.parquet")
OUT = Path("dissertation")
GREEN, RED, GREY, CARD = "#1B4332", "#E4002B", "#565E6C", "#F5F7FA"
TOP_K = 20

MODELS = [("ensemble_unweighted", "Stacking ensemble", GREEN, 2.6),
          ("random_forest",       "Random forest",     "#2A5F45", 1.5),
          ("stgnn",               "Graph network",     "#6B9080", 1.5),
          ("logistic",            "Penalised logistic", "#9DB4A6", 1.5),
          ("recency",             "Recency baseline",  RED, 1.8)]


def roc_plot(d: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    for col, label, colour, lw in MODELS:
        fpr, tpr, _ = roc_curve(d["occurred"], d[col])
        auc = roc_auc_score(d["occurred"], d[col])
        ax.plot(fpr, tpr, color=colour, lw=lw, label=f"{label}  ({auc:.3f})")
    ax.plot([0, 1], [0, 1], ls=(0, (5, 4)), color="#B4B4B4", lw=1.2, label="Chance  (0.500)")
    ax.set_xlabel("False positive rate", fontsize=12, color=GREY)
    ax.set_ylabel("True positive rate", fontsize=12, color=GREY)
    ax.tick_params(labelsize=11, colors=GREY)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#D7DCE3")
    leg = ax.legend(loc="lower right", fontsize=11, frameon=False, title="Area under the curve")
    leg.get_title().set_fontsize(11)
    leg.get_title().set_color(GREY)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.savefig(OUT / "deck_roc_curve.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def confusion_plot(d: pd.DataFrame) -> dict:
    """Predict positive for the TOP_K highest-ranked areas in each week."""
    d = d.copy()
    d["rank"] = d.groupby("week")["ensemble_unweighted"].rank(ascending=False, method="first")
    d["flagged"] = d["rank"] <= TOP_K
    tp = int(((d.occurred == 1) & d.flagged).sum())
    fp = int(((d.occurred == 0) & d.flagged).sum())
    fn = int(((d.occurred == 1) & ~d.flagged).sum())
    tn = int(((d.occurred == 0) & ~d.flagged).sum())

    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    # Rows are what the system said, columns what happened, so the off-diagonal cells are
    # false positive (flagged, no event) and false negative (not flagged, event).
    cells = np.array([[tp, fp], [fn, tn]])
    labels = [["True positive", "False positive"], ["False negative", "True negative"]]
    fills = [[GREEN, CARD], ["#EED5CE", CARD]]
    texts = [["white", GREY], [GREY, GREY]]
    for r in range(2):
        for c in range(2):
            ax.add_patch(plt.Rectangle((c, 1 - r), 1, 1, facecolor=fills[r][c],
                                       edgecolor="white", lw=3))
            ax.text(c + 0.5, 1.68 - r, f"{cells[r][c]:,}", ha="center", va="center",
                    fontsize=27, fontweight="bold", color=texts[r][c])
            ax.text(c + 0.5, 1.38 - r, labels[r][c], ha="center", va="center",
                    fontsize=12, color=texts[r][c])
    ax.set_xticks([0.5, 1.5]); ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["Event occurred", "No event"], fontsize=12, color=GREY)
    ax.set_yticklabels(["Not in top 20", "In top 20"], fontsize=12, color=GREY)
    ax.set_xlabel("What actually happened", fontsize=12.5, color=GREEN, labelpad=10)
    ax.set_ylabel("What the system said", fontsize=12.5, color=GREEN, labelpad=10)
    ax.set_xlim(0, 2); ax.set_ylim(0, 2)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(OUT / "deck_confusion_matrix.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def main() -> None:
    d = pd.read_parquet(SCORES)
    roc_plot(d)
    cm = confusion_plot(d)
    weeks = d["week"].nunique()
    recall = cm["tp"] / (cm["tp"] + cm["fn"])
    precision = cm["tp"] / (cm["tp"] + cm["fp"])
    print(f"rows {len(d):,}  weeks {weeks}  positives {int(d.occurred.sum()):,}")
    print(f"top-{TOP_K} per week: TP {cm['tp']:,}  FP {cm['fp']:,}  "
          f"FN {cm['fn']:,}  TN {cm['tn']:,}")
    print(f"recall {recall:.4f}   precision {precision:.4f}")
    print(f"flagged per week = {TOP_K}, total flagged {cm['tp'] + cm['fp']:,} "
          f"(= {weeks} weeks x {TOP_K})")


if __name__ == "__main__":
    main()
