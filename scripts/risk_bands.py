"""Derive and check the risk bands the interface displays, as specified in section 3.8.

The band boundaries decide how many areas a commander is sent to, so they are a
research decision rather than a presentation one and are set from the data rather than
chosen for visual balance. Boundaries are multiples of the base rate, at two, four and
eight times, so that each band states its own meaning: an Elevated area is at least
twice as likely as a typical area-week to record an event.

Which base rate the multiples apply to is the part that needed testing, and three
candidates are compared here rather than one assumed.

    evaluation   The base rate over the test years themselves. Well behaved, and
                 retrospective: a system issuing a forecast cannot know the rate of the
                 year it is forecasting.
    training     The base rate over the whole training period, which a deployed system
                 does know. It fails. The recorded rate rises across the window, partly
                 through the coverage discontinuity of section 3.4, so an average over
                 all history understates the present and every multiple becomes too easy
                 to clear.
    recent       The base rate over the 52 completed weeks before the forecast. Known at
                 forecast time, tracks the current level, and is the rule adopted.

Scores are read from `ensemble_test_scores.parquet`, written by `train_ensemble.py`,
and the column used is the stack under its unweighted meta-learner, which section 3.6
establishes as the form whose numbers can be read as probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ("Severe", "High", "Elevated", "Low")
MULTIPLES = (8, 4, 2)          # Severe, High, Elevated; Low is everything below
TEST_WEEKS = 52
FIRST_TEST_WEEK = 395          # fold one begins here, matching train_ensemble.py
SCORE = "ensemble_unweighted"


def assign(scores: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Band each row against its own anchor rate."""
    return np.select([scores >= MULTIPLES[0] * anchor,
                      scores >= MULTIPLES[1] * anchor,
                      scores >= MULTIPLES[2] * anchor],
                     ["Severe", "High", "Elevated"], "Low")


def anchors(panel: pd.DataFrame, scores: pd.DataFrame, kind: str) -> np.ndarray:
    weeks = np.sort(panel["week"].unique())
    starts = {f: FIRST_TEST_WEEK + TEST_WEEKS * (f - 1) for f in sorted(scores["fold"].unique())}
    if kind == "evaluation":
        return np.full(len(scores), scores["occurred"].mean())
    per_fold = {}
    for fold, start in starts.items():
        window = weeks[:start] if kind == "training" else weeks[start - TEST_WEEKS:start]
        per_fold[fold] = panel.loc[panel["week"].isin(window), "occurred"].mean()
    return scores["fold"].map(per_fold).to_numpy()


def profile(scores: pd.DataFrame, band: np.ndarray) -> pd.DataFrame:
    y = scores["occurred"].to_numpy()
    n_weeks = scores["week"].nunique()
    base = y.mean()
    rows = []
    for name in BANDS:
        mask = band == name
        rows.append({
            "band": name,
            "areas_per_week": mask.sum() / n_weeks,
            "realised_rate": y[mask].mean() if mask.any() else float("nan"),
            "times_base": (y[mask].mean() / base) if mask.any() else float("nan"),
            "share_of_events": y[mask].sum() / y.sum(),
            "areas_ever": scores.loc[mask, "pcode"].nunique(),
        })
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--scores", default="data/processed/ensemble_test_scores.parquet")
    parser.add_argument("--out", default="data/processed/risk_bands.json")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel, columns=["week", "occurred"])
    scores = pd.read_parquet(args.scores).sort_values(["pcode", "week"]).reset_index(drop=True)
    y = scores["occurred"].to_numpy()
    print(f"{len(scores):,} scored area-weeks, {scores['week'].nunique()} test weeks, "
          f"{scores['pcode'].nunique()} areas, base rate {y.mean():.4f}")

    results = {}
    for kind in ("evaluation", "training", "recent"):
        anchor = anchors(panel, scores, kind)
        band = assign(scores[SCORE].to_numpy(), anchor)
        table = profile(scores, band)
        results[kind] = table.to_dict("records")
        print(f"\nanchor: {kind}   cuts at {MULTIPLES[2]}x, {MULTIPLES[1]}x, {MULTIPLES[0]}x "
              f"of a rate running {anchor.min():.4f} to {anchor.max():.4f}")
        print(f"  {'band':<10}{'areas/wk':>10}{'realised':>10}{'x base':>9}{'of events':>11}")
        for r in results[kind]:
            print(f"  {r['band']:<10}{r['areas_per_week']:>10.1f}{r['realised_rate']:>10.3f}"
                  f"{r['times_base']:>9.1f}{r['share_of_events']:>10.1%}")
        # Per-fold band sizes, because a pooled average hides the failure mode that
        # rules out the training anchor: it is a single fold that becomes unusable.
        sizes = (pd.crosstab(scores["fold"], band) / TEST_WEEKS).reindex(columns=list(BANDS))
        print("  areas per week by fold: " + "; ".join(
            f"fold {f} " + ", ".join(f"{b} {row[b]:.1f}" for b in BANDS if row[b] == row[b])
            for f, row in sizes.iterrows()))
        results[f"{kind}_by_fold"] = sizes.to_dict("index")

    # everything below describes the adopted rule
    anchor = anchors(panel, scores, "recent")
    scores["band"] = assign(scores[SCORE].to_numpy(), anchor)

    print("\nadopted rule: share of all events banded Low, by fold")
    by_fold = {}
    for fold, block in scores.groupby("fold"):
        share = block.loc[block["band"] == "Low", "occurred"].sum() / block["occurred"].sum()
        by_fold[int(fold)] = float(share)
        empty = [b for b in BANDS if not (block["band"] == b).any()]
        print(f"  fold {fold}: {share:>5.1%}" + (f"   (no area reached {', '.join(empty)})" if empty else ""))
    pooled_low = y[scores["band"].to_numpy() == "Low"].sum() / y.sum()
    print(f"  pooled : {pooled_low:>5.1%}")

    previous = scores.groupby("pcode")["band"].shift()
    known = previous.notna()
    unchanged = (scores["band"][known] == previous[known]).mean()
    moved = scores["band"][known] != previous[known]
    events_in_moves = scores["occurred"][known][moved].sum() / scores["occurred"][known].sum()
    print(f"\nband unchanged from the previous week: {unchanged:.1%}")
    print(f"the {moved.mean():.1%} of area-weeks that move carry {events_in_moves:.1%} of events")
    transitions = pd.crosstab(previous[known], scores["band"][known], normalize="index")
    transitions = transitions.reindex(index=list(BANDS)[::-1], columns=list(BANDS)[::-1])
    print("\ntransition matrix, rows last week, columns this week")
    print(transitions.round(3).to_string())

    reach = {b: int(scores.loc[scores["band"] == b, "pcode"].nunique()) for b in BANDS}
    print(f"\nareas reaching each band at least once, of {scores['pcode'].nunique()}: {reach}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "anchors": results,
        "adopted": "recent",
        "low_share_by_fold": by_fold,
        "low_share_pooled": float(pooled_low),
        "unchanged_week_to_week": float(unchanged),
        "events_in_moving_rows": float(events_in_moves),
        "transitions": transitions.to_dict(),
        "areas_reaching_band": reach,
    }, indent=1))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
