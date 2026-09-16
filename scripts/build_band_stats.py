"""Validate the risk bands at 7, 14 and 28 days and write the table the interface reads.

Section 3.8 sets the band boundaries at two, four and eight times the base rate of the
52 completed weeks before the forecast, and reports the realised event rate each band
carries. Those realised rates were measured at seven days only, because that is the
horizon the evaluation settles on.

The interface offers three windows, so it needs the same validation at each. Showing a
band at fourteen days whose realised rate had only ever been measured at seven would be
presenting an unmeasured number as a measured one. This script measures all three, from
the per-horizon test scores `evaluate_protocol.py` saves.

The base rate differs by horizon, roughly 0.036 at seven days against 0.113 at
twenty-eight, so each window is anchored to its own. That is what keeps a band meaning
the same thing at every window.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ("Severe", "High", "Elevated", "Low")
MULTIPLES = (8, 4, 2)
TEST_WEEKS = 52
FIRST_TEST_WEEK = 395


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--scores", default="data/processed/horizon_test_scores.parquet")
    parser.add_argument("--out", default="data/processed/band_stats.csv")
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel, columns=["week", "occurred"])
    weeks = np.sort(panel["week"].unique())
    scores = pd.read_parquet(args.scores)
    print(f"{len(scores):,} scored rows across "
          f"{scores['horizon_days'].nunique()} horizons")

    rows = []
    for days, block in scores.groupby("horizon_days"):
        horizon_weeks = days // 7
        # The anchor a deployed system would have: the labelled rate over the 52 weeks
        # before each test block, computed at this horizon rather than borrowed from
        # another one.
        anchors = {}
        for fold in sorted(block["fold"].unique()):
            start = FIRST_TEST_WEEK + TEST_WEEKS * (int(fold) - 1)
            window = weeks[start - TEST_WEEKS:start]
            prior = block[block["week"].isin(window)]
            if len(prior):
                anchors[fold] = float(prior["y"].mean())
            else:
                # Fold one's anchor window sits before the scored period, so it is taken
                # from the panel's own labels at this horizon instead.
                lab = panel[panel["week"].isin(window)]
                anchors[fold] = float(lab["occurred"].mean()) * horizon_weeks
        anchor = block["fold"].map(anchors).to_numpy()
        p = block["score"].to_numpy()
        band = np.select([p >= MULTIPLES[0] * anchor, p >= MULTIPLES[1] * anchor,
                          p >= MULTIPLES[2] * anchor], ["Severe", "High", "Elevated"], "Low")
        y = block["y"].to_numpy()
        n_weeks = block["week"].nunique()
        base = y.mean()
        print(f"\n{days} days: base rate {base:.4f}, anchor {np.mean(anchor):.4f} "
              f"over {n_weeks} test weeks")
        for name in BANDS:
            mask = band == name
            realised = float(y[mask].mean()) if mask.any() else float("nan")
            rows.append({
                "horizon_days": int(days), "band": name,
                # The horizon comparison holds the model fixed at the calibrated linear
                # member, for the reason section 3.7 gives: a model that changed between
                # windows would confound the window with the algorithm. So these realised
                # rates describe that model, not the stack the interface displays, and
                # the interface has to say so rather than let a reader assume otherwise.
                "validated_on": "calibrated linear model, all three windows",
                "areas_per_week": mask.sum() / n_weeks,
                "realised_rate": realised,
                "times_base": realised / base if mask.any() else float("nan"),
                "share_of_events": float(y[mask].sum() / y.sum()),
                "anchor_rate": float(np.mean(anchor)),
            })
            print(f"  {name:<9} {mask.sum() / n_weeks:>7.1f} areas/week  "
                  f"realised {realised:.3f}  "
                  f"{'(' + str(round(realised / base, 1)) + 'x base)' if mask.any() else '(empty)':>12}  "
                  f"{100 * y[mask].sum() / y.sum():>5.1f}% of events")

    table = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    print(f"\nwritten to {args.out}")

    low = table[table.band == "Low"]
    print("\nshare of events falling in the Low band, by horizon:")
    for _, r in low.iterrows():
        print(f"  {int(r.horizon_days):>2} days: {100 * r.share_of_events:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
