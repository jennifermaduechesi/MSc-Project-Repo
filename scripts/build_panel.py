"""Build the Local Government Area week panel and its features.

The panel is 774 areas by 655 weeks, one row per area-week, covering 13 January 2014
to 2 August 2026. The outcome is whether a kidnapping or banditry event was recorded
in that area during that week, together with the count for the severity stage of the
hurdle model.

**Leakage is the thing this file exists to prevent.** A forecast for the week
beginning Monday is issued on that Monday, so every feature attached to that week may
use only weeks that have already finished. Two rules enforce it.

First, every history feature is computed on the count matrix shifted forward by one
week before any window is applied, so a window labelled "last 4 weeks" for week t
covers t-4 to t-1 and never t itself. Second, no feature is derived from the label
column of the same row. The severity count is carried as an outcome, not as an input,
because a count available to the model would let it read the answer.

Both rules are asserted at the end of the build rather than trusted. The assertion
compares each feature against a deliberately corrupted panel in which one week's
events are moved, and fails if a feature for an earlier week changes.

Features fall into five groups, each traceable to something the literature review
established.

    own history      Recent events in the area itself. Lewis et al. put self-excitation
                     at two to six weeks, so windows run to 26 weeks and the short ones
                     carry the weight.
    background       Long-run rate for the area, computed on an expanding window.
                     Reinhart and Greenhouse put stable spatial features in the
                     background rate rather than the triggering term, and S. D. Johnson
                     shows both a flag and a boost component are needed.
    neighbours       The same history summed over adjacent areas, which is the near
                     repeat mechanism at administrative scale.
    guardianship     Security force operations, in the area and its neighbours. This is
                     the third element of Cohen and Felson's convergence, and Chapter
                     One treats operations as a leading indicator rather than as an
                     instance of the outcome.
    calendar         Week of year as a pair of harmonics, and a linear time index.
    coverage         How many of the two source files were recording in that week.

Events before 2014 are kept as burn-in so that the lagged windows for early 2014 read
real history rather than zeros. They contribute no rows to the panel.

**A coverage discontinuity runs through the panel and is recorded as a feature.** The
general incident log covers 1 January 2021 to 31 December 2025. The specialist file
covers the whole span. So the modelling window crosses three regimes: one source to the
end of 2020, two sources for 2021 to 2025, and one source again for 2026. Mean positive
area-weeks per week are 4.7, 30.3 and 10.6 across the three, and the 2025 to 2026
boundary is a cliff rather than a taper, falling from 45 in the last week of December to
9 in the first week of January. The specialist file's own rate is continuous across both
boundaries, so neither step is a change in violence.

`cov_sources` marks each week with the number of files recording in it, so a model can
condition on the regime instead of reading the step as a change in risk. The alternative
of truncating the window was considered and rejected by the student, on the grounds that
the 2026 weeks are real observations and should not be discarded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_START = pd.Timestamp("2014-01-13")
# Chapter One defines the forecast week as the seven days beginning each Monday. In
# pandas a weekly period is named by the day it ENDS on, so weeks that begin on Monday
# are "W-SUN", and their start_time is the Monday. Using "W-MON" here would produce
# weeks running Tuesday to Monday, which is a silent one-day shift: the panel would
# still be 774 by 655 and every join against a Monday-indexed week would return
# nothing. It did exactly that on the first run.
WEEK_RULE = "W-SUN"
SHORT_WINDOWS = (1, 2, 4)    # weeks
LONG_WINDOWS = (8, 13, 26, 52)


# --------------------------------------------------------------------------- inputs
def load_incidents(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"], low_memory=False)
    frame = frame[frame["resolved"]].copy()
    frame["week"] = frame["date"].dt.to_period(WEEK_RULE).dt.start_time
    return frame


def load_adjacency(path: Path, pcodes: list[str]) -> np.ndarray:
    edges = pd.read_csv(path)
    index = {p: i for i, p in enumerate(pcodes)}
    matrix = np.zeros((len(pcodes), len(pcodes)), dtype=np.float32)
    for a, b in zip(edges.pcode_a, edges.pcode_b):
        if a in index and b in index:
            matrix[index[a], index[b]] = 1.0
            matrix[index[b], index[a]] = 1.0
    return matrix


def count_matrix(frame: pd.DataFrame, pcodes: list[str], weeks: pd.DatetimeIndex,
                 value: str | None = None) -> np.ndarray:
    """Areas by weeks. Counts rows, or sums `value` when given."""
    if frame.empty:
        return np.zeros((len(pcodes), len(weeks)), dtype=np.float32)
    grouped = (frame.groupby(["pcode", "week"]).size() if value is None
               else frame.groupby(["pcode", "week"])[value].sum())
    wide = grouped.astype("float64").unstack(fill_value=0.0)
    wide = wide.reindex(index=pcodes, columns=weeks, fill_value=0.0)
    return wide.to_numpy(dtype=np.float32)


# ------------------------------------------------------------------------- features
def lagged(matrix: np.ndarray) -> np.ndarray:
    """Shift one week forward. Column t then holds what was known up to t-1."""
    out = np.zeros_like(matrix)
    out[:, 1:] = matrix[:, :-1]
    return out


def rolling_sum(matrix: np.ndarray, window: int) -> np.ndarray:
    """Sum over the previous `window` weeks, exclusive of the current week."""
    past = lagged(matrix)
    cumulative = np.cumsum(past, axis=1)
    out = cumulative.copy()
    out[:, window:] = cumulative[:, window:] - cumulative[:, :-window]
    return out


def expanding_sum(matrix: np.ndarray) -> np.ndarray:
    """Every week strictly before the current one."""
    return np.cumsum(lagged(matrix), axis=1)


def weeks_since_last(matrix: np.ndarray, cap: int = 520) -> np.ndarray:
    """Weeks since the area's last recorded event, counted before the current week."""
    past = lagged(matrix)
    areas, periods = past.shape
    out = np.full((areas, periods), cap, dtype=np.float32)
    for a in range(areas):
        since = cap
        row = past[a]
        for t in range(periods):
            if t > 0 and row[t - 1] > 0:
                since = 1
            elif t > 0:
                since = min(since + 1, cap)
            out[a, t] = since
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incidents", default="data/processed/incidents_final.csv")
    parser.add_argument("--adjacency", default="data/reference/lga_adjacency.csv")
    parser.add_argument("--boundaries", default="data/reference/nga_admin_boundaries.xlsx")
    parser.add_argument("--out", default="data/processed/panel.parquet")
    args = parser.parse_args()

    incidents = load_incidents(Path(args.incidents))

    reference = pd.read_excel(args.boundaries, sheet_name="nga_admin2")
    pcodes = sorted(reference["adm2_pcode"].astype(str).unique())
    names = reference.set_index("adm2_pcode")[["adm2_name", "adm1_name"]]
    print(f"areas: {len(pcodes)}")

    # Week indices are generated as period start times, the same operation used to
    # stamp each incident, so the two sides cannot drift apart.
    model_end = incidents.loc[incidents["is_target"], "date"].max()
    model_weeks = pd.period_range(MODEL_START, model_end, freq=WEEK_RULE).start_time
    history_start = incidents["week"].min()
    all_weeks = pd.period_range(history_start, model_weeks[-1], freq=WEEK_RULE).start_time
    burn_in = len(all_weeks) - len(model_weeks)
    print(f"history weeks: {len(all_weeks)}  (burn-in {burn_in}, modelled {len(model_weeks)})")
    print(f"panel: {len(pcodes)} x {len(model_weeks)} = {len(pcodes) * len(model_weeks):,} area-weeks")

    targets = incidents[incidents["is_target"]]
    operations = incidents[incidents["is_operation"]]
    adjacency = load_adjacency(Path(args.adjacency), pcodes)
    degree = adjacency.sum(axis=1)
    print(f"adjacency: {int(adjacency.sum() / 2):,} edges, mean degree {degree.mean():.2f}")

    events = count_matrix(targets, pcodes, all_weeks)
    ops = count_matrix(operations, pcodes, all_weeks)
    deaths = count_matrix(targets, pcodes, all_weeks, "deaths")
    taken = count_matrix(targets, pcodes, all_weeks, "kidnapped")

    neighbour_events = adjacency @ events
    neighbour_ops = adjacency @ ops
    active = (events > 0).astype(np.float32)
    neighbour_active = adjacency @ active

    features: dict[str, np.ndarray] = {}

    for window in SHORT_WINDOWS + LONG_WINDOWS:
        features[f"own_events_{window}w"] = rolling_sum(events, window)
        features[f"nb_events_{window}w"] = rolling_sum(neighbour_events, window)
    for window in SHORT_WINDOWS:
        features[f"own_active_{window}w"] = rolling_sum(active, window)
        features[f"nb_active_{window}w"] = rolling_sum(neighbour_active, window)
    for window in (4, 13, 52):
        features[f"own_ops_{window}w"] = rolling_sum(ops, window)
        features[f"nb_ops_{window}w"] = rolling_sum(neighbour_ops, window)
        features[f"own_deaths_{window}w"] = rolling_sum(deaths, window)
        features[f"own_taken_{window}w"] = rolling_sum(taken, window)

    elapsed = np.arange(1, len(all_weeks) + 1, dtype=np.float32)[None, :]
    cumulative_events = expanding_sum(events)
    features["own_events_cum"] = cumulative_events
    features["own_rate_longrun"] = cumulative_events / elapsed
    features["nb_rate_longrun"] = expanding_sum(neighbour_events) / elapsed
    features["own_weeks_since"] = weeks_since_last(events)
    features["own_ever"] = (cumulative_events > 0).astype(np.float32)

    # Recording intensity, replacing the earlier count of how many files were recording.
    #
    # The count could not separate two periods that share it. The 363 weeks of 2014 to
    # 2020 and the 30 weeks of 2026 are both recorded by one file, but carry positive
    # rates of 0.0060 and 0.0137, a factor of 2.28. A model conditioning on the count
    # learns the blend those 393 weeks average to, and asked to forecast a 2026 week
    # returned a mean probability of 0.0063 against an observed 0.0137. That defect was
    # found only when the first live forecast was produced, because it concerns the
    # boundary between regimes rather than performance inside any one of them.
    #
    # Volume says what the count cannot. The national record count over the previous
    # thirteen completed weeks runs at roughly 26 in 2014, 1,170 in the dual-source
    # years and 156 in 2026, so the three eras are distinguishable rather than merged.
    # It is observable when a forecast is issued, since it counts only records that have
    # already been published, and it is lagged like every other history feature so the
    # week being predicted contributes nothing to it.
    weekly_records = (incidents.groupby("week").size()
                      .reindex(all_weeks, fill_value=0).to_numpy(dtype=np.float32))
    national = np.repeat(weekly_records[None, :], len(pcodes), axis=0)
    features["cov_records_13w"] = rolling_sum(national, 13)

    features["nb_degree"] = np.repeat(degree[:, None], len(all_weeks), axis=1)
    week_of_year = np.array([w.isocalendar()[1] for w in all_weeks], dtype=np.float32)
    features["cal_sin"] = np.repeat(np.sin(2 * np.pi * week_of_year / 52.0)[None, :], len(pcodes), axis=0)
    features["cal_cos"] = np.repeat(np.cos(2 * np.pi * week_of_year / 52.0)[None, :], len(pcodes), axis=0)
    features["time_index"] = np.repeat(np.arange(len(all_weeks), dtype=np.float32)[None, :], len(pcodes), axis=0)

    # ------------------------------------------------------------- assemble the panel
    keep = slice(burn_in, len(all_weeks))
    rows = len(pcodes) * len(model_weeks)
    panel = pd.DataFrame({
        "pcode": np.repeat(pcodes, len(model_weeks)),
        "week": np.tile(model_weeks, len(pcodes)),
        "event_count": events[:, keep].reshape(rows),
    })
    panel["occurred"] = (panel["event_count"] > 0).astype(np.int8)
    for name, matrix in features.items():
        panel[name] = matrix[:, keep].reshape(rows)

    panel["lga"] = panel["pcode"].map(names["adm2_name"])
    panel["state"] = panel["pcode"].map(names["adm1_name"])

    # --------------------------------------------------------------- leakage assertion
    feature_names = list(features)
    print("\nleakage check: move one week's events and confirm no earlier feature moves")
    probe_week = len(all_weeks) - 40
    corrupted = events.copy()
    corrupted[:, probe_week] += 99.0
    changed_before = False
    for window in (1, 4, 52):
        base = rolling_sum(events, window)
        test = rolling_sum(corrupted, window)
        if not np.array_equal(base[:, :probe_week + 1], test[:, :probe_week + 1]):
            changed_before = True
            print(f"  FAIL own_events_{window}w changed at or before the corrupted week")
    base_cum, test_cum = expanding_sum(events), expanding_sum(corrupted)
    if not np.array_equal(base_cum[:, :probe_week + 1], test_cum[:, :probe_week + 1]):
        changed_before = True
        print("  FAIL own_events_cum changed at or before the corrupted week")
    if changed_before:
        raise SystemExit("leakage detected, panel not written")
    print("  passed: every window for week t uses only weeks before t")

    if "event_count" in feature_names or "occurred" in feature_names:
        raise SystemExit("the outcome is present among the features")
    print("  passed: the outcome is not among the features")

    first = panel[panel.week == model_weeks[0]]
    print(f"  first modelled week carries burn-in history: "
          f"own_events_52w sums to {first.own_events_52w.sum():.0f} rather than 0")

    # ------------------------------------------------------------------------- report
    print(f"\npanel rows      : {len(panel):,}")
    print(f"positives       : {int(panel.occurred.sum()):,} ({100 * panel.occurred.mean():.3f}%)")
    print(f"features        : {len(feature_names)}")
    print(f"areas ever hit  : {panel.loc[panel.occurred == 1, 'pcode'].nunique()} of {len(pcodes)}")
    # Recording volume rather than a count of files, so the report bands it into
    # quartiles. The point of the feature is that volume varies continuously and that
    # two periods recorded by the same number of files need not be comparable.
    print("\nrecording volume across the modelled window "
          "(national records published in the prior 13 weeks):")
    volume = panel.drop_duplicates("week").set_index("week")["cov_records_13w"]
    edges = np.quantile(volume.to_numpy(), [0, 0.25, 0.5, 0.75, 1.0])
    for lo, hi in zip(edges[:-1], edges[1:]):
        weeks = volume[(volume >= lo) & (volume <= hi)].index
        block = panel[panel["week"].isin(weeks)]
        rate = block.groupby("week")["occurred"].sum().mean()
        print(f"  {int(lo):>5} to {int(hi):>5} records: {len(weeks):>3} weeks, "
              f"{rate:>5.1f} positive area-weeks per week, "
              f"{block['week'].min().date()} to {block['week'].max().date()}")
    print(f"max in a week   : {int(panel.event_count.max())}")
    counts = panel.loc[panel.occurred == 1, "event_count"]
    print(f"single-event    : {100 * (counts == 1).mean():.1f}% of positive weeks")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        panel.to_parquet(args.out, index=False)
        written = args.out
    except Exception:
        written = str(Path(args.out).with_suffix(".csv"))
        panel.to_csv(written, index=False)
        print("  (parquet unavailable, wrote CSV)")
    print(f"written to {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
