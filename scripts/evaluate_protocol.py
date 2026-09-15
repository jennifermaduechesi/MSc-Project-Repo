"""Three experiments that make up the evaluation protocol in section 3.7.

Each answers a supervisor correction that the modelling sections do not reach.

  A. Horizon      C7 asked for a comparison of 7, 14 and 30 day horizons. The panel is
                  weekly, so the horizons that can actually be built are one, two and
                  four weeks, which is 7, 14 and 28 days. The substitution is stated
                  rather than made quietly.
  B. Delay        C2 asked for the one to two week reporting lag in conflict data to be
                  accounted for. A forecast issued on the Monday of week t cannot in
                  practice see events from week t-1 if those events take a fortnight to
                  reach a published record. The delay is simulated by withholding the
                  most recent weeks from every feature.
  C. Under-report C4 asked for under-reporting to be handled in the model rather than
                  noted in the ethics section, on the grounds that it biases the loss
                  function. Detection is estimated from the overlap between the two
                  supplied files, and the models are refitted with the loss weighted by
                  the inverse of that estimate.

One model is used throughout, the calibrated penalised logistic regression specified
in section 3.5. Holding the model fixed is the point. These experiments ask what the
data and the forecasting setup do to performance, and a changing model would confound
that with a difference of algorithm. The graph network is also far too slow to refit
across nine configurations on the processor available.

Every feature matrix here is rebuilt rather than read from the committed panel, because
the delay experiment needs the windows shifted and the horizon experiment needs the
labels extended. The rebuild is checked against `panel.parquet` at a delay of zero
before anything else runs, so a divergence between this script and `build_panel.py`
stops the run instead of quietly producing different numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_panel import (  # noqa: E402
    MODEL_START, WEEK_RULE, SHORT_WINDOWS, LONG_WINDOWS,
    load_incidents, load_adjacency, count_matrix, expanding_sum,
)

TEST_WEEKS = 52
N_FOLDS = 5
TOP_K = (10, 20, 50)
HORIZONS = (1, 2, 4)          # weeks, which is 7, 14 and 28 days
DELAYS = (0, 1, 2)            # weeks withheld on top of the standard one-week lag
DUAL_MATCH_DAYS = (0, 1)      # tolerance for linking a record across the two files


# --------------------------------------------------------- features at a chosen delay
def shift(matrix: np.ndarray, k: int) -> np.ndarray:
    """Move every column k weeks forward. Column t then holds week t-k."""
    if k <= 0:
        return matrix
    out = np.zeros_like(matrix)
    out[:, k:] = matrix[:, :-k]
    return out


def rolling_sum(matrix: np.ndarray, window: int, delay: int = 0) -> np.ndarray:
    """Sum the `window` weeks ending at t-1-delay."""
    past = shift(matrix, 1 + delay)
    cumulative = np.cumsum(past, axis=1)
    out = cumulative.copy()
    out[:, window:] = cumulative[:, window:] - cumulative[:, :-window]
    return out


def weeks_since_last(matrix: np.ndarray, delay: int = 0, cap: int = 520) -> np.ndarray:
    past = shift(matrix, 1 + delay)
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


def build_features(events, ops, deaths, taken, adjacency, degree, all_weeks,
                   dual_from, dual_to, delay: int) -> dict[str, np.ndarray]:
    """The forty-two features of section 3.4, with an extra `delay` weeks withheld.

    This mirrors `build_panel.py` deliberately. `check_against_panel` below asserts
    that at delay zero it reproduces the committed panel exactly, which is what keeps
    the two from drifting apart.
    """
    neighbour_events = adjacency @ events
    neighbour_ops = adjacency @ ops
    active = (events > 0).astype(np.float32)
    neighbour_active = adjacency @ active
    f: dict[str, np.ndarray] = {}

    for w in SHORT_WINDOWS + LONG_WINDOWS:
        f[f"own_events_{w}w"] = rolling_sum(events, w, delay)
        f[f"nb_events_{w}w"] = rolling_sum(neighbour_events, w, delay)
    for w in SHORT_WINDOWS:
        f[f"own_active_{w}w"] = rolling_sum(active, w, delay)
        f[f"nb_active_{w}w"] = rolling_sum(neighbour_active, w, delay)
    for w in (4, 13, 52):
        f[f"own_ops_{w}w"] = rolling_sum(ops, w, delay)
        f[f"nb_ops_{w}w"] = rolling_sum(neighbour_ops, w, delay)
        f[f"own_deaths_{w}w"] = rolling_sum(deaths, w, delay)
        f[f"own_taken_{w}w"] = rolling_sum(taken, w, delay)

    elapsed = np.arange(1, len(all_weeks) + 1, dtype=np.float32)[None, :]
    cumulative = expanding_sum(shift(events, delay))
    f["own_events_cum"] = cumulative
    f["own_rate_longrun"] = cumulative / elapsed
    f["nb_rate_longrun"] = expanding_sum(shift(neighbour_events, delay)) / elapsed
    f["own_weeks_since"] = weeks_since_last(events, delay)
    f["own_ever"] = (cumulative > 0).astype(np.float32)

    # The coverage indicator describes which files were recording, which a forecaster
    # knows without waiting for any report, so a reporting delay does not touch it.
    dual = ((all_weeks >= dual_from) & (all_weeks <= dual_to)).astype(np.float32)
    f["cov_sources"] = np.repeat((1.0 + dual)[None, :], events.shape[0], axis=0)
    f["nb_degree"] = np.repeat(degree[:, None], len(all_weeks), axis=1)
    week_of_year = np.array([w.isocalendar()[1] for w in all_weeks], dtype=np.float32)
    f["cal_sin"] = np.repeat(np.sin(2 * np.pi * week_of_year / 52.0)[None, :], events.shape[0], axis=0)
    f["cal_cos"] = np.repeat(np.cos(2 * np.pi * week_of_year / 52.0)[None, :], events.shape[0], axis=0)
    f["time_index"] = np.repeat(np.arange(len(all_weeks), dtype=np.float32)[None, :], events.shape[0], axis=0)
    return f


def forward_label(events: np.ndarray, horizon: int) -> np.ndarray:
    """1 where at least one event falls in weeks t to t+horizon-1 inclusive."""
    areas, periods = events.shape
    out = np.zeros((areas, periods), dtype=np.int8)
    cumulative = np.cumsum(events, axis=1)
    for t in range(periods):
        end = min(t + horizon, periods)
        total = cumulative[:, end - 1] - (cumulative[:, t - 1] if t > 0 else 0.0)
        out[:, t] = (total > 0).astype(np.int8)
    return out


# ------------------------------------------------------------------------- evaluation
def recall_at_k(frame: pd.DataFrame, column: str, k: int) -> float:
    hits = total = 0
    for _, block in frame.groupby("week", sort=False):
        positives = int(block["y"].sum())
        if positives == 0:
            continue
        hits += int(block.nlargest(k, column)["y"].sum())
        total += positives
    return hits / total if total else float("nan")


def evaluate(frame: pd.DataFrame, column: str, n_areas: int) -> dict:
    y, s = frame["y"].to_numpy(), frame[column].to_numpy()
    base = float(y.mean())
    ap = float(average_precision_score(y, s))
    out = {"base_rate": base, "average_precision": ap,
           "ap_lift_over_base": ap / base if base else float("nan"),
           "roc_auc": float(roc_auc_score(y, s)),
           "brier": float(brier_score_loss(y, np.clip(s, 0, 1)))}
    for k in TOP_K:
        r = recall_at_k(frame, column, k)
        out[f"recall_at_{k}"] = r
        out[f"lift_at_{k}"] = r / (k / n_areas) if r == r else float("nan")
    return out


def calibrate(base, x, y, area_of_row, n_splits=3):
    """Isotonic calibration on folds split by area. See section 3.6 for why."""
    codes = pd.factorize(area_of_row)[0]
    idx = np.arange(len(codes))
    splits = []
    for fold in range(n_splits):
        held = codes % n_splits == fold
        if held.sum() < 50 or (~held).sum() < 50:
            base.fit(x, y)
            return base
        splits.append((idx[~held], idx[held]))
    model = CalibratedClassifierCV(base, method="isotonic", cv=splits)
    model.fit(x, y)
    return model


def run_folds(frame: pd.DataFrame, feature_names: list[str], weeks: np.ndarray,
              n_areas: int, sample_weight: np.ndarray | None = None) -> list[dict]:
    """Rolling origin, five folds, the calibrated logistic model of section 3.5."""
    bounds, end = [], len(weeks)
    for _ in range(N_FOLDS):
        start = end - TEST_WEEKS
        if start <= 0:
            break
        bounds.append((start, end))
        end = start

    results = []
    for fold, (test_start, test_end) in enumerate(reversed(bounds), 1):
        train = frame[frame.week.isin(weeks[:test_start])]
        test = frame[frame.week.isin(weeks[test_start:test_end])].copy()
        x = train[feature_names].to_numpy(np.float32)
        y = train["y"].to_numpy()
        base = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
        if sample_weight is not None:
            # Weighting is applied by resampling indices rather than by sample_weight,
            # because the calibration wrapper refits the estimator internally and does
            # not pass a weight vector through to it.
            w = sample_weight[train.index.to_numpy()]
            rng = np.random.default_rng(7 + fold)
            probability = w / w.sum()
            draw = rng.choice(len(w), size=len(w), replace=True, p=probability)
            x, y = x[draw], y[draw]
            areas = train["pcode"].to_numpy()[draw]
        else:
            areas = train["pcode"].to_numpy()
        model = calibrate(base, x, y, areas)
        test["score"] = model.predict_proba(test[feature_names].to_numpy(np.float32))[:, 1]
        metrics = evaluate(test, "score", n_areas)
        metrics["fold"] = fold
        metrics["test_from"] = str(pd.Timestamp(weeks[test_start]).date())
        metrics["test_to"] = str(pd.Timestamp(weeks[test_end - 1]).date())
        metrics["train_weeks"] = int(test_start)
        results.append(metrics)
    return results


def summarise(rows: list[dict]) -> dict:
    keys = ("average_precision", "ap_lift_over_base", "recall_at_20", "lift_at_20",
            "roc_auc", "brier", "base_rate")
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def show(label: str, rows: list[dict]) -> None:
    m = summarise(rows)
    print(f"  {label:<26} base {m['base_rate']:.4f} | AP {m['average_precision']:.4f} "
          f"| AP lift {m['ap_lift_over_base']:>5.1f}x | R@20 {m['recall_at_20']:.3f} "
          f"| lift@20 {m['lift_at_20']:>4.1f} | Brier {m['brier']:.4f}")


# ------------------------------------------------------------------ under-reporting
def detection_by_state(incidents: pd.DataFrame, pcode_to_state: dict,
                       tolerance_days: int) -> pd.DataFrame:
    """Chapman capture-recapture on the two supplied files, by state.

    The two files overlap between 2021 and 2025. Treating each as an independent
    attempt to record the same events gives a coverage estimate per state. The
    assumptions are only partly met and section 3.7 says so: the specialist file
    selects on kidnapping rather than sampling at random, and the two draw on
    overlapping media, so the figure describes differential coverage between the
    sources rather than a true detection rate.
    """
    target = incidents[incidents["is_target"] & incidents["pcode"].notna()].copy()
    dual = target[(target["date"] >= "2021-01-01") & (target["date"] <= "2025-12-31")]
    rows = []
    for state, group in dual.groupby(group_state(dual, pcode_to_state)):
        main = group[group.source_file == "main"]
        extra = group[group.source_file == "additional"]
        m_cells = set(zip(main["pcode"], main["date"]))
        a_cells = set(zip(extra["pcode"], extra["date"]))
        if tolerance_days:
            matched = 0
            for pcode, day in a_cells:
                offsets = range(-tolerance_days, tolerance_days + 1)
                if any((pcode, day + pd.Timedelta(days=o)) in m_cells for o in offsets):
                    matched += 1
        else:
            matched = len(m_cells & a_cells)
        n1, n2 = len(m_cells), len(a_cells)
        chapman = ((n1 + 1) * (n2 + 1) / (matched + 1)) - 1
        observed = len(m_cells | a_cells)
        rows.append({"state": state, "n_main": n1, "n_additional": n2,
                     "matched": matched, "observed": observed,
                     "estimated_total": chapman,
                     "detection": observed / chapman if chapman > 0 else np.nan})
    return pd.DataFrame(rows).sort_values("observed", ascending=False)


def group_state(frame: pd.DataFrame, pcode_to_state: dict) -> pd.Series:
    return frame["pcode"].map(pcode_to_state).fillna("unknown")


# ------------------------------------------------------------------------------ main
def check_against_panel(panel: pd.DataFrame, rebuilt: pd.DataFrame,
                        feature_names: list[str]) -> None:
    """Refuse to run if this script's features differ from the committed panel."""
    left = panel.sort_values(["pcode", "week"]).reset_index(drop=True)
    right = rebuilt.sort_values(["pcode", "week"]).reset_index(drop=True)
    if len(left) != len(right):
        raise SystemExit(f"row count differs: panel {len(left)}, rebuilt {len(right)}")
    worst_name, worst = None, 0.0
    for name in feature_names:
        gap = float(np.abs(left[name].to_numpy() - right[name].to_numpy()).max())
        if gap > worst:
            worst_name, worst = name, gap
    if worst > 1e-4:
        raise SystemExit(f"rebuilt features differ from the panel: {worst_name} by {worst}")
    label_gap = int((left["occurred"].to_numpy() != right["y"].to_numpy()).sum())
    if label_gap:
        raise SystemExit(f"rebuilt labels differ from the panel in {label_gap} rows")
    print(f"  rebuild matches the committed panel, largest feature difference "
          f"{worst:.2e} ({worst_name})")


def assemble(pcodes, model_weeks, features, labels, keep, names) -> pd.DataFrame:
    rows = len(pcodes) * len(model_weeks)
    frame = pd.DataFrame({"pcode": np.repeat(pcodes, len(model_weeks)),
                          "week": np.tile(model_weeks, len(pcodes)),
                          "y": labels[:, keep].reshape(rows)})
    for name, matrix in features.items():
        frame[name] = matrix[:, keep].reshape(rows)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--incidents", default="data/processed/incidents_final.csv")
    parser.add_argument("--adjacency", default="data/reference/lga_adjacency.csv")
    parser.add_argument("--boundaries", default="data/reference/nga_admin_boundaries.xlsx")
    parser.add_argument("--out", default="data/processed/protocol_results.json")
    args = parser.parse_args()

    incidents = load_incidents(Path(args.incidents))
    reference = pd.read_excel(args.boundaries, sheet_name="nga_admin2")
    pcodes = sorted(reference["adm2_pcode"].astype(str).unique())
    names = reference.set_index("adm2_pcode")[["adm2_name", "adm1_name"]]
    pcode_to_state = names["adm1_name"].to_dict()

    model_end = incidents.loc[incidents["is_target"], "date"].max()
    model_weeks = pd.period_range(MODEL_START, model_end, freq=WEEK_RULE).start_time
    all_weeks = pd.period_range(incidents["week"].min(), model_weeks[-1],
                                freq=WEEK_RULE).start_time
    burn_in = len(all_weeks) - len(model_weeks)
    keep = slice(burn_in, len(all_weeks))

    targets = incidents[incidents["is_target"]]
    operations = incidents[incidents["is_operation"]]
    adjacency = load_adjacency(Path(args.adjacency), pcodes)
    degree = adjacency.sum(axis=1)
    events = count_matrix(targets, pcodes, all_weeks)
    ops = count_matrix(operations, pcodes, all_weeks)
    deaths = count_matrix(targets, pcodes, all_weeks, "deaths")
    taken = count_matrix(targets, pcodes, all_weeks, "kidnapped")
    main_log = incidents[incidents["source_file"] == "main"]
    dual_from, dual_to = main_log["week"].min(), main_log["week"].max()

    print(f"areas {len(pcodes)}, modelled weeks {len(model_weeks)}, burn-in {burn_in}")

    panel = pd.read_parquet(args.panel)
    results: dict[str, object] = {}

    # ------------------------------------------------------------------ verification
    print("\nverification")
    features0 = build_features(events, ops, deaths, taken, adjacency, degree,
                               all_weeks, dual_from, dual_to, delay=0)
    feature_names = list(features0)
    labels1 = forward_label(events, 1)
    base_frame = assemble(pcodes, model_weeks, features0, labels1, keep, names)
    check_against_panel(panel, base_frame, feature_names)

    # ------------------------------------------------------------- A. horizon (C7)
    print("\nA. horizon comparison, 7 / 14 / 28 days")
    horizon_rows = {}
    for h in HORIZONS:
        labels = forward_label(events, h)
        frame = assemble(pcodes, model_weeks, features0, labels, keep, names)
        # The last h-1 modelled weeks have no complete forward window. Filling them
        # with zero would teach the model that the end of the record was peaceful.
        if h > 1:
            frame = frame[frame.week <= model_weeks[-(h - 1) - 1]]
            dropped = len(pcodes) * len(model_weeks) - len(frame)
            assert dropped == len(pcodes) * (h - 1), (
                f"truncation dropped {dropped}, expected {len(pcodes) * (h - 1)}")
            print(f"  horizon {h}w: dropped {dropped:,} rows "
                  f"({len(pcodes)} areas x {h - 1} week(s))")
        weeks = np.sort(frame["week"].unique())
        rows = run_folds(frame.reset_index(drop=True), feature_names, weeks, len(pcodes))
        horizon_rows[f"{h * 7}d"] = rows
        show(f"{h * 7} days ({h}w)", rows)
    results["horizon"] = horizon_rows

    # --------------------------------------------------------------- B. delay (C2)
    print("\nB. reporting delay, 0 / 1 / 2 weeks withheld")
    delay_rows = {}
    for d in DELAYS:
        features = features0 if d == 0 else build_features(
            events, ops, deaths, taken, adjacency, degree, all_weeks,
            dual_from, dual_to, delay=d)
        frame = assemble(pcodes, model_weeks, features, labels1, keep, names)
        weeks = np.sort(frame["week"].unique())
        rows = run_folds(frame.reset_index(drop=True), feature_names, weeks, len(pcodes))
        delay_rows[f"{d}w"] = rows
        show(f"delay {d} week(s)", rows)
    results["delay"] = delay_rows

    # ------------------------------------------------- C. under-reporting (C4)
    print("\nC. under-reporting, detection estimated from the two supplied files")
    detection_tables = {}
    for tolerance in DUAL_MATCH_DAYS:
        table = detection_by_state(incidents, pcode_to_state, tolerance)
        detection_tables[f"pm{tolerance}d"] = table.to_dict("records")
        finite = table["detection"].dropna()
        print(f"  match tolerance +/-{tolerance} day(s): {len(table)} states, "
              f"detection median {finite.median():.3f}, "
              f"range {finite.min():.3f} to {finite.max():.3f}, "
              f"{int((table['matched'] == 0).sum())} state(s) with no linked record")
    results["detection"] = detection_tables

    table = detection_by_state(incidents, pcode_to_state, 0)
    detection = table.set_index("state")["detection"]
    median = float(detection.median())
    frame = assemble(pcodes, model_weeks, features0, labels1, keep, names)
    frame["state"] = frame["pcode"].map(pcode_to_state)
    frame = frame.reset_index(drop=True)
    weights = frame["state"].map(detection).fillna(median).rpow(-1.0).to_numpy()
    weights = np.clip(weights, 0.0, np.percentile(weights, 99))
    weeks = np.sort(frame["week"].unique())
    plain = run_folds(frame, feature_names, weeks, len(pcodes))
    reweighted = run_folds(frame, feature_names, weeks, len(pcodes), sample_weight=weights)
    show("unweighted", plain)
    show("inverse-detection", reweighted)
    results["under_reporting"] = {"unweighted": plain, "reweighted": reweighted,
                                  "weight_min": float(weights.min()),
                                  "weight_max": float(weights.max())}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1, default=str))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
