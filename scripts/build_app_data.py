"""Score every area for the coming week at 7, 14 and 28 days, for the Shiny interface.

The interface reads a scored table rather than fitting anything, for the reason given in
section 3.8: it keeps the deployed artefact from drifting away from the models the
dissertation evaluates. This script produces that table.

Three things make it different from the evaluation scripts.

First, it forecasts forward rather than backward. The panel ends at the last week the
corpus covers. A real forecast is issued for the week *after* that, so one extra week is
appended, its features are computed from completed weeks only, and it carries no label
because the answer is not known yet. That is the row the interface displays.

Second, it scores three horizons. The evaluation in section 3.7 compares 7, 14 and 28 days
and finds the seven-day window best on every comparable measure, so seven days is the
default the interface opens on. All three are produced because the comparison view lets a
user see how an area's risk and rank move as the window widens, which is a question the
evaluation answers in aggregate and the interface answers per area.

Third, bands are anchored per horizon. Section 3.8 sets the boundaries at two, four and
eight times the base rate over the 52 completed weeks before the forecast. That base rate
differs by horizon, roughly 0.036 at seven days against 0.113 at twenty-eight, so each
horizon gets its own anchor and the bands stay comparable in meaning across all three.

The model is the stack under its unweighted meta-learner, which section 3.6 establishes as
the form whose numbers can be read as probabilities. Drivers come from the penalised linear
member, whose contributions are exactly additive on the log-odds scale, and are labelled in
the interface as that member's account rather than the stack's.
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_panel import (  # noqa: E402
    MODEL_START, WEEK_RULE, SHORT_WINDOWS, LONG_WINDOWS,
    load_incidents, load_adjacency, count_matrix, expanding_sum,
)
from evaluate_protocol import (  # noqa: E402
    build_features, forward_label, calibrate, assemble,
)
from train_ensemble import fit_predict_gnn, logit  # noqa: E402

def days_label(h):
    return f"{h * 7}d"


HORIZONS = (1, 2, 4)                 # weeks: 7, 14 and 28 days
MULTIPLES = (8, 4, 2)                # Severe, High, Elevated cut points
ANCHOR_WEEKS = 52
TOP_DRIVERS = 6
RECENT_WEEKS = 8                     # incidents listed per area in the detail panel

READABLE = {
    "own_events_1w": "events here last week",
    "own_events_2w": "events here, last 2 weeks",
    "own_events_4w": "events here, last 4 weeks",
    "own_events_8w": "events here, last 8 weeks",
    "own_events_13w": "events here, last 13 weeks",
    "own_events_26w": "events here, last 26 weeks",
    "own_events_52w": "events here, last 52 weeks",
    "nb_events_1w": "events in neighbouring areas last week",
    "nb_events_2w": "events in neighbouring areas, last 2 weeks",
    "nb_events_4w": "events in neighbouring areas, last 4 weeks",
    "nb_events_8w": "events in neighbouring areas, last 8 weeks",
    "nb_events_13w": "events in neighbouring areas, last 13 weeks",
    "nb_events_26w": "events in neighbouring areas, last 26 weeks",
    "nb_events_52w": "events in neighbouring areas, last 52 weeks",
    "own_active_1w": "weeks with an event here, last 1",
    "own_active_2w": "weeks with an event here, last 2",
    "own_active_4w": "weeks with an event here, last 4",
    "nb_active_1w": "neighbour weeks with an event, last 1",
    "nb_active_2w": "neighbour weeks with an event, last 2",
    "nb_active_4w": "neighbour weeks with an event, last 4",
    "own_events_cum": "events here since 2011",
    "own_rate_longrun": "long-run weekly rate here",
    "nb_rate_longrun": "long-run weekly rate nearby",
    "own_weeks_since": "weeks since the last event here",
    "own_ever": "this area has recorded an event before",
    "own_ops_4w": "security operations here, last 4 weeks",
    "own_ops_13w": "security operations here, last 13 weeks",
    "own_ops_52w": "security operations here, last 52 weeks",
    "nb_ops_4w": "operations in neighbouring areas, last 4 weeks",
    "nb_ops_13w": "operations in neighbouring areas, last 13 weeks",
    "nb_ops_52w": "operations in neighbouring areas, last 52 weeks",
    "own_deaths_4w": "people killed here, last 4 weeks",
    "own_deaths_13w": "people killed here, last 13 weeks",
    "own_deaths_52w": "people killed here, last 52 weeks",
    "own_taken_4w": "people abducted here, last 4 weeks",
    "own_taken_13w": "people abducted here, last 13 weeks",
    "own_taken_52w": "people abducted here, last 52 weeks",
    "cov_records_13w": "records published nationally, last 13 weeks",
    "nb_degree": "number of neighbouring areas",
    "cal_sin": "time of year",
    "cal_cos": "time of year",
    "time_index": "position in the record",
}


# Features measuring nearly the same thing are reported together, because their separate
# weights are not interpretable even though their sum is.
#
# `own_rate_longrun` is `own_events_cum` divided by weeks elapsed, and the two correlate
# at 0.98 across the panel. A penalised linear model is free to split a large positive and
# a large negative between them without changing the total, and it does: in 641 of the 774
# areas both land in the top six, with magnitudes averaging 1.34 that combine to -0.09.
# Shown separately they read as "events here since 2011 lowers risk", which is not what
# the model means and is not something anyone should act on. Grouped, they read as one
# figure for how much this area has recorded over time.
DRIVER_FAMILY = {
    "own_events_cum": "history", "own_rate_longrun": "history", "own_ever": "history",
    "cal_sin": "season", "cal_cos": "season",
}
FAMILY_LABEL = {
    "history": "how much this area has recorded over the years",
    "season": "time of year",
}


def recalibrate_to_regime(probability: np.ndarray, target: float) -> tuple[np.ndarray, float]:
    """Shift predictions on the log-odds scale so their mean matches the recent rate.

    Still required, and the reason is worth stating because the expectation was otherwise.

    The coverage feature used to count how many files were publishing, which could not
    separate the sparse weeks of 2014 to 2020 from the 30 weeks of 2026 that share that
    count. Replacing it with publishing volume fixed that, and improved every model. It
    did not remove the level error, and the shift solved for here barely moved, from
    +0.867 to +0.841.

    The remaining error is in the history windows rather than in the coverage feature.
    They straddle the same boundary. Measured over the last eighteen completed weeks
    against the dual-source years, the one and four week counts fall to 0.29 of their
    former level while the fifty-two week count rises to 1.29, because that window still
    reaches back into 2025 when publishing was heavy. The model therefore sees an area
    with a busy year behind it and a very quiet month, which is the signature of a place
    that has calmed down, and reads it as lower risk. Knowing that publishing is thin does
    not by itself tell a model to discount that contrast; it would have to learn the
    interaction, and there are thirty weeks of evidence to learn it from.

    So the shift stays, as a correction rather than a diagnostic. It is solved for rather
    than assumed, monotone, and reported, so a reader can see exactly how far the forecast
    period has drifted from what the model was fitted on.

    That is a level error, not a ranking error. A single shift on the log-odds scale is
    monotone, so it moves every probability without reordering any area, and the order is
    what the ranked list and every evaluation measure in section 3.5 depend on. What it
    does fix is the banding, because section 3.8 compares a predicted probability against
    an observed rate, and that comparison is only meaningful when the two are on the same
    scale.

    Solving for the shift rather than assuming one keeps the correction auditable: the
    returned value says exactly how far the model had to be moved, and a shift near zero
    would mean no correction was needed.
    """
    from scipy.optimize import brentq
    clipped = np.clip(probability, 1e-9, 1 - 1e-9)
    odds = np.log(clipped / (1 - clipped))

    def gap(shift):
        return float(np.mean(1.0 / (1.0 + np.exp(-(odds + shift))))) - target

    if gap(0.0) == 0:
        return probability, 0.0
    lo, hi = -12.0, 12.0
    if gap(lo) * gap(hi) > 0:                 # target unreachable, leave it alone
        return probability, float("nan")
    shift = brentq(gap, lo, hi, xtol=1e-10)
    return 1.0 / (1.0 + np.exp(-(odds + shift))), float(shift)


def band_of(probability: np.ndarray, anchor: float) -> np.ndarray:
    return np.select([probability >= MULTIPLES[0] * anchor,
                      probability >= MULTIPLES[1] * anchor,
                      probability >= MULTIPLES[2] * anchor],
                     ["Severe", "High", "Elevated"], "Low")


def fit_stack(train, forecast_row, features, gnn):
    """The four base learners and the unweighted meta-learner of section 3.6.

    The inner block is the last 52 weeks of the training period, exactly as in
    `train_ensemble.py`, so the meta-learner is fitted on out-of-sample predictions.
    The graph network is included rather than dropped for convenience: section 3.6 found
    it takes the largest meta weight in four of the five folds, so a stack without it is
    a different model from the one the dissertation evaluates.
    """
    from xgboost import XGBClassifier
    gx, gy, edge_index, n_train_weeks, forecast_index = gnn
    weeks = np.sort(train["week"].unique())
    inner_start = len(weeks) - ANCHOR_WEEKS
    inner_train = train[train.week.isin(weeks[:inner_start])]
    inner_val = train[train.week.isin(weeks[inner_start:])]

    def tabular(kind, fit_on, predict_on):
        x = fit_on[features].to_numpy(np.float32)
        y = fit_on["y"].to_numpy()
        if kind == "logistic":
            base = make_pipeline(StandardScaler(),
                                 LogisticRegression(max_iter=2000, class_weight="balanced"))
            model = calibrate(base, x, y, fit_on["pcode"].to_numpy())
        elif kind == "random_forest":
            model = RandomForestClassifier(n_estimators=300, max_depth=14, min_samples_leaf=20,
                                           class_weight="balanced_subsample", n_jobs=4,
                                           random_state=7).fit(x, y)
        else:
            model = XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
                                  tree_method="hist", n_jobs=4, random_state=7).fit(x, y)
        return model.predict_proba(predict_on[features].to_numpy(np.float32))[:, 1], model

    names = ["logistic", "random_forest", "gradient_boosting", "stgnn"]
    meta_x, fore_x, fitted = {}, {}, {}
    for kind in names[:3]:
        meta_x[kind], _ = tabular(kind, inner_train, inner_val)
        fore_x[kind], fitted[kind] = tabular(kind, train, forecast_row)

    # The graph network runs strictly forward in time, so it predicts into the inner
    # block from a fit that stopped before it, then into the forecast week from a fit
    # that used every completed week.
    inner_end = n_train_weeks - ANCHOR_WEEKS
    meta_x["stgnn"] = fit_predict_gnn(gx, gy, edge_index, inner_end,
                                      range(inner_end, n_train_weeks), 7).reshape(-1)
    # The recurrence carries a hidden state forward one week at a time, so it has to be
    # walked through every week between the end of training and the forecast week rather
    # than jumped straight to it. At a four-week horizon the last three completed weeks
    # carry no settled label and so are excluded from training, but their features are
    # known and the state must still pass through them. Feeding only the forecast index
    # would silently drop them from the network's memory.
    walk = range(n_train_weeks, forecast_index + 1)
    fore_x["stgnn"] = fit_predict_gnn(gx, gy, edge_index, n_train_weeks,
                                      walk, 7)[-1].reshape(-1)

    meta = LogisticRegression(max_iter=2000)
    meta.fit(np.column_stack([logit(meta_x[k]) for k in names]),
             inner_val["y"].to_numpy())
    probability = meta.predict_proba(
        np.column_stack([logit(fore_x[k]) for k in names]))[:, 1]
    return probability, fitted["logistic"], dict(zip(names, meta.coef_[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incidents", default="data/processed/incidents_final.csv")
    parser.add_argument("--adjacency", default="data/reference/lga_adjacency.csv")
    parser.add_argument("--boundaries", default="data/reference/nga_admin_boundaries.xlsx")
    parser.add_argument("--out", default="app/data")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    incidents = load_incidents(Path(args.incidents))
    reference = pd.read_excel(args.boundaries, sheet_name="nga_admin2")
    pcodes = sorted(reference["adm2_pcode"].astype(str).unique())
    names = reference.set_index("adm2_pcode")[["adm2_name", "adm1_name"]]

    last_observed = incidents.loc[incidents["is_target"], "date"].max()
    model_weeks = pd.period_range(MODEL_START, last_observed, freq=WEEK_RULE).start_time
    forecast_week = model_weeks[-1] + pd.Timedelta(days=7)
    all_weeks = pd.period_range(incidents["week"].min(), forecast_week,
                                freq=WEEK_RULE).start_time
    burn_in = len(all_weeks) - len(model_weeks) - 1
    print(f"last completed week {model_weeks[-1].date()}, "
          f"forecasting week beginning {forecast_week.date()}")
    print(f"areas {len(pcodes)}, history weeks {len(all_weeks)}")

    targets = incidents[incidents["is_target"]]
    operations = incidents[incidents["is_operation"]]
    adjacency = load_adjacency(Path(args.adjacency), pcodes)
    degree = adjacency.sum(axis=1)
    events = count_matrix(targets, pcodes, all_weeks)
    ops = count_matrix(operations, pcodes, all_weeks)
    deaths = count_matrix(targets, pcodes, all_weeks, "deaths")
    taken = count_matrix(targets, pcodes, all_weeks, "kidnapped")
    # National publishing volume per week, which the recording-intensity feature of
    # section 3.4 is built from. Counted over the whole corpus rather than the target
    # events alone, because what it measures is how much is being published, not how
    # much violence there is.
    weekly_records = (incidents.groupby("week").size()
                      .reindex(all_weeks, fill_value=0).to_numpy(np.float32))

    feature_map = build_features(events, ops, deaths, taken, adjacency, degree,
                                 all_weeks, weekly_records, delay=0)
    features = list(feature_map)
    keep = slice(burn_in, len(all_weeks))
    all_model_weeks = list(model_weeks) + [forecast_week]

    # Tensors for the graph network, shaped weeks by areas by features, over the whole
    # span including the forecast week.
    import torch
    gx_raw = np.stack([feature_map[f] for f in features], axis=-1).transpose(1, 0, 2)
    flat = gx_raw.reshape(-1, len(features))
    gx = torch.from_numpy(((gx_raw - flat.mean(0)) / (flat.std(0) + 1e-6)).astype(np.float32))
    edges = pd.read_csv(args.adjacency)
    index = {p: i for i, p in enumerate(pcodes)}
    pairs = [(index[a], index[b]) for a, b in zip(edges.pcode_a, edges.pcode_b)
             if a in index and b in index]
    edge_index = torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).t().contiguous()

    rows = []
    driver_rows = []
    summary = {}
    for horizon in HORIZONS:
        labels = forward_label(events, horizon)
        frame = assemble(pcodes, all_model_weeks, feature_map, labels, keep, names)
        forecast = frame[frame.week == forecast_week].reset_index(drop=True)
        # Training stops where a complete forward window exists. At horizon h the last
        # h-1 completed weeks have no settled answer, so they are dropped rather than
        # filled with zero, exactly as in section 3.7.
        usable = [w for w in model_weeks[:len(model_weeks) - (horizon - 1)]]
        train = frame[frame.week.isin(usable)].reset_index(drop=True)
        # The anchor has to describe the same recording regime as the week being
        # forecast, not merely the weeks nearest to it.
        #
        # Section 3.4 records that the general incident log stops on 31 December 2025,
        # so the weeks after it are published at a far lower volume than the weeks
        # before. A plain 52-week window straddles that boundary: at the forecast week
        # it mixed 22 high-volume weeks with 30 low-volume ones and returned 0.0325
        # against 0.0137 for the low-volume weeks alone, a factor of 2.37. The model
        # conditions on recording volume and predicts what will be *recorded* at the
        # current volume, so banding those predictions against a rate drawn partly from
        # a busier period compares two different things and empties the upper bands. On
        # the first build it left one area in Severe and two in High.
        #
        # The anchor is therefore taken from the most recent *contiguous* run of weeks
        # sharing the forecast week's coverage value, still capped at 52. This is the
        # same principle section 3.8 states, which is that the anchor should say what is
        # normal now, applied to a panel where what counts as normal changed partway
        # through.
        #
        # Contiguity matters and the first attempt at this fix got it wrong. Collecting
        # every week at this coverage reaches back past the dual-source era into 2020,
        # when the specialist file recorded far less, and produced an anchor window
        # running from 2020-07-27 to 2026-07-27. Two periods can share a coverage count
        # and nothing else.
        # The anchor still has to describe a comparable recording regime. With volume
        # rather than a file count, "comparable" means within a factor of two of the
        # forecast week's own recent publishing rate, which keeps 2026 weeks together
        # and excludes both the dual-source years and the sparse early ones.
        forecast_volume = float(forecast["cov_records_13w"].iloc[0])
        volume_by_week = (frame.drop_duplicates("week")
                          .set_index("week")["cov_records_13w"].astype(float))
        lo, hi = forecast_volume / 2.0, forecast_volume * 2.0
        run: list = []
        for week in reversed(list(usable)):
            v = volume_by_week.get(week)
            if v is None or not (lo <= float(v) <= hi):
                break
            run.append(week)
        run.reverse()
        anchor_window = (run or list(usable))[-ANCHOR_WEEKS:]
        anchor = float(frame.loc[frame.week.isin(anchor_window), "y"].mean())
        print(f"  {horizon * 7:>2}d anchor from {len(anchor_window)} weeks near a volume of "
              f"{forecast_volume:.0f} ({anchor_window[0].date()} to "
              f"{anchor_window[-1].date()}): {anchor:.4f}")

        gy = torch.from_numpy(labels.astype(np.float32).T)
        n_train_weeks = burn_in + len(usable)
        forecast_index = len(all_weeks) - 1
        raw, linear, weights = fit_stack(
            train, forecast, features,
            (gx, gy, edge_index, n_train_weeks, forecast_index))
        probability, shift = recalibrate_to_regime(raw, anchor)
        print(f"  {days_label(horizon):>3} mean prediction {raw.mean():.4f} -> "
              f"{probability.mean():.4f} against a recent rate of {anchor:.4f} "
              f"(log-odds shift {shift:+.3f})")
        band = band_of(probability, anchor)
        order = (-probability).argsort()
        rank = np.empty(len(probability), dtype=int)
        rank[order] = np.arange(1, len(probability) + 1)

        days = horizon * 7
        block = pd.DataFrame({
            "pcode": forecast["pcode"], "horizon_days": days,
            "probability": probability, "band": band, "rank": rank,
            "anchor_rate": anchor,
        })
        rows.append(block)
        summary[str(days)] = {
            "anchor_rate": anchor,
            "anchor_weeks": len(anchor_window),
            "anchor_volume": forecast_volume,
            "anchor_from": str(anchor_window[0].date()),
            "raw_mean_prediction": float(raw.mean()),
            "recalibration_log_odds_shift": shift,
            "anchor_to": str(anchor_window[-1].date()),
            "cuts": {"Elevated": MULTIPLES[2] * anchor, "High": MULTIPLES[1] * anchor,
                     "Severe": MULTIPLES[0] * anchor},
            "band_counts": {b: int((band == b).sum()) for b in
                            ("Severe", "High", "Elevated", "Low")},
            "meta_weights": {k: float(v) for k, v in weights.items()},
            "training_weeks": len(usable),
        }
        print(f"  {days:>2}d: anchor {anchor:.4f}, cuts "
              f"{MULTIPLES[2] * anchor:.3f}/{MULTIPLES[1] * anchor:.3f}/"
              f"{MULTIPLES[0] * anchor:.3f}, bands " +
              ", ".join(f"{b} {int((band == b).sum())}" for b in
                        ("Severe", "High", "Elevated", "Low")))

        # Drivers from the linear member. Contributions are coefficient times the
        # standardised feature value, which is exactly additive on the log-odds scale.
        pipeline = linear.calibrated_classifiers_[0].estimator if hasattr(
            linear, "calibrated_classifiers_") else linear
        scaler = pipeline.named_steps["standardscaler"]
        coefficients = pipeline.named_steps["logisticregression"].coef_[0]
        x = forecast[features].to_numpy(np.float32)
        contribution = (x - scaler.mean_) / np.sqrt(scaler.var_) * coefficients
        groups = [DRIVER_FAMILY.get(f, f) for f in features]
        distinct = list(dict.fromkeys(groups))
        membership = {g: [j for j, gg in enumerate(groups) if gg == g] for g in distinct}
        for i, pcode in enumerate(forecast["pcode"]):
            totals = {g: float(contribution[i, idx].sum()) for g, idx in membership.items()}
            top = sorted(totals, key=lambda g: -abs(totals[g]))[:TOP_DRIVERS]
            for g in top:
                idx = membership[g]
                # A group with one member reports that member's value. A group with
                # several reports none, because there is no single quantity to report,
                # and inventing one would be worse than leaving the column blank.
                driver_rows.append({
                    "pcode": pcode, "horizon_days": days,
                    "feature": g,
                    "label": FAMILY_LABEL.get(g, READABLE.get(g, g)),
                    "value": float(x[i, idx[0]]) if len(idx) == 1 else float("nan"),
                    "contribution": totals[g],
                    "n_features": len(idx),
                })

    scored = pd.concat(rows, ignore_index=True)
    scored["lga"] = scored["pcode"].map(names["adm2_name"])
    scored["state"] = scored["pcode"].map(names["adm1_name"])
    scored.to_csv(out / "forecast.csv", index=False)
    pd.DataFrame(driver_rows).to_csv(out / "drivers.csv", index=False)

    # Recent incidents for the per-area detail panel.
    cutoff = model_weeks[-1] - pd.Timedelta(weeks=RECENT_WEEKS - 1)
    recent = targets[(targets["date"] >= cutoff) & targets["pcode"].notna()].copy()
    recent = recent[["date", "pcode", "location", "deaths", "kidnapped",
                     "perpetrator", "narrative"]]
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    recent = recent.sort_values("date", ascending=False)
    recent.to_csv(out / "recent_incidents.csv", index=False)


    summary["forecast_week"] = str(forecast_week.date())
    summary["last_completed_week"] = str(model_weeks[-1].date())
    summary["areas"] = len(pcodes)
    summary["generated_from"] = "unweighted stack, section 3.6"
    (out / "meta.json").write_text(json.dumps(summary, indent=1))

    print(f"\nwritten to {out}/: forecast.csv ({len(scored):,} rows), "
          f"drivers.csv ({len(driver_rows):,}), recent_incidents.csv ({len(recent):,}), "
          f"meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
