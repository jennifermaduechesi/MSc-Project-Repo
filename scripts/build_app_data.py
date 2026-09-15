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
    "cov_sources": "number of source files recording",
    "nb_degree": "number of neighbouring areas",
    "cal_sin": "time of year",
    "cal_cos": "time of year",
    "time_index": "position in the record",
}


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
    fore_x["stgnn"] = fit_predict_gnn(gx, gy, edge_index, n_train_weeks,
                                      [forecast_index], 7).reshape(-1)

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
    main_log = incidents[incidents["source_file"] == "main"]
    dual_from, dual_to = main_log["week"].min(), main_log["week"].max()

    feature_map = build_features(events, ops, deaths, taken, adjacency, degree,
                                 all_weeks, dual_from, dual_to, delay=0)
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
        anchor_window = usable[-ANCHOR_WEEKS:]
        anchor = float(frame.loc[frame.week.isin(anchor_window), "y"].mean())

        gy = torch.from_numpy(labels.astype(np.float32).T)
        n_train_weeks = burn_in + len(usable)
        forecast_index = len(all_weeks) - 1
        probability, linear, weights = fit_stack(
            train, forecast, features,
            (gx, gy, edge_index, n_train_weeks, forecast_index))
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
        for i, pcode in enumerate(forecast["pcode"]):
            top = np.argsort(-np.abs(contribution[i]))[:TOP_DRIVERS]
            for j in top:
                driver_rows.append({
                    "pcode": pcode, "horizon_days": days,
                    "feature": features[j],
                    "label": READABLE.get(features[j], features[j]),
                    "value": float(x[i, j]),
                    "contribution": float(contribution[i, j]),
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

    history = (pd.DataFrame({"pcode": np.repeat(pcodes, len(all_weeks)),
                             "week": np.tile(all_weeks, len(pcodes)),
                             "events": events.reshape(-1)})
               .query("week >= '2021-01-01'"))
    history.to_csv(out / "history.csv", index=False)

    summary["forecast_week"] = str(forecast_week.date())
    summary["last_completed_week"] = str(model_weeks[-1].date())
    summary["areas"] = len(pcodes)
    summary["generated_from"] = "unweighted stack, section 3.6"
    (out / "meta.json").write_text(json.dumps(summary, indent=1))

    print(f"\nwritten to {out}/: forecast.csv ({len(scored):,} rows), "
          f"drivers.csv ({len(driver_rows):,}), recent_incidents.csv ({len(recent):,}), "
          f"history.csv ({len(history):,}), meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
