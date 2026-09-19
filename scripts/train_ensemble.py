"""Stack four base learners under a logistic meta-learner, against two baselines.

The ensemble combines four base learners under a meta-learner, as specified in
section 3.6 of the methodology.

Chapter Two gives the rationale for stacking rather than bagging or boosting.
Hajihosseinlou et al. (2024) argue that bagging targets variance and boosting targets
bias, while stacking aims at both by learning how to integrate genuinely different model
families. Barton and Lennox (2022) add that stacking avoids model selection, since
choosing one model by cross-validation is not guaranteed to pick the one that
generalises best. The four base learners here are deliberately different in kind: a
penalised linear model, a bagged tree ensemble, a boosted tree ensemble, and a recurrent
graph network.

**Two baselines are carried through the whole comparison**, because an ensemble that
beats its own members but not a trivial rule has not earned anything.

    recency        Rank areas by how recently and how often they were attacked, using
                   the count over the previous four weeks with the long-run rate
                   breaking ties. This is the current practice Chapter One describes,
                   which is allocating by reference to the most recent incident list,
                   and objective three asks explicitly whether the system improves
                   on it.
    long_run       Rank areas by their long-run event rate alone, ignoring recent
                   activity. This separates the flag component from the boost component
                   in S. D. Johnson's terms: if the long-run rate alone ranks nearly as
                   well as anything else, then what the models add is mostly knowing
                   which areas are dangerous in general rather than which are dangerous
                   this week.

**Leakage in a stack is subtle and is handled explicitly.** A meta-learner trained on
predictions the base learners made about rows they were fitted on would learn from
memorised answers. Each outer fold therefore holds out an inner block of 52 weeks: base
learners are fitted on everything before it, predict into it, and the meta-learner is
fitted on those out-of-sample predictions alone. The base learners are then refitted on
the full training period and the fitted meta-learner is applied to their test-period
predictions. No model is ever asked to combine predictions it made about its own
training rows.

The meta-learner is a logistic regression on the log-odds of the base predictions rather
than on the probabilities. On the log-odds scale a linear combination is a weighted
product of odds, which is the natural way to pool probabilistic forecasts, and it keeps
the meta-learner from being dominated by whichever base model happens to be worst
calibrated.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
from torch_geometric_temporal.nn.recurrent import GConvGRU  # noqa: E402

TEST_WEEKS = 52
INNER_WEEKS = 52
N_FOLDS = 5
TOP_K = (10, 20, 50)
NOT_FEATURES = {"pcode", "week", "lga", "state", "event_count", "occurred"}
GNN_HIDDEN, GNN_K, GNN_EPOCHS, GNN_LR = 32, 2, 10, 0.01
EPS = 1e-6
PROBABILISTIC = {"logistic", "random_forest", "gradient_boosting", "stgnn",
                 "ensemble", "ensemble_unweighted", "long_run"}


# ------------------------------------------------------------------------ measures
def recall_at_k(frame: pd.DataFrame, column: str, k: int) -> float:
    hits = total = 0
    for _, block in frame.groupby("week", sort=False):
        positives = int(block["occurred"].sum())
        if positives == 0:
            continue
        hits += int(block.nlargest(k, column)["occurred"].sum())
        total += positives
    return hits / total if total else float("nan")


def evaluate(frame: pd.DataFrame, column: str) -> dict:
    y, s = frame["occurred"].to_numpy(), frame[column].to_numpy()
    base = float(y.mean())
    out = {
        "base_rate": base,
        "average_precision": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        "ap_lift_over_base": float(average_precision_score(y, s)) / base if base else float("nan"),
    }
    # Brier is reported only for scores that are meant to be read as a probability of
    # occurrence. The four learners and the stack qualify by construction. `long_run`
    # qualifies by argument rather than by construction: it is a mean event count per
    # week, not a probability, but 81.3 per cent of positive area-weeks hold exactly
    # one event, so it approximates the probability of at least one closely enough to
    # be worth scoring. `recency` is a raw four-week count reaching 27 and does not
    # qualify at all.
    #
    # This is stated as a named set rather than tested as a range. The range test
    # returns the same answer on this panel, but only because events are scarce
    # enough to keep `long_run` below 0.25, which is a property of the data and not a
    # decision anyone made. The calibration fault recorded in `calibrate_probabilities`
    # came from exactly that kind of accidental dependence.
    if column in PROBABILISTIC:
        out["brier"] = float(brier_score_loss(y, np.clip(s, 0.0, 1.0)))
    for k in TOP_K:
        r = recall_at_k(frame, column, k)
        out[f"recall_at_{k}"] = r
        out[f"lift_at_{k}"] = r / (k / 774) if r == r else float("nan")
    return out


# --------------------------------------------------------------------- base learners

def calibrate_probabilities(base, x, y, area_of_row, n_splits=3):
    """Learn an isotonic calibration map using folds split by area, not by time.

    `CalibratedClassifierCV(..., cv=3)` cannot be used directly here. It runs KFold
    without shuffling, so its folds follow whatever order the rows arrive in. With
    rows ordered by area each fold spans the whole period and calibration is
    harmless; with rows ordered by week each fold becomes a contiguous time block,
    every internal model trains on two thirds of the period and extrapolates over
    the rest, and average precision on fold four fell from 0.262 to 0.183 for that
    reason alone. Neither script chose that behaviour. Both inherited it from row
    order, which is why it is replaced with something explicit.

    Splitting by area was chosen over holding out the most recent weeks. A temporal
    holdout is the intuitive choice, but it costs the base model the most recent and
    most informative fifth of its training data, and measured across the five folds
    it reduced average precision from 0.188 to 0.160 for the linear model and from
    0.173 to 0.134 for the boosted one. Splitting by area keeps every sub-model
    trained on the full time range while still calibrating on rows it did not fit.
    Nothing here uses information from after the training cutoff, so the split is a
    choice about variance rather than about leakage.
    """
    codes = pd.factorize(area_of_row)[0]
    indices = np.arange(len(codes))
    splits = []
    for fold in range(n_splits):
        held = codes % n_splits == fold
        if held.sum() < 50 or (~held).sum() < 50:
            base.fit(x, y)
            return base
        splits.append((indices[~held], indices[held]))
    model = CalibratedClassifierCV(base, method="isotonic", cv=splits)
    model.fit(x, y)
    return model


def fit_predict_tabular(kind, train, predict_on, features):
    x, y = train[features].to_numpy(np.float32), train["occurred"].to_numpy()
    if kind == "logistic":
        base = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=2000, class_weight="balanced"))
        model = calibrate_probabilities(base, x, y, train["pcode"].to_numpy())
        return model.predict_proba(predict_on[features].to_numpy(np.float32))[:, 1]
    elif kind == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=20,
            class_weight="balanced_subsample", n_jobs=4, random_state=7)
    elif kind == "gradient_boosting":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, eval_metric="aucpr", tree_method="hist",
            n_jobs=4, random_state=7)
    else:
        raise ValueError(kind)
    model.fit(x, y)
    return model.predict_proba(predict_on[features].to_numpy(np.float32))[:, 1]


class RiskNet(torch.nn.Module):
    def __init__(self, in_channels, hidden, k):
        super().__init__()
        self.recurrent = GConvGRU(in_channels=in_channels, out_channels=hidden, K=k)
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, hidden):
        hidden = self.recurrent(x, edge_index, None, hidden)
        return self.head(hidden).squeeze(-1), hidden


def fit_predict_gnn(x, y, edge_index, train_end, predict_slice, seed):
    torch.manual_seed(seed)
    model = RiskNet(x.shape[2], GNN_HIDDEN, GNN_K)
    optimiser = torch.optim.Adam(model.parameters(), lr=GNN_LR)
    weight = torch.tensor(float((y[:train_end] == 0).sum() / max((y[:train_end] == 1).sum(), 1)))
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    for _ in range(GNN_EPOCHS):
        model.train()
        hidden = None
        optimiser.zero_grad()
        for t in range(train_end):
            logits, hidden = model(x[t], edge_index, hidden)
            criterion(logits, y[t]).backward()
            hidden = hidden.detach()
            if (t + 1) % 26 == 0 or t == train_end - 1:
                optimiser.step()
                optimiser.zero_grad()
    model.eval()
    with torch.no_grad():
        hidden = None
        for t in range(train_end):
            _, hidden = model(x[t], edge_index, hidden)
        scores = []
        for t in predict_slice:
            logits, hidden = model(x[t], edge_index, hidden)
            scores.append(torch.sigmoid(logits).numpy())
    return np.stack(scores)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", default="data/processed/panel.parquet")
    parser.add_argument("--adjacency", default="data/reference/lga_adjacency.csv")
    parser.add_argument("--out", default="data/processed/ensemble_results.json")
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    panel = pd.read_parquet(args.panel).sort_values(["week", "pcode"]).reset_index(drop=True)
    features = [c for c in panel.columns if c not in NOT_FEATURES]
    weeks = np.sort(panel["week"].unique())
    pcodes = np.sort(panel["pcode"].unique())
    print(f"panel {len(panel):,} rows, {len(features)} features, {len(weeks)} weeks")

    # tensors for the graph network, shaped weeks by areas by features
    gx = panel[features].to_numpy(np.float32).reshape(len(weeks), len(pcodes), len(features))
    mean, std = gx.reshape(-1, len(features)).mean(0), gx.reshape(-1, len(features)).std(0) + 1e-6
    gx = torch.from_numpy((gx - mean) / std)
    gy = torch.from_numpy(panel["occurred"].to_numpy(np.float32).reshape(len(weeks), len(pcodes)))
    edges = pd.read_csv(args.adjacency)
    index = {p: i for i, p in enumerate(pcodes)}
    pairs = [(index[a], index[b]) for a, b in zip(edges.pcode_a, edges.pcode_b) if a in index and b in index]
    edge_index = torch.tensor(pairs + [(b, a) for a, b in pairs], dtype=torch.long).t().contiguous()

    BASE = ["logistic", "random_forest", "gradient_boosting", "stgnn"]
    results: dict[str, list[dict]] = {name: [] for name in
                                      BASE + ["ensemble", "ensemble_unweighted",
                                              "recency", "long_run"]}
    weights_log = []
    predictions = []

    bounds, end = [], len(weeks)
    for _ in range(args.folds):
        start = end - TEST_WEEKS
        if start - INNER_WEEKS <= 0:
            break
        bounds.append((start, end))
        end = start

    for fold, (test_start, test_end) in enumerate(reversed(bounds), 1):
        inner_start = test_start - INNER_WEEKS
        print(f"\nfold {fold}: base-fit to {pd.Timestamp(weeks[inner_start - 1]).date()}, "
              f"meta on {pd.Timestamp(weeks[inner_start]).date()} to "
              f"{pd.Timestamp(weeks[test_start - 1]).date()}, "
              f"test {pd.Timestamp(weeks[test_start]).date()} to {pd.Timestamp(weeks[test_end - 1]).date()}")

        inner_train = panel[panel.week.isin(weeks[:inner_start])]
        inner_val = panel[panel.week.isin(weeks[inner_start:test_start])].copy()
        full_train = panel[panel.week.isin(weeks[:test_start])]
        test = panel[panel.week.isin(weeks[test_start:test_end])].copy()

        meta_x, test_x = {}, {}
        for name in BASE:
            if name == "stgnn":
                meta_x[name] = fit_predict_gnn(gx, gy, edge_index, inner_start,
                                               range(inner_start, test_start), 7 + fold).reshape(-1)
                test_x[name] = fit_predict_gnn(gx, gy, edge_index, test_start,
                                               range(test_start, test_end), 7 + fold).reshape(-1)
            else:
                meta_x[name] = fit_predict_tabular(name, inner_train, inner_val, features)
                test_x[name] = fit_predict_tabular(name, full_train, test, features)
            test[name] = test_x[name]
            results[name].append({**evaluate(test, name), "fold": fold})
            print(f"    {name:<18} AP {results[name][-1]['average_precision']:.4f} | "
                  f"R@20 {results[name][-1]['recall_at_20']:.3f}")

        # Two meta-learners are fitted on the same inputs and differ only in whether
        # the loss is reweighted toward the positive class. This repeats at the stack
        # level the comparison stage one makes in `train_hurdle.py`, and for the same
        # reason: reweighting moves the predicted values away from the base rate by
        # construction, so a weighted meta-learner can rank well and still return a
        # number that cannot be read as a probability. The first run of this script
        # fitted only the weighted form and returned a mean Brier score of 0.194
        # against 0.032 for the calibrated logistic model, which would have left the
        # stack unable to supply the probability the interface reports.
        meta_features = np.column_stack([logit(meta_x[n]) for n in BASE])
        test_features = np.column_stack([logit(test_x[n]) for n in BASE])
        inner_y = inner_val["occurred"].to_numpy()
        for column, weighting in (("ensemble", "balanced"),
                                  ("ensemble_unweighted", None)):
            meta = LogisticRegression(max_iter=2000, class_weight=weighting)
            meta.fit(meta_features, inner_y)
            test[column] = meta.predict_proba(test_features)[:, 1]
            results[column].append({**evaluate(test, column), "fold": fold})
            print(f"    {column:<18} AP {results[column][-1]['average_precision']:.4f} | "
                  f"R@20 {results[column][-1]['recall_at_20']:.3f} | "
                  f"Brier {results[column][-1]['brier']:.4f}")
            weights_log.append({"fold": fold, "meta": column,
                                "intercept": float(meta.intercept_[0]),
                                **{n: float(w) for n, w in zip(BASE, meta.coef_[0])}})
            if column == "ensemble":
                print("    meta weights: " +
                      ", ".join(f"{n} {w:+.2f}" for n, w in zip(BASE, meta.coef_[0])))

        test["recency"] = test["own_events_4w"] + 0.01 * test["own_rate_longrun"]
        test["long_run"] = test["own_rate_longrun"]
        for name in ("recency", "long_run"):
            results[name].append({**evaluate(test, name), "fold": fold})
            print(f"    {name:<18} AP {results[name][-1]['average_precision']:.4f} | "
                  f"R@20 {results[name][-1]['recall_at_20']:.3f}")

        # Keep the scores so the meta stage can be re-examined without refitting the
        # base learners, which is the expensive part of this script.
        keep = ["pcode", "week", "occurred", "ensemble", "ensemble_unweighted",
                "recency", "long_run"] + BASE
        block = test[keep].copy()
        block["fold"] = fold
        predictions.append(block)

    print("\n" + "=" * 78)
    print(f"{'model':<22}{'mean AP':>10}{'mean R@20':>12}{'mean lift@20':>14}{'mean Brier':>14}")
    print("-" * 78)
    order = sorted(results, key=lambda n: -np.mean([r["average_precision"] for r in results[n]]))
    for name in order:
        rows = results[name]
        briers = [r["brier"] for r in rows if "brier" in r]
        print(f"{name:<22}{np.mean([r['average_precision'] for r in rows]):>10.4f}"
              f"{np.mean([r['recall_at_20'] for r in rows]):>12.3f}"
              f"{np.mean([r['lift_at_20'] for r in rows]):>14.1f}"
              f"{(f'{np.mean(briers):.4f}' if briers else 'n/a'):>14}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"results": results, "meta_weights": weights_log,
         "config": {"base": BASE, "test_weeks": TEST_WEEKS, "inner_weeks": INNER_WEEKS,
                    "gnn": {"hidden": GNN_HIDDEN, "K": GNN_K, "epochs": GNN_EPOCHS}}}, indent=1))
    print(f"\nwritten to {args.out}")
    if predictions:
        scores_path = Path(args.out).with_name("ensemble_test_scores.parquet")
        pd.concat(predictions, ignore_index=True).to_parquet(scores_path, index=False)
        print(f"test-period scores written to {scores_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
