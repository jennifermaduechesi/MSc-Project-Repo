"""Reproduce the fold-four calibration diagnostic quoted in the methodology chapter.

`CalibratedClassifierCV` given an integer `cv` runs K-fold without shuffling, so its
internal folds follow whatever order the rows arrive in. That turned the same nominal
model into two different models depending on which script called it. This fits one
penalised logistic specification five ways on fold four and reports average precision
on the test block, so the numbers written into the chapter can be reproduced rather
than remembered.

Expected output, and what the chapter states:

    no calibration                   AP 0.2623
    calibrated, rows by area         AP 0.2619
    calibrated, rows by week         AP 0.1827     <- the fault
    calibrated, TimeSeriesSplit      AP 0.2264
    calibrated, explicit by area     AP 0.2609     <- the fix now used everywhere
"""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

NOT_FEATURES = {"pcode", "week", "lga", "state", "event_count", "occurred"}
panel = pd.read_parquet("data/processed/panel.parquet")
features = [c for c in panel.columns if c not in NOT_FEATURES]
weeks = np.sort(panel["week"].unique())

# fold four: test block is weeks 551..603, training is everything before 551
train_end, test_end = 551, 603
train = panel[panel.week.isin(weeks[:train_end])]
test = panel[panel.week.isin(weeks[train_end:test_end])]
print(f"fold 4: train to {pd.Timestamp(weeks[train_end - 1]).date()}, "
      f"test {pd.Timestamp(weeks[train_end]).date()} to {pd.Timestamp(weeks[test_end - 1]).date()}")
print(f"train rows {len(train):,}  test rows {len(test):,}  "
      f"test positives {int(test.occurred.sum()):,}")

xt = test[features].to_numpy(np.float32)
yt = test["occurred"].to_numpy()


def base():
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))


def score(model):
    return average_precision_score(yt, model.predict_proba(xt)[:, 1])


for label, order, cv in (
    ("no calibration",                 ["pcode", "week"], None),
    ("calibrated, rows by area",       ["pcode", "week"], 3),
    ("calibrated, rows by week",       ["week", "pcode"], 3),
    ("calibrated, TimeSeriesSplit",    ["week", "pcode"], "ts"),
    ("calibrated, explicit by area",   ["week", "pcode"], "area"),
):
    frame = train.sort_values(order)
    x = frame[features].to_numpy(np.float32)
    y = frame["occurred"].to_numpy()
    if cv is None:
        model = base()
        model.fit(x, y)
    elif cv == "ts":
        model = CalibratedClassifierCV(base(), method="isotonic", cv=TimeSeriesSplit(3))
        model.fit(x, y)
    elif cv == "area":
        codes = pd.factorize(frame["pcode"].to_numpy())[0]
        idx = np.arange(len(codes))
        splits = [(idx[codes % 3 != f], idx[codes % 3 == f]) for f in range(3)]
        model = CalibratedClassifierCV(base(), method="isotonic", cv=splits)
        model.fit(x, y)
    else:
        model = CalibratedClassifierCV(base(), method="isotonic", cv=cv)
        model.fit(x, y)
    print(f"  {label:<32} AP {score(model):.4f}")
