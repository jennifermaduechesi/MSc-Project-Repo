"""Measure the collinear driver pair on the current panel and store the result.

Chapter Four states how often the cumulative event count and the long-run weekly rate
both reach an area's leading drivers with opposite signs. Those figures cannot be read
back out of `app/data/drivers.csv`, because that file ships the pair already grouped.
Without this script the chapter would be quoting a diagnostic nobody can reproduce,
which is how the earlier figures survived a panel rebuild that had moved all three of
them.

Writes `data/processed/collinearity.json` for `verify_chapter4.py` to check against.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROC = Path("data/processed")
PAIR = ("own_events_cum", "own_rate_longrun")
TOP = 6

panel = pd.read_parquet(PROC / "panel.parquet")
panel["week"] = pd.to_datetime(panel["week"])
non = {"pcode", "week", "event_count", "occurred", "lga", "state"}
features = [c for c in panel.columns if c not in non]
meta = json.load(open("app/data/meta.json"))
last = pd.Timestamp(meta["last_completed_week"])

# The same specification the interface's linear member uses, fitted on the same rows.
train = panel[panel["week"] <= last]
pipe = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=2000, class_weight="balanced"))
pipe.fit(train[features].to_numpy(np.float32), train["occurred"].to_numpy())
sc = pipe.named_steps["standardscaler"]
coef = pipe.named_steps["logisticregression"].coef_[0]

fx = panel[panel["week"] == last]
x = fx[features].to_numpy(np.float32)
contrib = (x - sc.mean_) / np.sqrt(sc.var_) * coef
idx = {f: i for i, f in enumerate(features)}
a, b = contrib[:, idx[PAIR[0]]], contrib[:, idx[PAIR[1]]]

top = np.argsort(-np.abs(contrib), axis=1)[:, :TOP]
both = np.array([(idx[PAIR[0]] in r) and (idx[PAIR[1]] in r) for r in top])
opposed = both & (np.sign(a) != np.sign(b)) & (a != 0) & (b != 0)

out = {
    "areas": int(len(fx)),
    "pair": list(PAIR),
    "correlation": float(panel[list(PAIR)].corr().iloc[0, 1]),
    "both_in_top_six": int(both.sum()),
    "both_in_top_six_opposite_signs": int(opposed.sum()),
    "mean_absolute_contribution": float(
        np.mean(np.abs(np.concatenate([a[opposed], b[opposed]])))),
    "mean_net_contribution": float(np.mean((a + b)[opposed])),
    "forecast_week": meta["forecast_week"],
}
(PROC / "collinearity.json").write_text(json.dumps(out, indent=1))
for k, v in out.items():
    print(f"  {k}: {v}")
