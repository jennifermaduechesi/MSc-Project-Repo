"""Re-derive every checkable figure in Chapter Four and compare it to the text.

Same principle as `verify_chapter3.py` and `verify_tables.py`: the stated side of every
check is read out of the chapter, never transcribed here, so updating the chapter cannot
quietly stop a check from testing anything.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

CHAPTER = Path("dissertation/chapter4_results.md")
PROC = Path("data/processed")
APP = Path("app/data")


def table(number: str) -> list[list[str]]:
    text = CHAPTER.read_text()
    block = text[text.index(f"\nTable {number}\n"):]
    rows, started = [], False
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append(cells); started = True
        elif started and not s:
            continue
        elif started:
            break
    return rows


def says(pattern: str, group: int = 1) -> str:
    m = re.search(pattern, CHAPTER.read_text())
    if not m:
        raise SystemExit(f"chapter text not found for: {pattern}")
    return m.group(group)


def num(s: str) -> float:
    # A figure pulled from prose often brings the sentence's full stop with it.
    return float(re.sub(r"[^0-9.\-]", "", s).rstrip("."))


PASS, FAIL = [], []
def check(label, stated, computed, tol=0.0):
    try:
        ok = abs(float(stated) - float(computed)) <= tol
    except (TypeError, ValueError):
        ok = str(stated) == str(computed)
    (PASS if ok else FAIL).append((label, stated, computed))


ens = json.load(open(PROC / "ensemble_results.json"))
hur = json.load(open(PROC / "hurdle_results.json"))
prot = json.load(open(PROC / "protocol_results.json"))
bands = json.load(open(PROC / "risk_bands.json"))
meta = json.load(open(APP / "meta.json"))
scores = pd.read_parquet(PROC / "ensemble_test_scores.parquet")
panel = pd.read_parquet(PROC / "panel.parquet")

def mean_of(group: str, key: str, metric: str) -> float:
    return float(np.mean([f[metric] for f in group[key] if f.get(metric) is not None]))


# ------------------------------------------------------- Table 4.1, the model comparison
KEY = {"Baseline: long-run rate": "long_run", "Baseline: recency": "recency",
       "Gradient boosting": "gradient_boosting", "Penalised logistic, calibrated": "logistic",
       "Recurrent graph network": "stgnn", "Random forest": "random_forest",
       "Stack, weighted meta": "ensemble", "Stack, unweighted meta": "ensemble_unweighted"}
COLS = [(1, "average_precision", 0.00005), (2, "ap_lift_over_base", 0.05),
        (3, "roc_auc", 0.0005), (4, "brier", 0.00005),
        (5, "recall_at_10", 0.0005), (6, "recall_at_20", 0.0005), (7, "recall_at_50", 0.0005)]
for row in table("4.1")[1:]:
    key = KEY[row[0]]
    for idx, metric, tol in COLS:
        if "applicable" in row[idx]:
            check(f"T4.1 {row[0]} {metric} absent", True,
                  all(f.get(metric) is None for f in ens["results"][key]))
            continue
        check(f"T4.1 {row[0]} {metric}", num(row[idx]), mean_of(ens["results"], key, metric), tol)

# ------------------------------------------------ Table 4.2, what calibration does
V = {"Penalised logistic, uncalibrated": "logistic",
     "Penalised logistic, calibrated": "logistic_calibrated",
     "Gradient boosting, uncalibrated": "gradient_boosting",
     "Gradient boosting, calibrated": "gradient_boosting_calibrated"}
for row in table("4.2")[1:]:
    k = V[row[0]]
    for idx, metric, tol in ((1, "average_precision", 0.00005), (2, "roc_auc", 0.0005),
                             (3, "brier", 0.00005), (4, "recall_at_20", 0.0005)):
        check(f"T4.2 {row[0]} {metric}", num(row[idx]), mean_of(hur["stage_one"], k, metric), tol)

# ------------------------------------------------------ Table 4.3, stage two vs constant
for row in table("4.3")[1:]:
    f = hur["stage_two"][int(row[0]) - 1]
    check(f"T4.3 fold {row[0]} positives", num(row[1]), f["n_positive"])
    check(f"T4.3 fold {row[0]} mean actual", num(row[2]), f["mean_actual"], 0.0005)
    check(f"T4.3 fold {row[0]} mean predicted", num(row[3]), f["mean_predicted"], 0.0005)
    check(f"T4.3 fold {row[0]} mae fitted", num(row[4]), f["mae"], 0.0005)
    check(f"T4.3 fold {row[0]} mae always one", num(row[5]), f["mae_always_one"], 0.0005)
check("T4.3 fitted stage two loses in every fold", 5,
      sum(1 for f in hur["stage_two"] if f["mae"] > f["mae_always_one"]))

# ------------------------------------------------------- Table 4.4, meta-learner weights
w = {x["fold"]: x for x in ens["meta_weights"] if x["meta"] == "ensemble_unweighted"}
for row in table("4.4")[1:]:
    f = w[int(row[0])]
    for idx, key in ((1, "logistic"), (2, "random_forest"), (3, "gradient_boosting"),
                     (4, "stgnn"), (5, "intercept")):
        check(f"T4.4 fold {row[0]} {key}", num(row[idx]), f[key], 0.0005)

# --------------------------------------------------------- Table 4.5, geography by state
s = scores.merge(panel[["pcode", "state"]].drop_duplicates(), on="pcode", how="left")
st = s.groupby("state").agg(events=("occurred", "sum"), observed=("occurred", "mean"),
                            predicted=("ensemble_unweighted", "mean"))
for row in table("4.5")[1:]:
    g = st.loc[row[0]]
    check(f"T4.5 {row[0]} events", num(row[1]), g["events"])
    check(f"T4.5 {row[0]} observed", num(row[2]), g["observed"], 0.00005)
    check(f"T4.5 {row[0]} predicted", num(row[3]), g["predicted"], 0.00005)
    check(f"T4.5 {row[0]} ratio", num(row[4]), g["predicted"] / g["observed"], 0.005)

# ------------------------------------------------------------------- Table 4.6, drivers
dr = pd.read_csv(APP / "drivers.csv")
counts = dr["label"].value_counts()
for row in table("4.6")[1:]:
    label = row[0][0].lower() + row[0][1:]
    check(f"T4.6 {row[0][:44]}", num(row[1]), counts.get(label, 0))

# --------------------------------------------------- Table 4.7, the worked example
fc = pd.read_csv(APP / "forecast.csv")
sb = fc[fc["lga"] == "Sabon Birni"]
pcode = sb["pcode"].iloc[0]
t22 = table("4.7")
for row in t22[1:4]:
    h = int(num(row[0]))
    r = sb[sb["horizon_days"] == h].iloc[0]
    check(f"T4.7 {h}d probability", num(row[1]), 100 * r["probability"], 0.05)
    check(f"T4.7 {h}d band", row[2], r["band"])
    check(f"T4.7 {h}d rank", num(row[3]), r["rank"])
d7 = dr[(dr["pcode"] == pcode) & (dr["horizon_days"] == 7)].set_index("label")
for row in table("4.8")[1:]:
    label = row[0][0].lower() + row[0][1:]
    if label not in d7.index:
        FAIL.append((f"T4.8 driver {row[0][:40]}", "listed", "not in drivers.csv")); continue
    check(f"T4.8 driver {row[0][:40]} contribution", num(row[2]),
          d7.loc[label, "contribution"], 0.005)
    if row[1] != "grouped":
        check(f"T4.8 driver {row[0][:40]} value", num(row[1]), d7.loc[label, "value"], 0.0005)

# ------------------------------------------------------------------ prose figures
check("4.2 held-out area-weeks", num(says(r"on ([\d,]+) held-out area-weeks")), len(scores))
check("4.2 test weeks", num(says(r"across (\d+) test weeks")), scores["week"].nunique())
check("4.2 base rate", num(says(r"base rate across the five test periods is ([\d.]+)")),
      scores["occurred"].mean(), 0.00005)

def caught(col, k=20):
    hits = tot = 0
    for _, wk in scores.groupby("week"):
        hits += wk.nlargest(k, col)["occurred"].sum(); tot += wk["occurred"].sum()
    return int(hits), int(tot)
a, tot = caught("ensemble_unweighted")
b, _ = caught("logistic")
c, _ = caught("random_forest")
r, _ = caught("recency")
check("4.5 stack catches", num(says(r"identifies ([\d,]+) of the [\d,]+ attacked")), a)
check("4.5 total attacked", num(says(r"identifies [\d,]+ of the ([\d,]+) attacked")), tot)
check("4.5 logistic catches", num(says(r"against ([\d,]+) for the calibrated logistic model and")), b)
check("4.5 forest catches", num(says(r"and ([\d,]+) for the random forest")), c)
check("4.5 stack over logistic", num(says(r"finds (\d+) more attacked areas")), a - b)
check("4.5 recency catches", num(says(r"The recency list finds ([\d,]+)")), r)
# "The stack finds N more" appears twice; anchor on the clause that names the week.
check("4.5 stack over recency",
      num(says(r"The stack finds (\d+) more, which is one every other week")), a - r)

folds = sorted(scores["fold"].unique())
per = []
for f in folds:
    g = scores[scores["fold"] == f]; hits = tot_f = 0
    for _, wk in g.groupby("week"):
        hits += wk.nlargest(20, "ensemble_unweighted")["occurred"].sum()
        tot_f += wk["occurred"].sum()
    per.append(hits / tot_f)
stated = [float(x) for x in says(
    r"Recall at the top twenty runs ([\d., and]+) across folds one to five"
).replace(" and", ",").replace(" ", "").strip(",").split(",") if x]
check("4.8 per-fold recall at 20", stated, [round(float(v), 3) for v in per])

ap = [f["average_precision"] for f in ens["results"]["ensemble_unweighted"]]
# Anchored on the following word rather than a full stop: the sentence was merged
# during the trim, and "([\d.]+)\." then matched just the leading zero.
check("4.2 stack AP fold range", num(says(r"a range of ([\d.]+), while")),
      max(ap) - min(ap), 0.0005)
f5 = {k: v[4]["average_precision"] for k, v in ens["results"].items()}
check("4.2 model range within fold five",
      num(says(r"range between models within fold five is ([\d.]+)")),
      max(f5.values()) - min(f5.values()), 0.0005)

check("4.7 pooled detection", num(says(r"pooled figure of ([\d.]+)")),
      prot["under_reporting"]["pooled_detection"], 0.0005)
check("4.7 reliable states", num(says(r"the (\d+) states where the estimate is reliable")),
      prot["under_reporting"]["states_with_own_estimate"])
det = prot["detection"]["pm0d"]
rel = [x for x in det if x["reliable"]]
check("4.7 share of records in reliable states",
      num(says(r"hold ([\d.]+) per cent of all records")),
      100 * sum(x["observed"] for x in rel) / sum(x["observed"] for x in det), 0.05)
check("4.7 weight min", num(says(r"weights ranged from ([\d.]+) to")),
      prot["under_reporting"]["weight_min"], 0.0005)
check("4.7 weight max", num(says(r"weights ranged from [\d.]+ to ([\d.]+)")),
      prot["under_reporting"]["weight_max"], 0.0005)

check("4.8 five states share", num(says(r"hold ([\d.]+) per cent of all events recorded")),
      100 * st.nlargest(5, "events")["events"].sum() / st["events"].sum(), 0.05)
check("4.8 ten states share", num(says(r"ten largest hold ([\d.]+) per cent")),
      100 * st.nlargest(10, "events")["events"].sum() / st["events"].sum(), 0.05)
st2 = st.assign(ratio=st["predicted"] / st["observed"])
check("4.8 states outside the band", num(says(r"([\w]+) fall outside a band of 0.75 to 1.33")
      .replace("ten", "10")), int(((st2["ratio"] < 0.75) | (st2["ratio"] > 1.33)).sum()))

check("4.10 Severe count", num(says(r"places (\d+) areas in Severe")), meta["7"]["band_counts"]["Severe"])
check("4.10 High count", num(says(r"(\d+) in High")), meta["7"]["band_counts"]["High"])
check("4.10 Elevated count", num(says(r"(\d+) in Elevated")), meta["7"]["band_counts"]["Elevated"])
check("4.10 Low count", num(says(r"leaving (\d+) in Low")), meta["7"]["band_counts"]["Low"])
check("4.10 anchor rate", num(says(r"eight times a base rate of ([\d.]+)")),
      meta["7"]["anchor_rate"], 0.00005)
check("4.10 anchor weeks", num(says(r"over the (\w+) completed weeks").replace("eighteen", "18")),
      meta["7"]["anchor_weeks"])
check("4.10 low share of events", num(says(r"([\d.]+) per cent of all recorded events fall")),
      100 * bands["low_share_pooled"], 0.05)

ri = pd.read_csv(APP / "recent_incidents.csv")
sbi = ri[ri["pcode"] == pcode]
check("4.10 recent incidents", num(says(r"had (\w+) recorded incidents").replace("twelve", "12")),
      len(sbi))
own = dr[(dr["horizon_days"] == 7) & dr["feature"].str.match(r"own_(events|active)_\d+w", na=False)]
g = own.groupby("pcode")["contribution"].agg(["min", "max"])
check("4.9 sign-disagreement areas",
      num(says(r"signs disagree in only (\w+) of the 774").replace("sixteen", "16")),
      int(((g["min"] < 0) & (g["max"] > 0)).sum()))

# ------------------------------- claims a first pass of this checker did not cover.
# Every one of these was written into the chapter without anything re-deriving it, and
# four of them were wrong. They are pinned here so that cannot happen silently again.
check("4.2 quiet share of test rows",
      num(says(r"order the (\d+) per cent of quiet area-weeks")),
      round(100 * (1 - scores["occurred"].mean())))
check("4.2 recency share of attacked areas",
      num(says(r"it puts ([\d.]+) per cent of the week's attacked")),
      100 * mean_of(ens["results"], "recency", "recall_at_20"), 0.05)
check("4.2 recency share restated as a percentage",
      num(says(r"it puts ([\d.]+) per cent of the week's attacked")),
      100 * mean_of(ens["results"], "recency", "recall_at_20"), 0.05)

# The graph network against the linear member, fold by fold. The first draft said it
# lost folds four and five; it loses only fold four and wins fold five by its widest
# margin, which changes what the section is entitled to claim.
gm = [f["average_precision"] for f in ens["results"]["stgnn"]]
lm = [f["average_precision"] for f in ens["results"]["logistic"]]
wins = [round(g - l, 3) for g, l in zip(gm, lm) if g > l]
losses = [round(l - g, 3) for g, l in zip(gm, lm) if g <= l]
stated_wins = [float(x) for x in re.findall(
    r"[\d.]+", says(r"in four folds of the five, by ([\d., and]+?), and loses"))]
check("4.4 graph network fold wins", sorted(stated_wins), sorted(wins))
check("4.4 graph network fold count", num(says(r"beats the logistic model .*? in (\w+) folds")
      .replace("four", "4")), len(wins))
check("4.4 graph network loss margin",
      num(says(r"loses only in fold four, by ([\d.]+)")), losses[0], 0.0005)
check("4.4 aggregate margin", num(says(r"aggregate margin of ([\d.]+)")),
      mean_of(ens["results"], "stgnn", "average_precision")
      - mean_of(ens["results"], "logistic", "average_precision"), 0.0005)

check("4.5 meta largest weight count",
      num(says(r"largest weight of any member in (\w+) of the five").replace("three", "3")),
      sum(1 for f in w.values()
          if max(("logistic", "random_forest", "gradient_boosting", "stgnn"),
                 key=lambda m: f[m]) == "stgnn"))

check("4.8 Zamfara share of area-weeks",
      num(says(r"from ([\d.]+) per cent of the area-weeks")),
      100 * len(s[s["state"] == "Zamfara"]) / len(s), 0.05)
check("4.8 Zamfara event rate", num(says(r"at an event rate of ([\d.]+) against a national")),
      st.loc["Zamfara", "observed"], 0.0005)
check("4.8 Zamfara share of events", num(says(r"Zamfara alone carries ([\d.]+) per cent")),
      100 * st.loc["Zamfara", "events"] / st["events"].sum(), 0.05)
check("4.8 Niger over-prediction", num(says(r"the model predicts (\d+) per cent too much")),
      round(100 * (st.loc["Niger", "predicted"] / st.loc["Niger", "observed"] - 1)))
check("4.8 FCT under-prediction", num(says(r"it predicts (\d+) per cent too little")),
      round(100 * (1 - st.loc["Federal Capital Territory", "predicted"]
                   / st.loc["Federal Capital Territory", "observed"])))

col = json.load(open(PROC / "collinearity.json"))
check("4.9 collinear pair correlation", num(says(r"for an area correlate at ([\d.]+)")),
      col["correlation"], 0.005)
check("4.9 areas with the pair opposed",
      num(says(r"in (\d+) of the 774 areas both appear")),
      col["both_in_top_six_opposite_signs"])
check("4.9 mean magnitude of the pair", num(says(r"at magnitudes averaging ([\d.]+)")),
      col["mean_absolute_contribution"], 0.005)
check("4.9 net of the pair", num(says(r"netting to about -([\d.]+)")),
      abs(col["mean_net_contribution"]), 0.005)

hz = prot["horizon"]
for window, word in ((7, "next week's"), (14, "next fortnight's")):
    check(f"4.6 recall per cent at {window}d",
          num(says(rf"(\d+) per cent of the {re.escape(word)}")),
          round(100 * np.mean([f["recall_at_20"] for f in hz[f"{window}d"]])))
bs = pd.read_csv(PROC / "band_stats.csv")
for days, pat in ((28, r"falling to ([\d.]+) areas per week"),
                  (7, r"areas per week against ([\d.]+) at seven days")):
    row = bs[(bs.horizon_days == days) & (bs.band == "Severe")].iloc[0]
    check(f"4.6 Severe areas per week at {days}d", num(says(pat)), row["areas_per_week"], 0.05)

dl = prot["delay"]
base_ap = np.mean([f["average_precision"] for f in dl["0w"]])
base_r = np.mean([f["recall_at_20"] for f in dl["0w"]])
check("4.7 one-week AP cost", num(says(r"records costs ([\d.]+) of average precision")),
      base_ap - np.mean([f["average_precision"] for f in dl["1w"]]), 0.0005)
check("4.7 one-week recall cost",
      num(says(r"from 0\.183 to 0\.172, and ([\d.]+) of recall at twenty")),
      base_r - np.mean([f["recall_at_20"] for f in dl["1w"]]), 0.0005)
check("4.7 two-week costs",
      [float(x.rstrip(".")) for x in
       re.findall(r"[\d.]+", says(r"two weeks costs ([\d.]+ and [\d.]+)"))],
      [round(float(base_ap - np.mean([f["average_precision"] for f in dl["2w"]])), 3),
       round(float(base_r - np.mean([f["recall_at_20"] for f in dl["2w"]])), 3)])

check("4.2 calibration AP cost", num(says(r"costs the logistic model ([\d.]+) of average")),
      mean_of(hur["stage_one"], "logistic", "average_precision")
      - mean_of(hur["stage_one"], "logistic_calibrated", "average_precision"), 0.0005)
best_ap = max(mean_of(ens["results"], m, "average_precision")
              for m in ("logistic", "random_forest", "gradient_boosting", "stgnn"))
best_r = max(mean_of(ens["results"], m, "recall_at_20")
             for m in ("logistic", "random_forest", "gradient_boosting", "stgnn"))
check("4.11 best member on AP is the random forest", "random_forest",
      max(("logistic", "random_forest", "gradient_boosting", "stgnn"),
          key=lambda m: mean_of(ens["results"], m, "average_precision")))
check("4.11 best member on recall is the logistic", "logistic",
      max(("logistic", "random_forest", "gradient_boosting", "stgnn"),
          key=lambda m: mean_of(ens["results"], m, "recall_at_20")))

# Three claims Chapter Five's checker caught in this chapter, pinned here so they cannot
# recur: the per-fold list quoted in prose, the graph network's Brier quoted to three
# places rather than four, and the two lift figures, which are different quantities and
# had been conflated ("5.4 times better than chance at the top of the list": 5.4 is
# average precision over the base rate, and the top-of-list figure is 7.6).
stated_folds = [float(x) for x in re.findall(
    r"[\d.]+", says(r"Average precision for the stack runs ([\d., ]+?) across the five folds"))]
check("4.2 per-fold AP quoted in prose", stated_folds,
      [round(f["average_precision"], 3) for f in ens["results"]["ensemble_unweighted"]])
check("4.2 graph network Brier in prose",
      num(says(r"the graph network at ([\d.]+), the weighted stack")),
      mean_of(ens["results"], "stgnn", "brier"), 0.0005)
check("4.4 graph network Brier restated",
      num(says(r"Its Brier score of ([\d.]+) is the worst")),
      mean_of(ens["results"], "stgnn", "brier"), 0.0005)
check("4.11 fold AP maximum restated",
      num(says(r"ranges from [\d.]+ to ([\d.]+) across the five folds")),
      max(f["average_precision"] for f in ens["results"]["ensemble_unweighted"]), 0.0005)

w_ = max(len(l) for l, _, _ in PASS + FAIL) + 2
for l, s_, c_ in PASS: print(f"  [pass] {l:<{w_}} stated {str(s_)[:30]:>32}  computed {str(c_)[:30]:>32}")
for l, s_, c_ in FAIL: print(f"  [FAIL] {l:<{w_}} stated {str(s_)[:30]:>32}  computed {str(c_)[:30]:>32}")
print("=" * 118)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(PASS)+len(FAIL)} checks")
print("=" * 118)
sys.exit(1 if FAIL else 0)
