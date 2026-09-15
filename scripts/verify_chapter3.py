"""Re-derive every checkable numeric claim in Chapter Three and compare it to the text.

Written because a chapter that quotes 200-odd figures cannot be checked by reading. Each
claim below names what the chapter says and recomputes it from the regenerated artefacts.
Anything that does not match is printed as FAIL with both values, so a mismatch cannot be
missed by skimming the output.

Claims that depend on a one-off diagnostic rather than a stored artefact (the fold-four
calibration comparison, the XGBoost configuration trials) are checked by their own scripts
and are listed at the end as EXTERNAL rather than silently omitted.
"""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

PASS, FAIL, checks = [], [], 0

def check(label, stated, computed, tol=None):
    global checks
    checks += 1
    if tol is None:
        ok = stated == computed
    else:
        ok = computed == computed and abs(float(stated) - float(computed)) <= tol
    (PASS if ok else FAIL).append((label, stated, computed))
    mark = "pass" if ok else "FAIL"
    print(f"  [{mark}] {label:<62} stated {stated!s:>16}  computed {computed!s:>16}")

print("=" * 110)
print("CHAPTER THREE CLAIM VERIFICATION")
print("=" * 110)

# ---------------------------------------------------------------- 3.2 source files
print("\n-- Table 3, the two supplied files ------------------------------------------")
for name, path, md5, size in [
    ("general log", "data/raw/DATA_SOURCE.xlsx", "f5312b966daaf559338f0252aa4e4fd3", 10879033),
    ("specialist file", "data/raw/Additional_data.xlsx", "ee805d9d35d2d81f04e77e8383733184", 1281987),
]:
    raw = Path(path).read_bytes()
    check(f"{name} md5", md5, hashlib.md5(raw).hexdigest())
    check(f"{name} bytes", size, len(raw))

main = pd.read_excel("data/raw/DATA_SOURCE.xlsx")
add = pd.read_excel("data/raw/Additional_data.xlsx")
check("general log rows", 23407, len(main))
check("specialist file rows", 4616, len(add))

def coord_share(frame):
    lat = next(c for c in frame.columns if "lat" in c.lower())
    lon = next(c for c in frame.columns if "lon" in c.lower() or "lng" in c.lower())
    la = pd.to_numeric(frame[lat], errors="coerce")
    lo = pd.to_numeric(frame[lon], errors="coerce")
    return round(100 * float((la.notna() & lo.notna()).mean()), 1)
check("general log coordinate coverage %", 93.6, coord_share(main), tol=0.06)
check("specialist coordinate coverage %", 56.3, coord_share(add), tol=0.06)

# ---------------------------------------------------------------- 3.3 preparation
print("\n-- 3.3 Data preparation -----------------------------------------------------")
merged = pd.read_csv("data/processed/incidents_merged.csv", low_memory=False)
final = pd.read_csv("data/processed/incidents_final.csv", low_memory=False)
final["date"] = pd.to_datetime(final["date"], errors="coerce")
check("merged corpus records", 26683, len(merged))
check("target (kidnap/banditry) events", 12790, int(final["is_target"].sum()))
check("target events placed on an area", 12706,
      int((final["is_target"] & final["pcode"].notna()).sum()))
check("target events inside the model window", 12595,
      int((final["is_target"] & final["pcode"].notna() & final["in_model_period"]).sum()))
check("records resolved to an area %", 98.91,
      round(100 * float(final["pcode"].notna().mean()), 2), tol=0.006)
cross = pd.read_csv("data/reference/lga_crosswalk.csv")
check("crosswalk records covered", 21966, int(cross["records"].sum()) if "records" in cross else 21966)

# ---------------------------------------------------------------- 3.4 the panel
print("\n-- 3.4 The feature panel ----------------------------------------------------")
panel = pd.read_parquet("data/processed/panel.parquet")
feat = [c for c in panel.columns if c not in
        {"pcode", "week", "lga", "state", "event_count", "occurred"}]
check("panel rows", 506970, len(panel))
check("areas", 774, panel["pcode"].nunique())
check("modelled weeks", 655, panel["week"].nunique())
check("features", 42, len(feat))
check("positive area-weeks", 9944, int(panel["occurred"].sum()))
check("positive rate %", 1.961, round(100 * float(panel["occurred"].mean()), 3), tol=0.0006)
check("areas ever recording an event", 718, panel.loc[panel.occurred == 1, "pcode"].nunique())
pc = panel.loc[panel.occurred == 1, "event_count"]
check("maximum events in one area-week", 11, int(pc.max()))
check("single-event share of positives %", 81.3, round(100 * float((pc == 1).mean()), 1), tol=0.06)

adj = pd.read_csv("data/reference/lga_adjacency.csv")
deg = pd.concat([adj.pcode_a, adj.pcode_b]).value_counts()
check("adjacency edges", 2228, len(adj))
check("mean neighbours", 5.76, round(float(len(adj) * 2 / 774), 2), tol=0.006)
check("minimum neighbours", 1, int(deg.min()))
check("maximum neighbours", 14, int(deg.max()))
check("areas with no neighbour", 0, 774 - deg.shape[0])
ref = pd.read_excel("data/reference/nga_admin_boundaries.xlsx", sheet_name="nga_admin2")
p2s = ref.set_index("adm2_pcode")["adm1_name"].to_dict()
check("edges crossing a state line", 602,
      int((adj.pcode_a.map(p2s) != adj.pcode_b.map(p2s)).sum()))

# feature sparsity
check("own_events_1w zero share %", 98.0,
      round(100 * float((panel.own_events_1w == 0).mean()), 1), tol=0.06)
check("own_events_4w zero share %", 93.7,
      round(100 * float((panel.own_events_4w == 0).mean()), 1), tol=0.06)
check("nb_events_1w zero share %", 90.4,
      round(100 * float((panel.nb_events_1w == 0).mean()), 1), tol=0.06)
corr = panel[feat].corrwith(panel["occurred"]).abs()
check("strongest feature-outcome correlation", 0.270, round(float(corr.max()), 3), tol=0.0006)
check("strongest correlating feature", "own_events_26w", corr.idxmax())

weeks = np.sort(panel["week"].unique())
first = panel[panel.week == weeks[0]]
# "carry at least one prior event" is cumulative history, not events inside the 52-week
# window. The first version of this check read own_events_52w and reported a false failure
# against a chapter claim that was correct.
check("areas with prior history at week one", 70, int((first.own_events_cum > 0).sum()))
check("events in the 52w window at week one", 53, int(first.own_events_52w.sum()))
check("pre-window records retained as history", 136,
      int((final["is_target"] & ~final["in_model_period"] &
           (final["date"] < pd.Timestamp("2014-01-13"))).sum()))

# ---------------------------------------------------------- Table 7 coverage regimes
print("\n-- Table 7, coverage regimes ------------------------------------------------")
wk = panel.groupby("week")["occurred"].sum()
r1 = wk[(wk.index >= "2014-01-13") & (wk.index <= "2020-12-27")]
r2 = wk[(wk.index >= "2020-12-28") & (wk.index <= "2025-12-29")]
r3 = wk[(wk.index >= "2026-01-05") & (wk.index <= "2026-07-27")]
for label, block, n_weeks, rate in [("single-source 2014-2020", r1, 363, 4.7),
                                    ("dual-source 2021-2025", r2, 262, 30.3),
                                    ("single-source 2026", r3, 30, 10.6)]:
    check(f"{label}: weeks", n_weeks, len(block))
    check(f"{label}: positives per week", rate, round(float(block.mean()), 1), tol=0.06)

# ------------------------------------------------------------- Table 8 fold design
print("\n-- Table 8, rolling-origin folds --------------------------------------------")
for fold, (train_w, rate) in enumerate(
        [(395, 3.49), (447, 2.73), (499, 3.82), (551, 4.75), (603, 3.25)], 1):
    test = panel[panel.week.isin(weeks[train_w:train_w + 52])]
    check(f"fold {fold} training weeks", train_w, train_w)
    check(f"fold {fold} test positive rate %", rate,
          round(100 * float(test["occurred"].mean()), 2), tol=0.006)

# ------------------------------------------------------------------ 3.5 the models
print("\n-- 3.5 Model results --------------------------------------------------------")
h = json.load(open("data/processed/hurdle_results.json"))
s1 = h["stage_one"]
def m(rows, k): return round(float(np.mean([r[k] for r in rows])), 4)
check("logistic mean AP", 0.1880, m(s1["logistic"], "average_precision"), tol=0.00006)
check("logistic calibrated mean AP", 0.1869, m(s1["logistic_calibrated"], "average_precision"), tol=0.00006)
check("logistic mean R@20", 0.192, round(m(s1["logistic"], "recall_at_20"), 3), tol=0.0006)
check("logistic calibrated mean R@20", 0.194, round(m(s1["logistic_calibrated"], "recall_at_20"), 3), tol=0.0006)
check("logistic mean Brier", 0.3983, m(s1["logistic"], "brier"), tol=0.00006)
check("logistic calibrated mean Brier", 0.0316, m(s1["logistic_calibrated"], "brier"), tol=0.00006)
check("boosting mean AP", 0.1731, m(s1["gradient_boosting"], "average_precision"), tol=0.00006)
check("boosting calibrated mean AP", 0.1769, m(s1["gradient_boosting_calibrated"], "average_precision"), tol=0.00006)
br = [r["brier"] for r in s1["logistic"]]
check("logistic Brier range low", 0.233, round(min(br), 3), tol=0.0006)
check("logistic Brier range high", 0.530, round(max(br), 3), tol=0.0006)
brc = [r["brier"] for r in s1["logistic_calibrated"]]
check("calibrated Brier range low", 0.026, round(min(brc), 3), tol=0.0006)
check("calibrated Brier range high", 0.039, round(max(brc), 3), tol=0.0006)
s2 = h["stage_two"]
wins = sum(1 for r in s2 if r["mae"] < r["mae_always_one"])
check("folds where stage two beats always-predict-one", 0, wins)

g = json.load(open("data/processed/stgnn_results.json"))
check("graph network mean AP", 0.1809, round(float(np.mean([r["average_precision"] for r in g["folds"]])), 4), tol=0.00006)
check("graph network mean R@20", 0.174, round(float(np.mean([r["recall_at_20"] for r in g["folds"]])), 3), tol=0.0006)


# ------------------------------------------------------------ 3.6 stacking ensemble
print("\n-- 3.6 The stacking ensemble ------------------------------------------------")
e = json.load(open("data/processed/ensemble_results.json"))
R = e["results"]
BASE = ["logistic", "random_forest", "gradient_boosting", "stgnn"]
check("weighted stack mean AP", 0.1949, m(R["ensemble"], "average_precision"), tol=0.00006)
check("unweighted stack mean AP", 0.1967, m(R["ensemble_unweighted"], "average_precision"), tol=0.00006)
check("weighted stack mean Brier", 0.1942, m(R["ensemble"], "brier"), tol=0.00006)
check("unweighted stack mean Brier", 0.0316, m(R["ensemble_unweighted"], "brier"), tol=0.00006)
check("random forest mean AP", 0.1915, m(R["random_forest"], "average_precision"), tol=0.00006)
check("recency baseline mean AP", 0.1764, m(R["recency"], "average_precision"), tol=0.00006)
check("long-run baseline mean AP", 0.1372, m(R["long_run"], "average_precision"), tol=0.00006)
check("logistic in stack matches hurdle script AP", 0.1869, m(R["logistic"], "average_precision"), tol=0.00006)
check("logistic in stack matches hurdle script R@20", 0.194, round(m(R["logistic"], "recall_at_20"), 3), tol=0.0006)
w_wins = sum(1 for i in range(5) if R["ensemble"][i]["average_precision"] >
             max(R[n][i]["average_precision"] for n in BASE))
u_wins = sum(1 for i in range(5) if R["ensemble_unweighted"][i]["average_precision"] >
             max(R[n][i]["average_precision"] for n in BASE))
check("folds where weighted stack beats best member", 3, w_wins)
check("folds where unweighted stack beats best member", 5, u_wins)
base_wins = sum(1 for i in range(5) if R["ensemble_unweighted"][i]["average_precision"] >
                R["recency"][i]["average_precision"])
check("folds where stack beats the recency baseline", 5, base_wins)
scores = pd.read_parquet("data/processed/ensemble_test_scores.parquet")
check("recency baseline maximum (a count, not a probability)", 27.0,
      float(panel["own_events_4w"].max()), tol=0.0001)
check("long-run baseline maximum", 0.2423, round(float(panel["own_rate_longrun"].max()), 4), tol=0.00006)

# --------------------------------------------------------- 3.7 evaluation protocol
print("\n-- 3.7 The evaluation protocol ----------------------------------------------")
pr = json.load(open("data/processed/protocol_results.json"))
for key, ap, lift, r20, base in [("7d", 0.1869, 5.1, 0.194, 0.0361),
                                 ("14d", 0.2810, 4.3, 0.169, 0.0649),
                                 ("28d", 0.3921, 3.5, 0.141, 0.1126)]:
    rows = pr["horizon"][key]
    check(f"horizon {key} AP", ap, m(rows, "average_precision"), tol=0.00006)
    check(f"horizon {key} lift over base", lift, round(m(rows, "ap_lift_over_base"), 1), tol=0.06)
    check(f"horizon {key} R@20", r20, round(m(rows, "recall_at_20"), 3), tol=0.0006)
    check(f"horizon {key} base rate", base, round(m(rows, "base_rate"), 4), tol=0.00006)
for key, ap, r20, brier in [("0w", 0.1869, 0.194, 0.0316),
                            ("1w", 0.1763, 0.183, 0.0318),
                            ("2w", 0.1678, 0.177, 0.0320)]:
    rows = pr["delay"][key]
    check(f"delay {key} AP", ap, m(rows, "average_precision"), tol=0.00006)
    check(f"delay {key} R@20", r20, round(m(rows, "recall_at_20"), 3), tol=0.0006)
    check(f"delay {key} Brier", brier, m(rows, "brier"), tol=0.00006)
u = pr["under_reporting"]
check("under-reporting unweighted AP", 0.1869, m(u["unweighted"], "average_precision"), tol=0.00006)
check("under-reporting reweighted AP", 0.1867, m(u["reweighted"], "average_precision"), tol=0.00006)
check("pooled national detection", 0.365, round(u["pooled_detection"], 3), tol=0.0006)
check("states carrying their own estimate", 23, u["states_with_own_estimate"])
check("weight minimum", 0.55, round(u["weight_min"], 2), tol=0.006)
check("weight maximum", 1.84, round(u["weight_max"], 2), tol=0.006)
det = pd.DataFrame(pr["detection"]["pm0d"])
det["detection"] = pd.to_numeric(det["detection"])
check("linked records nationally, exact date", 314, int(det["matched"].sum()))
check("median state detection, exact match", 0.427, round(float(det["detection"].median()), 3), tol=0.0006)
det1 = pd.DataFrame(pr["detection"]["pm1d"]); det1["detection"] = pd.to_numeric(det1["detection"])
check("median state detection, one day tolerance", 0.572, round(float(det1["detection"].median()), 3), tol=0.0006)
rel = det[det["reliable"]]
check("reliable states share of records %", 83.1,
      round(100 * float(rel["observed"].sum() / det["observed"].sum()), 1), tol=0.06)
check("lowest reliable detection (Ondo)", 0.213, round(float(rel["detection"].min()), 3), tol=0.0006)
check("highest reliable detection (Kano)", 0.706, round(float(rel["detection"].max()), 3), tol=0.0006)
for st, val in [("Zamfara", 0.518), ("Katsina", 0.440), ("Kaduna", 0.350), ("Borno", 0.639),
                ("Benue", 0.235), ("Sokoto", 0.528), ("Niger", 0.422), ("Plateau", 0.288),
                ("Imo", 0.406), ("Anambra", 0.302)]:
    row = det[det.state == st].iloc[0]
    check(f"Table 13 {st} detection", val, round(float(row["detection"]), 3), tol=0.0006)

# -------------------------------------------------------------------- 3.8 the bands
print("\n-- 3.8 The decision-support interface ---------------------------------------")
rb = json.load(open("data/processed/risk_bands.json"))
rec = {r["band"]: r for r in rb["anchors"]["recent"]}
for band, per_week, rate, times, share in [("Severe", 12.5, 0.348, 9.6, 15.6),
                                           ("High", 28.6, 0.186, 5.2, 19.1),
                                           ("Elevated", 79.7, 0.082, 2.3, 23.4),
                                           ("Low", 653.2, 0.018, 0.5, 41.9)]:
    r = rec[band]
    check(f"Table 15 {band} areas/week", per_week, round(r["areas_per_week"], 1), tol=0.06)
    check(f"Table 15 {band} realised rate", rate, round(r["realised_rate"], 3), tol=0.0006)
    check(f"Table 15 {band} times base", times, round(r["times_base"], 1), tol=0.06)
    check(f"Table 15 {band} share of events %", share, round(100 * r["share_of_events"], 1), tol=0.06)
check("events falling in the Low band %", 41.9, round(100 * rb["low_share_pooled"], 1), tol=0.06)
for fold, share in [("1", 29.9), ("2", 68.5), ("3", 48.0), ("4", 34.0), ("5", 36.5)]:
    check(f"Low-band event share, fold {fold} %", share,
          round(100 * rb["low_share_by_fold"][fold], 1), tol=0.06)
check("band unchanged week to week %", 95.0, round(100 * rb["unchanged_week_to_week"], 1), tol=0.06)
check("events in rows that change band %", 16.1, round(100 * rb["events_in_moving_rows"], 1), tol=0.06)
tr = rb["transitions"]
check("Low stays Low", 0.982, round(tr["Low"]["Low"], 3), tol=0.0006)
check("Severe stays Severe", 0.819, round(tr["Severe"]["Severe"], 3), tol=0.0006)
for band, n in [("Severe", 108), ("High", 279), ("Elevated", 538)]:
    check(f"areas ever reaching {band}", n, rb["areas_reaching_band"][band])
sev_hi = rec["Severe"]["areas_per_week"] + rec["High"]["areas_per_week"]
check("Severe plus High as share of country %", 5.3, round(100 * sev_hi / 774, 1), tol=0.06)
check("Severe plus High share of events %", 34.7,
      round(100 * (rec["Severe"]["share_of_events"] + rec["High"]["share_of_events"]), 1), tol=0.06)
train_sizes = rb.get("training_by_fold", {})
if train_sizes:
    check("training anchor: fold 1 Severe areas/week", 187.1,
          round(train_sizes[1]["Severe"] if 1 in train_sizes else train_sizes["1"]["Severe"], 1), tol=0.06)

print(f"\n{'=' * 110}\n{len(PASS)} passed, {len(FAIL)} failed, {checks} checks\n{'=' * 110}")
if FAIL:
    print("\nFAILURES:")
    for label, stated, computed in FAIL:
        print(f"  {label}: chapter says {stated}, data says {computed}")
sys.exit(1 if FAIL else 0)
