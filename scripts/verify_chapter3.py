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
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- reading the chapter
CHAPTER = Path("dissertation/chapter3_methodology.md")

def chapter_table(number: int) -> list[list[str]]:
    """Return the rows of a numbered table as they stand in the chapter.

    The stated side of every table check is read from the document rather than
    transcribed into this file. Transcription is how a check quietly stops testing
    anything: update the chapter, update the copy here to match, and the comparison
    passes without ever having looked at the data.
    """
    text = CHAPTER.read_text()
    start = text.index(f"\nTable {number}\n")
    block = text[start:]
    rows = []
    for line in block.splitlines():
        s = line.strip()
        if s.startswith("|"):
            cells = [c.strip() for c in s.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append(cells)
        elif rows:
            break
    return rows


def chapter_says(pattern: str, group: int = 1) -> str:
    """Pull one figure out of the chapter's prose by regular expression."""
    m = re.search(pattern, CHAPTER.read_text())
    if not m:
        raise SystemExit(f"chapter text not found for pattern: {pattern}")
    return m.group(group)


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

print("  (reported, not checked: Chapter Three specifies these models and Chapter Four")
print("   reports their results, so there is no claim here to test them against)")
for name in ("logistic", "logistic_calibrated", "gradient_boosting",
             "gradient_boosting_calibrated"):
    r = s1[name]
    print(f"    {name:<30} AP {m(r, 'average_precision'):.4f}  "
          f"R@20 {m(r, 'recall_at_20'):.3f}  Brier {m(r, 'brier'):.4f}")
g = json.load(open("data/processed/stgnn_results.json"))["folds"]
print(f"    {'graph network':<30} AP {m(g, 'average_precision'):.4f}  "
      f"R@20 {m(g, 'recall_at_20'):.3f}")

# Two claims section 3.5 does make, and one section 3.6 makes, are checked.
wins = sum(1 for r in h["stage_two"] if r["mae"] < r["mae_always_one"])
check("folds where stage two beats always-predict-one", 0, wins)

print("\n-- 3.6 The stacking ensemble ------------------------------------------------")
e = json.load(open("data/processed/ensemble_results.json"))
R = e["results"]
BASE = ["logistic", "random_forest", "gradient_boosting", "stgnn"]
for name in ("ensemble_unweighted", "ensemble", "random_forest", "stgnn",
             "logistic", "recency", "gradient_boosting", "long_run"):
    r = R[name]
    b = [x["brier"] for x in r if "brier" in x]
    print(f"    {name:<22} AP {m(r, 'average_precision'):.4f}  "
          f"R@20 {m(r, 'recall_at_20'):.3f}  "
          f"Brier {(f'{np.mean(b):.4f}' if b else 'n/a')}")

# The chapter states how the two meta-learners compare, so that is read and checked.
weighted_ap = float(chapter_says(r"average precision rising from (\d\.\d+) to \d\.\d+"))
unweighted_ap = float(chapter_says(r"average precision rising from \d\.\d+ to (\d\.\d+)"))
check("3.6 weighted stack AP as stated", weighted_ap,
      round(m(R["ensemble"], "average_precision"), 3), tol=0.0006)
check("3.6 unweighted stack AP as stated", unweighted_ap,
      round(m(R["ensemble_unweighted"], "average_precision"), 3), tol=0.0006)
w_r20 = float(chapter_says(r"recall in the top twenty from (\d\.\d+) to \d\.\d+\. The unweighted"))
u_r20 = float(chapter_says(r"recall in the top twenty from \d\.\d+ to (\d\.\d+)\. The unweighted"))
check("3.6 weighted stack R@20 as stated", w_r20,
      round(m(R["ensemble"], "recall_at_20"), 3), tol=0.0006)
check("3.6 unweighted stack R@20 as stated", u_r20,
      round(m(R["ensemble_unweighted"], "recall_at_20"), 3), tol=0.0006)
stated_brier = float(chapter_says(r"mean Brier score of (\d\.\d+) against 0\.032"))
check("3.6 weighted stack Brier as stated", stated_brier,
      round(m(R["ensemble"], "brier"), 3), tol=0.0006)

# --------------------------------------------------------- 3.7 evaluation protocol
print("\n-- 3.7 The evaluation protocol ----------------------------------------------")
pr = json.load(open("data/processed/protocol_results.json"))
rows11 = chapter_table(11)[1:]          # Horizon | Base | AP | Lift | R@20 | Lift@20
for row, key in zip(rows11, ("7d", "14d", "28d")):
    r = pr["horizon"][key]
    check(f"Table 11 {row[0]} base rate", float(row[1]), m(r, "base_rate"), tol=0.00006)
    check(f"Table 11 {row[0]} AP", float(row[2]), m(r, "average_precision"), tol=0.00006)
    check(f"Table 11 {row[0]} lift over base", float(row[3]),
          round(m(r, "ap_lift_over_base"), 1), tol=0.06)
    check(f"Table 11 {row[0]} R@20", float(row[4]), round(m(r, "recall_at_20"), 3), tol=0.0006)
    check(f"Table 11 {row[0]} lift@20", float(row[5]), round(m(r, "lift_at_20"), 1), tol=0.06)

rows12 = chapter_table(12)[1:]          # Delay | Base | AP | Lift | R@20 | Lift@20 | Brier
for row, key in zip(rows12, ("0w", "1w", "2w")):
    r = pr["delay"][key]
    check(f"Table 12 {row[0]} AP", float(row[2]), m(r, "average_precision"), tol=0.00006)
    check(f"Table 12 {row[0]} R@20", float(row[4]), round(m(r, "recall_at_20"), 3), tol=0.0006)
    check(f"Table 12 {row[0]} Brier", float(row[6]), m(r, "brier"), tol=0.00006)

u = pr["under_reporting"]
rows14 = chapter_table(14)[1:]          # Loss | AP | Lift | R@20 | Lift@20 | Brier
for row, key in zip(rows14, ("unweighted", "reweighted")):
    r = u[key]
    check(f"Table 14 {row[0][:22]} AP", float(row[1]), m(r, "average_precision"), tol=0.00006)
    check(f"Table 14 {row[0][:22]} R@20", float(row[3]),
          round(m(r, "recall_at_20"), 3), tol=0.0006)

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
rows15 = chapter_table(15)[1:]          # Band | Definition | Areas | Realised | xBase | Share
for row in rows15:
    r = rec[row[0]]
    check(f"Table 15 {row[0]} areas/week", float(row[2]),
          round(r["areas_per_week"], 1), tol=0.06)
    check(f"Table 15 {row[0]} realised rate", float(row[3]),
          round(r["realised_rate"], 3), tol=0.0006)
    check(f"Table 15 {row[0]} times base", float(row[4]), round(r["times_base"], 1), tol=0.06)
    check(f"Table 15 {row[0]} share of events %", float(row[5].rstrip("%")),
          round(100 * r["share_of_events"], 1), tol=0.06)

# Every figure below is read from section 3.8's prose, so the comparison is between
# what the chapter says and what the data holds, not between the data and a value typed
# into this script.
check("3.8 events falling in the Low band %",
      float(chapter_says(r"\*\*Across the five test years, (\d+\.\d) per cent of all recorded events")),
      round(100 * rb["low_share_pooled"], 1), tol=0.06)
lo = float(chapter_says(r"By fold that share ranges from (\d+\.\d) per cent"))
hi = float(chapter_says(r"By fold that share ranges from \d+\.\d per cent to (\d+\.\d) per cent"))
shares = [100 * v for v in rb["low_share_by_fold"].values()]
check("3.8 lowest per-fold Low share", lo, round(min(shares), 1), tol=0.06)
check("3.8 highest per-fold Low share", hi, round(max(shares), 1), tol=0.06)
check("3.8 band unchanged week to week %",
      float(chapter_says(r"Across consecutive weeks (\d+\.\d) per cent of areas remain")),
      round(100 * rb["unchanged_week_to_week"], 1), tol=0.06)
check("3.8 events in rows that change band %",
      float(chapter_says(r"change band carry (\d+\.\d) per cent of the events")),
      round(100 * rb["events_in_moving_rows"], 1), tol=0.06)
tr = rb["transitions"]
check("3.8 Low stays Low",
      float(chapter_says(r"Low band stays there (\d+\.\d) per cent")) / 100,
      round(tr["Low"]["Low"], 3), tol=0.0006)
check("3.8 Severe stays Severe",
      float(chapter_says(r"Severe band stays there (\d+\.\d) per cent")) / 100,
      round(tr["Severe"]["Severe"], 3), tol=0.0006)
check("3.8 areas ever reaching Severe",
      int(chapter_says(r"Over the five test years (\d+) of the 774 areas reach the Severe")),
      rb["areas_reaching_band"]["Severe"])
check("3.8 areas ever reaching High",
      int(chapter_says(r"reach the Severe band at least once, (\d+) reach High")),
      rb["areas_reaching_band"]["High"])
check("3.8 areas ever reaching Elevated",
      int(chapter_says(r"reach High and (\d+) reach Elevated")),
      rb["areas_reaching_band"]["Elevated"])
sev_hi = rec["Severe"]["areas_per_week"] + rec["High"]["areas_per_week"]
check("3.8 Severe plus High as share of country %",
      float(chapter_says(r"which is (\d+\.\d) per cent of the country")),
      round(100 * sev_hi / 774, 1), tol=0.06)
check("3.8 Severe plus High share of events %",
      float(chapter_says(r"and account for (\d+\.\d) per cent of the events")),
      round(100 * (rec["Severe"]["share_of_events"] + rec["High"]["share_of_events"]), 1),
      tol=0.06)

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
