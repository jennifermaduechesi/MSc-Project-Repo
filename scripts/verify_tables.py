"""Check every numeric cell of every table in Chapter Three against recomputed data.

`verify_chapter3.py` reads four tables (11, 12, 14 and 15) and the prose figures. This
covers the other nine, so that no cell in the chapter is stated without something
re-deriving it from the artefacts.

The stated side of every check is read out of the chapter. Nothing is transcribed here,
for the reason given in `verify_chapter3.py`: a transcribed expectation stops testing the
document the moment the document changes.
"""
from __future__ import annotations
import hashlib, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

CHAPTER = Path("dissertation/chapter3_methodology.md")
RAW, PROC, REF = Path("data/raw"), Path("data/processed"), Path("data/reference")


def table(number: str) -> list[list[str]]:
    text = CHAPTER.read_text()
    block = text[text.index(f"\nTable {number}\n"):]
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


def cell(number: str, row_label: str, col: int) -> str:
    for r in table(number):
        if r[0] == row_label:
            return r[col]
    raise SystemExit(f"Table {number}: no row {row_label!r}")


PASS, FAIL = [], []

def check(label, stated, computed, tol=None):
    """Compare numerically when both sides are numbers, textually otherwise.

    Without this, 692.0 parsed out of a table cell fails against the integer 692 and the
    run drowns in false failures, which is worse than no check at all because it trains
    the reader to skim the output.
    """
    try:
        a, b = float(stated), float(computed)
        ok = abs(a - b) <= (0.0 if tol is None else tol)
    except (TypeError, ValueError):
        ok = str(stated) == str(computed)
    (PASS if ok else FAIL).append((label, stated, computed))


def num(s: str) -> float:
    return float(re.sub(r"[^0-9.\-]", "", s))


# ------------------------------------------------------------------ Table 3.2, the sources
gen = pd.read_excel(RAW / "DATA_SOURCE.xlsx")
spec = pd.read_excel(RAW / "Additional_data.xlsx")

check("T3.2 general records", num(cell("3.2", "Records as supplied", 1)), len(gen))
check("T3.2 specialist records", num(cell("3.2", "Records as supplied", 2)), len(spec))

for col, df, datecol, name in ((1, gen, "Date of Incident", "general"),
                               (2, spec, "Date", "specialist")):
    d = pd.to_datetime(df[datecol], errors="coerce").dropna()
    stated = cell("3.2", "Period covered", col)
    lo, hi = [pd.to_datetime(x.strip()) for x in stated.split(" to ")]
    check(f"T3.2 {name} period from", lo.date(), d.min().date())
    check(f"T3.2 {name} period to", hi.date(), d.max().date())
    coords = df["Latitude"].notna() & df["Longitude"].notna()
    check(f"T3.2 {name} coordinate share", num(cell("3.2", "Records carrying coordinates", col)),
          round(100 * coords.mean(), 1), tol=0.05)

for col, f in ((1, "DATA_SOURCE.xlsx"), (2, "Additional_data.xlsx")):
    p = RAW / f
    check(f"T3.2 file size {f}", num(cell("3.2", "File size", col)), p.stat().st_size)
    check(f"T3.2 md5 {f}", cell("3.2", "MD5 checksum", col),
          hashlib.md5(p.read_bytes()).hexdigest())

# ------------------------------------------------------- Table 3.3, how records were placed
fin = pd.read_csv(PROC / "incidents_final.csv", low_memory=False)
BASIS = {"Coordinate and name agree": "agree",
         "Settlement gazetteer": "gazetteer",
         "Disagreement resolved in favour of the coordinate": "conflict_kept_geometry",
         "Name only, no usable coordinate": "name_only",
         "Coordinate only, name unresolved": "geometry_only",
         "Disagreement resolved in favour of the name": "conflict_kept_name",
         "Not placed": "none"}
counts = fin["basis"].value_counts()
total = len(fin)
for label, key in BASIS.items():
    check(f"T3.3 {key} count", num(cell("3.3", label, 1)), counts.get(key, 0))
    check(f"T3.3 {key} share", num(cell("3.3", label, 2)), round(100 * counts.get(key, 0) / total, 2),
          tol=0.005)
check("T3.3 total", num(cell("3.3", "Total", 1)), total)
check("T3.3 rows sum to total", sum(counts.get(k, 0) for k in BASIS.values()), total)

# ------------------------------------------------------------------ Table 3.4, the panel
panel = pd.read_parquet(PROC / "panel.parquet")
weeks = pd.to_datetime(panel["week"]).sort_values().unique()
ev = panel["event_count"]
stated = cell("3.4", "Period", 1)
lo, hi = [pd.to_datetime(x.strip()) for x in stated.split(" to ")]
# Weeks are labelled by their Monday, so the period the panel covers runs from the first
# week's Monday to the last week's Sunday. Comparing the stated end against the last
# week's label would be comparing against a different thing.
check("T3.4 period from", lo.date(), pd.Timestamp(weeks.min()).date())
check("T3.4 period to", hi.date(),
      (pd.Timestamp(weeks.max()) + pd.Timedelta(days=6)).date())
check("T3.4 weeks", num(cell("3.4", "Weeks", 1)), len(weeks))
check("T3.4 areas", num(cell("3.4", "Local Government Areas", 1)), panel["pcode"].nunique())
check("T3.4 area-weeks", num(cell("3.4", "Area-week observations", 1)), len(panel))
check("T3.4 events placed", num(cell("3.4", "Kidnapping and banditry events placed in the window", 1)),
      int(ev.sum()))
check("T3.4 positive area-weeks", num(cell("3.4", "Positive area-weeks", 1)), int((ev > 0).sum()))
check("T3.4 positive rate", num(cell("3.4", "Positive rate", 1)), round(100 * (ev > 0).mean(), 3),
      tol=0.0005)
check("T3.4 exactly one event", num(cell("3.4", "Positive weeks holding exactly one event", 1)),
      round(100 * (ev[ev > 0] == 1).mean(), 1), tol=0.05)
check("T3.4 max events", num(cell("3.4", "Maximum events in any area-week", 1)), int(ev.max()))
ever, of_all = [x.strip() for x in cell("3.4", "Areas experiencing at least one event", 1).split(" of ")]
check("T3.4 areas ever positive", num(ever), panel.loc[ev > 0, "pcode"].nunique())
check("T3.4 areas in total", num(of_all), panel["pcode"].nunique())
ops = pd.read_csv(PROC / "incidents_final.csv", low_memory=False)
check("T3.4 operations", num(cell("3.4", "Security force operations available as a predictor", 1)),
      int(ops["is_operation"].sum()))

# ------------------------------------------------------------ Table 3.5, the feature groups
NON_FEATURE = {"pcode", "week", "event_count", "occurred", "lga", "state"}
feats = [c for c in panel.columns if c not in NON_FEATURE]
GROUPS = {
    "Own history":            lambda c: re.fullmatch(r"own_(events|active)_\d+w", c),
    "Background":             lambda c: c in {"own_events_cum", "own_rate_longrun",
                                              "nb_rate_longrun", "own_weeks_since", "own_ever"},
    "Neighbour activity":     lambda c: re.fullmatch(r"nb_(events|active)_\d+w", c),
    "Guardianship":           lambda c: re.fullmatch(r"(own|nb)_ops_\d+w", c),
    "Severity context":       lambda c: re.fullmatch(r"own_(deaths|taken)_\d+w", c),
    "Calendar and structure": lambda c: c in {"cal_sin", "cal_cos", "time_index",
                                              "nb_degree", "cov_records_13w"},
}
assigned = []
for label, pred in GROUPS.items():
    got = [c for c in feats if pred(c)]
    assigned += got
    check(f"T3.5 {label} count", num(cell("3.5", label, 1)), len(got))
check("T3.5 groups cover every feature", sorted(assigned), sorted(feats))
check("T3.5 group counts sum to 42", sum(num(r[1]) for r in table("3.5")[1:]), len(feats))

# --------------------------------------------------------- Table 3.6, the coverage regimes
inc = fin[fin["is_target"] & fin["pcode"].notna()].copy()
inc["week"] = pd.to_datetime(inc["date"]).dt.to_period("W-SUN").dt.start_time
pos = panel.assign(w=pd.to_datetime(panel["week"]))
for row in table("3.6")[1:]:
    lo, hi = [pd.to_datetime(x.strip()) for x in row[0].split(" to ")]
    win = pos[(pos["w"] >= lo) & (pos["w"] <= hi)]
    nweeks = win["w"].nunique()
    check(f"T3.6 {row[0]} weeks", num(row[2]), nweeks)
    check(f"T3.6 {row[0]} mean positives/week", num(row[3]),
          round((win["event_count"] > 0).sum() / nweeks, 1), tol=0.05)
check("T3.6 regime weeks sum to panel weeks", sum(num(r[2]) for r in table("3.6")[1:]), len(weeks))

# ----------------------------------------------------------------- Table 3.7, the folds
hurdle = json.load(open(PROC / "hurdle_results.json"))
folds = hurdle["stage_one"]["logistic"]
allw = sorted(pd.to_datetime(panel["week"]).unique())
for row in table("3.7")[1:]:
    i = int(row[0]) - 1
    f = folds[i]
    check(f"T3.7 fold {row[0]} training weeks", num(row[1]),
          allw.index(pd.Timestamp(f["test_from"])) - 0)
    check(f"T3.7 fold {row[0]} training ends", pd.to_datetime(row[2]).date(),
          (pd.Timestamp(f["test_from"]) - pd.Timedelta(days=7)).date())
    tf, tt = [x.strip() for x in row[3].split(" to ")]
    check(f"T3.7 fold {row[0]} test from", pd.to_datetime(tf).date(),
          pd.Timestamp(f["test_from"]).date())
    check(f"T3.7 fold {row[0]} test to", pd.to_datetime(tt).date(),
          pd.Timestamp(f["test_to"]).date())
    check(f"T3.7 fold {row[0]} test base rate", num(row[4]),
          round(100 * f["base_rate"], 2), tol=0.005)

# ------------------------------------------------- Table 3.10, the stack's member learners
ens = json.load(open(PROC / "ensemble_results.json"))
gnn = ens["config"]["gnn"]
spec10 = cell("3.10", "Recurrent graph network", 2)
check("T3.10 gnn hidden width", num(re.search(r"hidden width (\d+)", spec10).group(1)), gnn["hidden"])
WORD = {"one": 1, "two": 2, "three": 3, "ten": 10, "twelve": 12}
def as_int(token: str) -> int:
    return WORD[token] if token in WORD else int(token)
check("T3.10 gnn Chebyshev size",
      as_int(re.search(r"filter size (\w+)", spec10).group(1)), gnn["K"])
check("T3.10 gnn passes in the stack",
      as_int(re.search(r"(\w+) passes", spec10).group(1)), gnn["epochs"])
stg = json.load(open(PROC / "stgnn_results.json"))["config"]
prose = CHAPTER.read_text()
check("3.5 gnn passes standalone",
      as_int(re.search(r"over (\w+) passes through the training weeks",
                       prose).group(1)), stg["epochs"])
check("3.5 gnn hidden width", num(re.search(r"hidden state of width (\d+)", prose).group(1)),
      stg["hidden"])
adj = pd.read_csv(REF / "lga_adjacency.csv")
check("3.5 adjacency relations", num(re.search(r"([\d,]+) adjacency relations", prose).group(1)),
      len(adj))
check("3.5 features per week", num(re.search(r"774 areas by (\d+) features", prose).group(1)),
      len(feats))

# ----------------------------------------------------- Table 3.13, the Chapman estimates
merged = fin[fin["is_target"]].copy()
prot = json.load(open(PROC / "protocol_results.json"))
det = {d["state"]: d for d in prot["detection"]["pm0d"]}
for row in table("3.13")[1:]:
    st = row[0]
    if st not in det:
        FAIL.append((f"T3.13 {st} present in results", st, "missing"))
        continue
    d = det[st]
    check(f"T3.13 {st} general log", num(row[1]), d["n_main"])
    check(f"T3.13 {st} specialist", num(row[2]), d["n_additional"])
    check(f"T3.13 {st} linked", num(row[3]), d["matched"])
    check(f"T3.13 {st} observed", num(row[4]), d["observed"])
    check(f"T3.13 {st} detection", num(row[5]), round(d["detection"], 3), tol=0.0005)

# ------------------------------------------------------------------------- report
w = max(len(l) for l, _, _ in PASS + FAIL) + 2
for label, stated, computed in PASS:
    print(f"  [pass] {label:<{w}} stated {str(stated)[:34]:>36}  computed {str(computed)[:34]:>36}")
for label, stated, computed in FAIL:
    print(f"  [FAIL] {label:<{w}} stated {str(stated)[:34]:>36}  computed {str(computed)[:34]:>36}")
print("=" * 118)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(PASS)+len(FAIL)} table checks")
print("=" * 118)
sys.exit(1 if FAIL else 0)
