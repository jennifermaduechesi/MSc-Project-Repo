"""Check Chapter Five twice over: against the data, and against the other chapters.

Chapter Five states no new results. Every figure in it is carried forward from Chapter
Three or Chapter Four, so it can fail in two ways that a single check would miss. A figure
can disagree with the data, and a figure can disagree with the chapter it came from while
still being arithmetically plausible. Both are tested here.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd

D = Path("dissertation")
PROC = Path("data/processed")
APP = Path("app/data")
CH5 = D / "chapter5_conclusions.md"
SOURCES = {"Chapter Three": D / "chapter3_methodology.md",
           "Chapter Four": D / "chapter4_results.md"}

PASS, FAIL = [], []
def check(label, stated, computed, tol=0.0):
    try:
        ok = abs(float(stated) - float(computed)) <= tol
    except (TypeError, ValueError):
        ok = str(stated) == str(computed)
    (PASS if ok else FAIL).append((label, stated, computed))

def says(pattern, group=1):
    m = re.search(pattern, CH5.read_text())
    if not m:
        raise SystemExit(f"Chapter Five text not found for: {pattern}")
    return m.group(group)

def num(s):
    return float(re.sub(r"[^0-9.\-]", "", s).rstrip("."))

ens = json.load(open(PROC / "ensemble_results.json"))
hur = json.load(open(PROC / "hurdle_results.json"))
prot = json.load(open(PROC / "protocol_results.json"))
bands = json.load(open(PROC / "risk_bands.json"))
scores = pd.read_parquet(PROC / "ensemble_test_scores.parquet")
panel = pd.read_parquet(PROC / "panel.parquet")
inc = pd.read_csv(PROC / "incidents_final.csv", low_memory=False)

def mean_of(key, metric, group=None):
    g = group if group is not None else ens["results"]
    return float(np.mean([f[metric] for f in g[key] if f.get(metric) is not None]))

def caught(col, k=20):
    hits = tot = 0
    for _, wk in scores.groupby("week", sort=False):
        hits += wk.nlargest(k, col)["occurred"].sum(); tot += wk["occurred"].sum()
    return int(hits), int(tot)

stack, total = caught("ensemble_unweighted")
rec, _ = caught("recency")

# ------------------------------------------------- part one: against the data
ev = panel["event_count"]
CHECKS = [
    ("areas", r"onto Nigeria's (\d+) Local Government Areas", panel["pcode"].nunique(), 0),
    ("records placed", r"placing ([\d,]+) records", len(inc), 0),
    ("records not placed", r"of which (\d+) could not be located", int(inc["pcode"].isna().sum()), 0),
    ("area-weeks", r"a panel of ([\d,]+) area-weeks", len(panel), 0),
    ("weeks", r"covering (\d+) weeks from January", pd.to_datetime(panel["week"]).nunique(), 0),
    ("positive area-weeks", r"in which ([\d,]+) area-weeks contain", int((ev > 0).sum()), 0),
    ("positive rate", r"a positive rate of ([\d.]+) per cent", 100 * (ev > 0).mean(), 0.0005),
    ("held-out rows", r"giving ([\d,]+) held-out area-weeks", len(scores), 0),
    ("test weeks", r"area-weeks across (\d+) test weeks", scores["week"].nunique(), 0),
    ("test base rate", r"at a base rate of ([\d.]+)", scores["occurred"].mean(), 0.00005),
    ("stack catches", r"ensemble identifies ([\d,]+) of the", stack, 0),
    ("attacked area-weeks", r"identifies [\d,]+ of the ([\d,]+) attacked", total, 0),
    ("recency catches", r"against ([\d,]+) for a list built", rec, 0),
    ("stack over recency", r"That is (\d+) additional attacked areas", stack - rec, 0),
    ("AP lift over base", r"sits ([\d.]+) times above the base rate",
     mean_of("ensemble_unweighted", "ap_lift_over_base"), 0.05),
    ("lift at top 20", r"catches ([\d.]+) times what a random list",
     mean_of("ensemble_unweighted", "lift_at_20"), 0.05),
    ("stack AP", r"average precision of ([\d.]+) against [\d.]+ for a four-week",
     mean_of("ensemble_unweighted", "average_precision"), 0.0005),
    ("recency AP", r"of [\d.]+ against ([\d.]+) for a four-week",
     mean_of("recency", "average_precision"), 0.0005),
    ("stack recall per cent", r"places ([\d.]+) per cent of a week's",
     100 * mean_of("ensemble_unweighted", "recall_at_20"), 0.05),
    ("recency recall per cent", r"in a list of twenty against ([\d.]+) per cent",
     100 * mean_of("recency", "recall_at_20"), 0.05),
    ("exactly one event", r"because ([\d.]+) per cent of positive area-weeks hold exactly one",
     100 * (ev[ev > 0] == 1).mean(), 0.05),
    ("graph AP", r"It reaches an average precision of ([\d.]+)",
     mean_of("stgnn", "average_precision"), 0.0005),
    ("logistic AP", r"of [\d.]+ against ([\d.]+) for the calibrated logistic model and a ROC",
     mean_of("logistic", "average_precision"), 0.0005),
    ("graph ROC", r"a ROC AUC of ([\d.]+) against", mean_of("stgnn", "roc_auc"), 0.0005),
    ("logistic ROC", r"ROC AUC of [\d.]+ against ([\d.]+), while", mean_of("logistic", "roc_auc"), 0.0005),
    ("graph recall", r"recall at the top twenty is ([\d.]+) against",
     mean_of("stgnn", "recall_at_20"), 0.0005),
    ("logistic recall", r"top twenty is [\d.]+ against ([\d.]+)\. It wins",
     mean_of("logistic", "recall_at_20"), 0.0005),
    ("graph Brier", r"carrying a Brier score of ([\d.]+), so it cannot",
     mean_of("stgnn", "brier"), 0.0005),
    # The restructured chapter states the stack's own Brier rather than comparing it to
    # the logistic member's, so this now checks the stack.
    ("stack Brier", r"carries a Brier score of ([\d.]+), matching the best-calibrated",
     mean_of("ensemble_unweighted", "brier"), 0.0005),
    ("stack recall", r"average precision and ([\d.]+) recall at twenty",
     mean_of("ensemble_unweighted", "recall_at_20"), 0.0005),
    ("weighted meta Brier", r"produced a Brier score of ([\d.]+), which would have left",
     mean_of("ensemble", "brier"), 0.0005),
    ("two-week AP cost", r"records costs ([\d.]+) of average precision",
     mean_of("0w", "average_precision", prot["delay"])
     - mean_of("2w", "average_precision", prot["delay"]), 0.0005),
    ("two-week recall cost", r"average precision and ([\d.]+) of recall at twenty",
     mean_of("0w", "recall_at_20", prot["delay"])
     - mean_of("2w", "recall_at_20", prot["delay"]), 0.0005),
    ("pooled detection", r"pooling at ([\d.]+)", prot["under_reporting"]["pooled_detection"], 0.0005),
    ("reliable states", r"across the (\d+) states where the estimate is reliable",
     prot["under_reporting"]["states_with_own_estimate"], 0),
    ("weight min", r"weights ranging from ([\d.]+) to", prot["under_reporting"]["weight_min"], 0.0005),
    ("weight max", r"ranging from [\d.]+ to ([\d.]+)", prot["under_reporting"]["weight_max"], 0.0005),
    ("low tier share of events", r"([\d.]+) per cent of all recorded events fell",
     100 * bands["low_share_pooled"], 0.05),
]
for label, pat, computed, tol in CHECKS:
    check(f"data: {label}", num(says(pat)), computed, tol)

det = prot["detection"]["pm0d"]
rel = [x for x in det if x["reliable"]]
lo, hi = min(rel, key=lambda x: x["detection"]), max(rel, key=lambda x: x["detection"])
check("data: lowest detection", num(says(r"detection between ([\d.]+) in")), lo["detection"], 0.0005)
check("data: lowest detection state", says(r"between [\d.]+ in (\w+) and"), lo["state"])
check("data: highest detection", num(says(r"and ([\d.]+) in \w+ across the")), hi["detection"], 0.0005)
check("data: highest detection state", says(r"and [\d.]+ in (\w+) across the"), hi["state"])
check("data: share of records in reliable states", num(says(r"which hold ([\d.]+) per cent of all records")),
      100 * sum(x["observed"] for x in rel) / sum(x["observed"] for x in det), 0.05)

ap = [f["average_precision"] for f in ens["results"]["ensemble_unweighted"]]
check("data: fold AP minimum", num(says(r"ranges from ([\d.]+) to [\d.]+ across the five")), min(ap), 0.0005)
check("data: fold AP maximum", num(says(r"from [\d.]+ to ([\d.]+) across the five")), max(ap), 0.0005)
check("data: fold AP spread", num(says(r"a spread of ([\d.]+)")), max(ap) - min(ap), 0.0005)
f5 = {k: v[4]["average_precision"] for k, v in ens["results"].items()}
check("data: within-fold spread", num(says(r"never exceeds ([\d.]+)")), max(f5.values()) - min(f5.values()), 0.0005)
check("data: stage two loses in every fold", 5,
      sum(1 for f in hur["stage_two"] if f["mae"] > f["mae_always_one"]))
check("data: graph wins four folds", 4,
      sum(1 for a, b in zip([f["average_precision"] for f in ens["results"]["stgnn"]],
                            [f["average_precision"] for f in ens["results"]["logistic"]]) if a > b))
check("data: meta largest weight count", 3,
      sum(1 for x in ens["meta_weights"] if x["meta"] == "ensemble_unweighted"
          and max(("logistic", "random_forest", "gradient_boosting", "stgnn"),
                  key=lambda m: x[m]) == "stgnn"))
hz = prot["horizon"]
for days, pat in ((7, r"falls from ([\d.]+) at seven days"), (14, r"to ([\d.]+) at fourteen"),
                  (28, r"and ([\d.]+) at twenty-eight, and recall")):
    check(f"data: lift at {days}d", num(says(pat)), mean_of(f"{days}d", "ap_lift_over_base", hz), 0.05)
for days, pat in ((7, r"top twenty from ([\d.]+) to"), (14, r"from [\d.]+ to ([\d.]+) to [\d.]+\. Within"),
                  (28, r"from [\d.]+ to [\d.]+ to ([\d.]+)\. Within")):
    check(f"data: recall at {days}d", num(says(pat)), mean_of(f"{days}d", "recall_at_20", hz), 0.0005)
check("data: AP at 28d", num(says(r"same range, from [\d.]+ to ([\d.]+)")),
      mean_of("28d", "average_precision", hz), 0.0005)
for days, pat in ((7, r"catching about (\d+) per cent of the week's"),
                  (28, r"the same list catches about (\d+) per cent")):
    check(f"data: rounded recall at {days}d", num(says(pat)),
          round(100 * mean_of(f"{days}d", "recall_at_20", hz)))

# ------------------------------- part two: against the chapters it carries forward from
ch5 = CH5.read_text()
body = {k: v.read_text() for k, v in SOURCES.items()}
WHITELIST = {"5", "20", "52", "42", "2", "4", "14", "28", "7"}
tokens = set()
for m in re.finditer(r"(?<![\d.\w])\d+(?:[.,]\d+)+|(?<![\d.\w])\d{3,}|(?<![\d.\w])\d+(?=\s*per cent)", ch5):
    t = m.group(0)
    if re.fullmatch(r"(19|20)\d\d", t.replace(",", "")) or t in WHITELIST:
        continue
    tokens.add(t)
def carried(t: str) -> bool:
    """A figure counts as carried forward if it appears in a source chapter as written,
    or as the decimal a percentage restates (19.6 per cent for a recall of 0.196)."""
    if any(t in v for v in body.values()):
        return True
    try:
        as_decimal = f"{float(t.replace(',', '')) / 100:.3f}"
    except ValueError:
        return False
    return any(as_decimal in v for v in body.values())

orphans = [t for t in sorted(tokens) if not carried(t)]
check("consistency: every figure also appears in Chapter Three or Four", [], orphans)
print(f"  ({len(tokens)} distinct figures carried forward, "
      f"{len(tokens) - len(orphans)} found in the source chapters)")

# ------------------------- part three: the questions and the house spelling convention
# The Research Questions section was removed at the supervisor's instruction, so Chapter
# Five now draws its conclusions against the objectives. Each objective must have one.
c1 = (D / "chapter1_introduction_v2.md").read_text()
block = c1[c1.index("The specific objectives are:"):]
block = block[:block.index("\n## ")]
objectives = [m.group(1) for m in re.finditer(r"^(\d+)\.\s+To ", block, re.M)]
headings = re.findall(r"^### Objective (\w+):", ch5, re.M)
WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
check("consistency: one conclusion per objective", len(objectives), len(headings))
check("consistency: objectives numbered in order", [str(i) for i in range(1, len(headings) + 1)],
      [str(WORDS[h.lower()]) for h in headings])
check("consistency: no research questions remain anywhere", 0,
      sum((D / f).read_text().lower().count("research question")
          for f in ["chapter1_introduction_v2.md", "chapter2_literature_review.md",
                    "chapter3_methodology.md", "chapter4_results.md",
                    "chapter5_conclusions.md"]))

# The dissertation is written in British English. "percent" is the American form and had
# drifted into one chapter, which would read as carelessness beside 74 uses of "per cent".
ALL = ["chapter1_introduction_v2.md", "chapter2_literature_review.md",
       "chapter3_methodology.md", "chapter4_results.md", "chapter5_conclusions.md"]
stray = {f: len(re.findall(r"\bpercent\b", (D / f).read_text())) for f in ALL}
check("consistency: no American 'percent' in any chapter", 0, sum(stray.values()))

w = max(len(l) for l, _, _ in PASS + FAIL) + 2
for l, s_, c_ in PASS: print(f"  [pass] {l:<{w}} stated {str(s_)[:30]:>32}  computed {str(c_)[:30]:>32}")
for l, s_, c_ in FAIL: print(f"  [FAIL] {l:<{w}} stated {str(s_)[:34]:>36}  computed {str(c_)[:34]:>36}")
print("=" * 118)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(PASS)+len(FAIL)} checks")
print("=" * 118)
sys.exit(1 if FAIL else 0)
