"""Check Chapter One's own-data claims against the artefacts, and against later chapters.

Chapter One was written before the panel was rebuilt and had never been checked. It quotes
figures of two kinds: those drawn from cited reports, which verify_citations.py covers, and
those drawn from this study's own corpus, which nothing covered until now. Only the second
kind is tested here, plus the requirement that any figure it shares with Chapters Three to
Five is the same figure in both places.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import pandas as pd

D = Path("dissertation")
CH1 = D / "chapter1_introduction_v2.md"
LATER = ["chapter3_methodology.md", "chapter4_results.md", "chapter5_conclusions.md"]
NORTH_WEST = ["Zamfara", "Katsina", "Kaduna", "Sokoto", "Kebbi", "Jigawa", "Kano"]
WINDOW = ("2024-07-01", "2025-06-30")

PASS, FAIL = [], []
def check(label, stated, computed, tol=0.0):
    try:
        ok = abs(float(stated) - float(computed)) <= tol
    except (TypeError, ValueError):
        ok = str(stated) == str(computed)
    (PASS if ok else FAIL).append((label, stated, computed))

text = CH1.read_text(encoding="utf-8")
def says(pattern, group=1):
    m = re.search(pattern, text)
    if not m:
        raise SystemExit(f"Chapter One text not found for: {pattern}")
    return m.group(group)

def num(s):
    return float(re.sub(r"[^0-9.\-]", "", s).rstrip("."))

inc = pd.read_csv("data/processed/incidents_final.csv", low_memory=False)
inc["date"] = pd.to_datetime(inc["date"])
panel = pd.read_parquet("data/processed/panel.parquet")
panel["week"] = pd.to_datetime(panel["week"])
raw_general = pd.read_excel("data/raw/DATA_SOURCE.xlsx")
raw_spec = pd.read_excel("data/raw/Additional_data.xlsx")

target = inc[inc["is_target"]]
placed = target[target["pcode"].notna()]
lo, hi = panel["week"].min(), panel["week"].max() + pd.Timedelta(days=6)
in_window = placed[(placed["date"] >= lo) & (placed["date"] <= hi)]
ev = panel["event_count"]

CHECKS = [
    ("general log records", r"a general incident log of ([\d,]+) records", len(raw_general), 0),
    ("specialist file records", r"specialist file of ([\d,]+) records", len(raw_spec), 0),
    ("duplicate records", r"the ([\d,]+) records in the second",
     len(raw_general) + len(raw_spec) - len(inc), 0),
    ("merged corpus size", r"a merged corpus of ([\d,]+) records", len(inc), 0),
    ("target events", r"of which ([\d,]+) are kidnapping or banditry events", len(target), 0),
    ("target events placed", r"Of those, ([\d,]+) could be placed", len(placed), 0),
    ("events inside the window", r"and ([\d,]+) fall inside the modelling window", len(in_window), 0),
    ("weeks in the window", r"a window of (\d+) weeks", panel["week"].nunique(), 0),
    ("area-weeks", r"([\d,]+) Local Government Area weeks", len(panel), 0),
    ("positive rate", r"only ([\d.]+) per cent of the [\d,]+ Local Government Area",
     100 * (ev > 0).mean(), 0.0005),
    ("share holding exactly one event", r"and ([\d.]+) per cent of those containing exactly one",
     100 * (ev[ev > 0] == 1).mean(), 0.05),
    ("areas", r"covers all (\d+) units of local administration", panel["pcode"].nunique(), 0),
    ("records not placed", r"leaving (\d+) records excluded", int(inc["pcode"].isna().sum()), 0),
    ("share of corpus placed", r"([\d.]+) per cent of the merged corpus was resolved",
     100 * inc["pcode"].notna().mean(), 0.05),
    ("events recorded in 2026", r"the (\d+) events recorded in 2026 are retained",
     int((in_window["date"].dt.year == 2026).sum()), 0),
]
for label, pat, computed, tol in CHECKS:
    check(f"data: {label}", num(says(pat)), computed, tol)

for label, pat, col in (("specialist coordinate coverage",
                         r"present on ([\d.]+) per cent of its rows", raw_spec),
                        ("general coordinate coverage",
                         r"against ([\d.]+) per cent of the main log", raw_general)):
    have = (col["Latitude"].notna() & col["Longitude"].notna()).mean()
    check(f"data: {label}", num(says(pat)), 100 * have, 0.05)

# The escalation test, run inside the specialist file alone as the chapter states.
spec = inc[inc["is_target"] & (inc["source_file"] == "additional") & inc["pcode"].notna()].copy()
spec_all = inc[inc["is_target"]].copy()
by_year = in_window.groupby(in_window["date"].dt.year)
check("data: target events in 2014", num(says(r"from (\d+) target events in 2014")),
      int(by_year.size().get(2014, 0)))
check("data: target events in 2025", num(says(r"in 2014 to ([\d,]+) in 2025")),
      int(by_year.size().get(2025, 0)))

# North West concentration, over the window the chapter names.
w = placed[(placed["date"] >= WINDOW[0]) & (placed["date"] <= WINDOW[1])]
nw, other = w[w["state"].isin(NORTH_WEST)], w[~w["state"].isin(NORTH_WEST)]
check("data: North West share of people kidnapped",
      num(says(r"account for ([\d.]+) per cent of the people recorded")),
      100 * nw["kidnapped"].fillna(0).sum() / w["kidnapped"].fillna(0).sum(), 0.05)
check("data: North West share of incidents",
      num(says(r"but only ([\d.]+) per cent of the kidnapping and banditry incidents")),
      100 * len(nw) / len(w), 0.05)
check("data: mean taken per North West incident",
      num(says(r"North West incident takes ([\d.]+) people")),
      nw["kidnapped"].fillna(0).mean(), 0.005)
check("data: mean taken per incident elsewhere",
      num(says(r"against ([\d.]+) elsewhere")), other["kidnapped"].fillna(0).mean(), 0.005)

# The background section now claims the recency baseline is already several times better
# than chance, which is a forward reference to Chapter Four and has to hold.
import json
import numpy as np
recency = json.load(open("data/processed/ensemble_results.json"))["results"]["recency"]
lift = float(np.mean([f["ap_lift_over_base"] for f in recency]))
check("data: recency baseline is several times better than chance", True, 2.0 < lift < 10.0)
print(f"      (recency lift over base is {lift:.2f}x)")

# ------------------------------------ any figure shared with a later chapter must match
later = "\n".join((D / f).read_text(encoding="utf-8") for f in LATER)
SHARED = ["1.961", "506,970", "81.3", "655", "774", "26,683", "292", "23,407", "4,616",
          "12,595", "905", "56.3", "93.6", "102", "513", "220", "1.38", "2.33", "366"]
missing = [t for t in SHARED if t in text and t not in later]
check("consistency: figures shared with Chapters Three to Five appear there too", [], missing)

w_ = max(len(l) for l, _, _ in PASS + FAIL) + 2
for l, s, c in PASS: print(f"  [pass] {l:<{w_}} stated {str(s)[:26]:>28}  computed {str(c)[:26]:>28}")
for l, s, c in FAIL: print(f"  [FAIL] {l:<{w_}} stated {str(s)[:26]:>28}  computed {str(c)[:26]:>28}")
print("=" * 108)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(PASS)+len(FAIL)} checks")
print("=" * 108)
sys.exit(1 if FAIL else 0)
