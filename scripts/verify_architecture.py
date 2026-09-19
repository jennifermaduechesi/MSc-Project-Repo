"""Check the architecture and algorithm sections of Chapter Three against the code.

These sections describe the model a reader is being asked to trust, so every number in
them is read back out of the prose and compared against the constant the training script
actually uses, or against the panel itself. A specification that drifts from the code is
worse than no specification, because it reads as authoritative.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
import pandas as pd

CH3 = Path("dissertation/chapter3_methodology.md").read_text(encoding="utf-8")
SRC = Path("scripts/train_ensemble.py").read_text()
PANEL = Path("data/processed/panel.parquet")
ADJ = Path("data/reference/lga_adjacency.csv")

fails, passes = [], 0


def check(label, expected, got):
    global passes
    ok = expected == got
    if ok:
        passes += 1
    else:
        fails.append(label)
    print(f"  [{'pass' if ok else 'FAIL'}] {label:<58} code/data {expected!r:>14}  chapter {got!r}")


def const(name: str) -> int | float:
    """Read a module-level constant out of the training script."""
    m = re.search(rf"^{name}\s*=\s*([\d.]+)", SRC, re.M)
    if m:
        return float(m.group(1)) if "." in m.group(1) else int(m.group(1))
    m = re.search(rf"^GNN_HIDDEN, GNN_K, GNN_EPOCHS, GNN_LR = ([\d., ]+)$", SRC, re.M)
    vals = [v.strip() for v in m.group(1).split(",")]
    order = {"GNN_HIDDEN": 0, "GNN_K": 1, "GNN_EPOCHS": 2, "GNN_LR": 3}
    v = vals[order[name]]
    return float(v) if "." in v else int(v)


def kw(pattern: str) -> str:
    """Read a keyword argument out of the training script."""
    return re.search(pattern, SRC).group(1)


WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
         "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def said(pattern: str, group: int = 1):
    """Read a number back out of the chapter, in digits or spelled out."""
    m = re.search(pattern, CH3)
    if not m:
        return None
    raw = m.group(group).replace(",", "")
    if raw.lower() in WORDS:
        return WORDS[raw.lower()]
    return float(raw) if "." in raw else int(raw)


def main() -> None:
    panel = pd.read_parquet(PANEL)
    NOT = {"pcode", "week", "lga", "state", "event_count", "occurred"}
    n_feat = len([c for c in panel.columns if c not in NOT])
    edges = len(pd.read_csv(ADJ))

    print("-- the panel and the graph -------------------------------------------------")
    check("areas", panel.pcode.nunique(), said(r"(\d+) areas by (\d+) weeks"))
    check("weeks", panel.week.nunique(), said(r"(\d+) areas by ([\d,]+) weeks", 2))
    check("features per row", n_feat, said(r"each row carrying (\d+) features"))
    check("area-weeks", len(panel), said(r"([\d,]+) area-weeks, positive in"))
    check("positive rate", round(100 * panel.occurred.mean(), 2),
          said(r"positive in ([\d.]+) per cent of rows"))
    check("undirected adjacency relations", edges,
          said(r"([\d,]+) undirected adjacency relations"))
    check("directed pairs", edges * 2, said(r"An edge list of ([\d,]+) directed pairs"))
    check("graph input matrices", panel.week.nunique(),
          said(r"sequence of ([\d,]+) weekly matrices"))
    check("graph matrix shape, areas", panel.pcode.nunique(),
          said(r"each of (\d+) areas by \d+ features"))
    check("graph matrix shape, features", n_feat,
          said(r"each of \d+ areas by (\d+) features"))

    print("\n-- the tabular learners ----------------------------------------------------")
    check("logistic iterations", int(kw(r"LogisticRegression\(max_iter=(\d+)")),
          said(r"up to ([\d,]+) iterations"))
    check("calibration groups", int(kw(r"def calibrate_probabilities\(.*?n_splits=(\d+)")),
          said(r"Divide the training areas into (three) groups"))
    check("forest trees", int(kw(r"n_estimators=(\d+), max_depth=14")),
          said(r"Draw (\d+) bootstrap samples"))
    check("forest depth", int(kw(r"max_depth=(\d+), min_samples_leaf")),
          said(r"Stop a branch at depth (\d+)"))
    check("forest minimum leaf", int(kw(r"min_samples_leaf=(\d+)")),
          said(r"fewer than (\d+) observations in a leaf"))
    check("boosting rounds", int(kw(r"n_estimators=(\d+), max_depth=4")),
          said(r"For each of (\d+) rounds"))
    check("boosting depth", int(kw(r"max_depth=(\d+), learning_rate")),
          said(r"regression tree of depth (\d+)"))
    check("boosting learning rate", float(kw(r"learning_rate=([\d.]+)")),
          said(r"scaled by a learning rate of ([\d.]+)"))
    check("boosting subsample per cent", int(float(kw(r"subsample=([\d.]+)")) * 100),
          said(r"Draw (\d+) per cent of the rows"))

    print("\n-- the graph network -------------------------------------------------------")
    check("hidden width", const("GNN_HIDDEN"), said(r"hidden width of (\d+)"))
    check("Chebyshev order", const("GNN_K"),
          said(r"Chebyshev graph convolutions of order (two)"))
    check("training passes", const("GNN_EPOCHS"), said(r"For each of (ten) passes"))
    check("optimiser learning rate", const("GNN_LR"),
          said(r"Adam at a learning rate of ([\d.]+)"))
    check("optimiser step interval", int(re.search(r"\(t \+ 1\) % (\d+) == 0", SRC).group(1)),
          said(r"every (\d+) weeks and at the end"))

    print("\n-- the stack ---------------------------------------------------------------")
    check("inner block weeks", const("INNER_WEEKS"), said(r"an inner block of (\d+) weeks"))
    check("test block weeks", const("TEST_WEEKS"),
          said(r"test block of the (\d+) weeks that follow"))
    check("folds", const("N_FOLDS"), said(r"each of the (five) folds"))
    check("fold step in weeks", const("TEST_WEEKS"),
          said(r"every boundary moved forward by (\d+) weeks"))
    check("base learners in the stack",
          len(re.search(r'BASE = \[(.*?)\]', SRC, re.S).group(1).split(",")),
          said(r"lower level holds (four) base learners"))

    print("\n-- structure ---------------------------------------------------------------")
    algos = re.findall(r"^Algorithm (\d+)$", CH3, re.M)
    check("algorithms numbered in order", [str(i) for i in range(1, len(algos) + 1)], algos)
    check("one algorithm per base learner plus the stack", 5, len(algos))
    for heading in ("Methodological Framework", "Architecture of the Proposed Model",
                    "The Algorithms Step by Step", "Justification of the Proposed Model"):
        check(f"section present: {heading}", True, f"### {heading}" in CH3)
    check("no em dashes", 0, CH3.count("—"))
    check("no supervisor references", 0, len(re.findall(r"supervisor", CH3, re.I)))

    print("=" * 96)
    print(f"{passes} passed, {len(fails)} failed, {passes + len(fails)} checks")
    print("=" * 96)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
