"""Build the Local Government Area adjacency graph from the official boundaries.

Two areas are neighbours when their polygons share a boundary. Shared boundaries are
detected with `touches`, which is true for polygons meeting along an edge or at a point
and false for polygons that merely come close. Coastal and riverine boundaries in the
source data are not always perfectly noded, so a small buffer is applied before the
test: two areas whose polygons come within the tolerance are treated as touching.

The tolerance matters and is set deliberately. At 0 the graph loses real neighbours
wherever the source polygons have slivers between them. Set too high it invents
neighbours across water. The value below was chosen by checking how many areas end up
with no neighbour at all, which should be a small number of genuine islands rather
than an artefact.

Output is an edge list and a summary, written to data/reference/.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
from shapely.geometry import shape
from shapely.strtree import STRtree

# Degrees. About 220 m, which closes digitising slivers without reaching across a
# river or a strait.
#
# The value was set by sensitivity rather than by preference. Between 0 and 0.005 the
# edge count moves only from 2,220 to 2,244, about one per cent, so the graph is not
# sensitive to the choice. What the value does decide is Bakassi in Cross River, whose
# nearest neighbour Akpabuyo sits 0.0014 degrees away, roughly 155 metres. Below 0.002
# Bakassi has no neighbour at all and would receive no neighbour features. Connecting
# the two is right on the source's own terms: the OCHA dataset notes that Bakassi is
# thought to be uninhabited and that any population it has is carried in the Akpabuyo
# record. At 0.002 that single edge is added and one other, and nothing else changes.
SNAP_TOLERANCE = 0.002


def load_boundaries(zip_path: Path):
    with zipfile.ZipFile(zip_path) as bundle:
        with bundle.open("nga_admin2.geojson") as handle:
            collection = json.load(handle)
    records = []
    for feature in collection["features"]:
        if feature.get("geometry") is None:
            continue
        properties = feature["properties"]
        records.append({
            "pcode": properties.get("adm2_pcode"),
            "lga": properties.get("adm2_name"),
            "state": properties.get("adm1_name"),
            "geometry": shape(feature["geometry"]),
        })
    return records


def build_edges(records, tolerance: float) -> set[tuple[str, str]]:
    geometries = [r["geometry"] for r in records]
    tree = STRtree(geometries)
    edges: set[tuple[str, str]] = set()
    for index, record in enumerate(records):
        geometry = record["geometry"]
        probe = geometry.buffer(tolerance) if tolerance else geometry
        for candidate in tree.query(probe):
            other = int(candidate)
            if other == index:
                continue
            if geometry.distance(geometries[other]) <= tolerance:
                a, b = record["pcode"], records[other]["pcode"]
                edges.add((a, b) if a < b else (b, a))
    return edges


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", default="data/reference/nga_admin_boundaries.geojson.zip")
    parser.add_argument("--out", default="data/reference/lga_adjacency.csv")
    parser.add_argument("--tolerance", type=float, default=SNAP_TOLERANCE)
    args = parser.parse_args()

    records = load_boundaries(Path(args.geometry))
    print(f"areas loaded: {len(records)}")
    if len(records) != 774:
        print(f"  WARNING: expected 774 areas, found {len(records)}")

    edges = build_edges(records, args.tolerance)
    degree = Counter()
    for a, b in edges:
        degree[a] += 1
        degree[b] += 1

    by_pcode = {r["pcode"]: r for r in records}
    isolated = [p for p in by_pcode if degree[p] == 0]

    print(f"\ntolerance        : {args.tolerance} degrees")
    print(f"edges            : {len(edges):,}")
    print(f"mean neighbours  : {2 * len(edges) / len(records):.2f}")
    print(f"min / max        : {min(degree.values()) if degree else 0} / {max(degree.values()) if degree else 0}")
    print(f"areas with none  : {len(isolated)}")
    for pcode in isolated:
        record = by_pcode[pcode]
        print(f"   {pcode}  {record['lga']}, {record['state']}")

    frame = pd.DataFrame(sorted(edges), columns=["pcode_a", "pcode_b"])
    frame["lga_a"] = frame.pcode_a.map(lambda p: by_pcode[p]["lga"])
    frame["lga_b"] = frame.pcode_b.map(lambda p: by_pcode[p]["lga"])
    frame["state_a"] = frame.pcode_a.map(lambda p: by_pcode[p]["state"])
    frame["state_b"] = frame.pcode_b.map(lambda p: by_pcode[p]["state"])
    frame["cross_state"] = frame.state_a != frame.state_b

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out, index=False)
    print(f"\ncross-state edges: {int(frame.cross_state.sum()):,} of {len(frame):,}")
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
