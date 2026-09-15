"""Place every merged incident on one of the official 774 LGAs.

Same evidence hierarchy as `assign_lgas.py`, applied to the merged corpus rather
than to the main log alone. Coordinates are resolved by point in polygon, names
through the state-constrained crosswalk, and the two are compared rather than one
being assumed to outrank the other. Where they disagree the coordinate is kept if
it falls inside the state named in the record, and the name is kept if the
coordinate falls outside that state while the name resolves cleanly inside it.

That rule exists because of a real failure: thirty-four records carrying State
"Benue" and LGA "Makurdi", both correct, also carry a latitude of 12.19 that is
impossible in Benue. Trusting the coordinate moved them to Kano silently.

The additional file makes this matter more, not less. Only 56.3% of its rows
carry coordinates against 93.6% of the main log, so a larger share of the merged
corpus depends on names and on the settlement gazetteer.
"""

from __future__ import annotations

import argparse
import json
import unicodedata
import zipfile
from pathlib import Path

import pandas as pd
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

SNAP_TOLERANCE_DEGREES = 0.025
NIGERIA_BBOX = (2.5, 3.9, 14.8, 14.1)
MIN_GAZETTEER_NAME = 5

STATE_ALIASES = {
    "fct": "federalcapitalterritory",
    "fctabuja": "federalcapitalterritory",
    "abuja": "federalcapitalterritory",
}


def normalise(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.replace("’", "'").replace("‘", "'").lower().strip()
    return "".join(ch for ch in text if ch.isalnum())


def state_key(value: object) -> str:
    key = normalise(value)
    return STATE_ALIASES.get(key, key)


def load_boundaries(zip_path: Path):
    with zipfile.ZipFile(zip_path) as bundle:
        with bundle.open("nga_admin2.geojson") as handle:
            collection = json.load(handle)
    geometries, attributes = [], []
    for feature in collection["features"]:
        if feature.get("geometry") is None:
            continue
        properties = feature["properties"]
        geometries.append(shape(feature["geometry"]))
        attributes.append({
            "pcode": properties.get("adm2_pcode"),
            "lga": properties.get("adm2_name"),
            "state": properties.get("adm1_name"),
        })
    return geometries, attributes


def locate(point, tree, geometries, attributes):
    for candidate in tree.query(point):
        index = int(candidate)
        if geometries[index].contains(point):
            return attributes[index]
    nearest = int(tree.nearest(point))
    if geometries[nearest].distance(point) <= SNAP_TOLERANCE_DEGREES:
        return attributes[nearest]
    return None


def build_gazetteer(xlsx: Path) -> dict[str, tuple[float, float]]:
    gazetteer: dict[str, tuple[float, float]] = {}
    for sheet in ("nga_admincapitals", "nga_adminpoints"):
        frame = pd.read_excel(xlsx, sheet_name=sheet)
        name_columns = [c for c in frame.columns if c.startswith("name")]
        for record in frame.itertuples():
            x, y = getattr(record, "x_coord", None), getattr(record, "y_coord", None)
            if x is None or y is None or pd.isna(x) or pd.isna(y):
                continue
            for column in name_columns:
                name = getattr(record, column, None)
                if name is None or pd.isna(name) or str(name) == "<Null>":
                    continue
                key = normalise(name)
                if len(key) >= MIN_GAZETTEER_NAME:
                    gazetteer.setdefault(key, (float(x), float(y)))
    return gazetteer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged", default="data/processed/incidents_merged.csv")
    parser.add_argument("--geometry", default="data/reference/nga_admin_boundaries.geojson.zip")
    parser.add_argument("--reference", default="data/reference/nga_admin_boundaries.xlsx")
    parser.add_argument("--crosswalk", default="data/reference/lga_crosswalk.csv")
    parser.add_argument("--out", default="data/processed/incidents_final.csv")
    args = parser.parse_args()

    incidents = pd.read_csv(args.merged, parse_dates=["date"]).reset_index(drop=True)
    geometries, attributes = load_boundaries(Path(args.geometry))
    tree = STRtree(geometries)

    crosswalk = pd.read_csv(args.crosswalk)
    crosswalk = crosswalk[crosswalk["pcode"].notna()]
    name_lookup = {
        (state_key(r.state_in_file), normalise(r.label_in_file)):
            {"pcode": r.pcode, "lga": r.official_name, "state": r.official_state}
        for r in crosswalk.itertuples()
    }
    # The official names themselves, so the additional file's labels resolve even
    # though they never appeared in the main log's crosswalk.
    official_lookup = {
        (state_key(a["state"]), normalise(a["lga"])): a for a in attributes
    }

    latitudes = pd.to_numeric(incidents["latitude"], errors="coerce")
    longitudes = pd.to_numeric(incidents["longitude"], errors="coerce")
    min_lon, min_lat, max_lon, max_lat = NIGERIA_BBOX
    has_coordinate = latitudes.between(min_lat, max_lat) & longitudes.between(min_lon, max_lon)
    print(f"records: {len(incidents):,} | usable coordinates: {int(has_coordinate.sum()):,}")

    rows = []
    for row_id in incidents.index:
        declared = state_key(incidents.at[row_id, "state_raw"])

        geometry_hit = None
        if has_coordinate[row_id]:
            geometry_hit = locate(
                Point(float(longitudes[row_id]), float(latitudes[row_id])),
                tree, geometries, attributes)

        key = (declared, normalise(incidents.at[row_id, "lga_raw"]))
        name_hit = name_lookup.get(key) or official_lookup.get(key)

        chosen, basis = None, "none"
        if geometry_hit and name_hit:
            if geometry_hit["pcode"] == name_hit["pcode"]:
                chosen, basis = geometry_hit, "agree"
            elif state_key(geometry_hit["state"]) == declared:
                chosen, basis = geometry_hit, "conflict_kept_geometry"
            elif state_key(name_hit["state"]) == declared:
                chosen, basis = name_hit, "conflict_kept_name"
            else:
                basis = "conflict_unresolved"
        elif geometry_hit:
            chosen, basis = geometry_hit, "geometry_only"
        elif name_hit:
            chosen, basis = name_hit, "name_only"

        rows.append({"row_id": row_id, "basis": basis,
                     **(chosen or {"pcode": None, "lga": None, "state": None})})

    assigned = pd.DataFrame(rows).set_index("row_id")

    gazetteer = build_gazetteer(Path(args.reference))
    ordered = sorted(gazetteer.items(), key=lambda kv: -len(kv[0]))
    recovered = 0
    for row_id in assigned.index[assigned["pcode"].isna()]:
        declared = state_key(incidents.at[row_id, "state_raw"])
        found = None
        for field in ("location", "narrative"):
            blob = normalise(incidents.at[row_id, field])
            if not blob:
                continue
            for name_key, (x, y) in ordered:
                if name_key in blob:
                    hit = locate(Point(x, y), tree, geometries, attributes)
                    if hit and state_key(hit["state"]) == declared:
                        found = hit
                        break
            if found:
                break
        if found:
            assigned.loc[row_id, ["pcode", "lga", "state"]] = [
                found["pcode"], found["lga"], found["state"]]
            assigned.loc[row_id, "basis"] = "gazetteer"
            recovered += 1

    output = incidents.join(assigned, how="left")
    output["resolved"] = output["pcode"].notna()

    total, got = len(output), int(output["resolved"].sum())
    print(f"\nRESOLVED {got:,} of {total:,} ({100*got/total:.2f}%)")
    print(f"distinct LGAs reached: {output['pcode'].nunique()} of 774")
    print("\nbasis:")
    print(output["basis"].value_counts().to_string())
    print(f"(gazetteer recovered {recovered:,})")

    model = output[output["in_model_period"] & output["is_target"] & output["resolved"]]
    print(f"\nmodelling window target events resolved: {len(model):,}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.out, index=False)
    print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
