"""Assign every incident to one of the official 774 LGAs, weighing the evidence.

An earlier version of this script trusted coordinates above everything else. That
was wrong, and the data showed why. Thirty-four records carry State "Benue" and
Local Government Area "Makurdi", which agree with each other and are correct, but
a latitude of 12.19 that is impossible anywhere in Benue. Point-in-polygon
therefore moved thirty-four Benue incidents into Kano, silently and plausibly.

So the two kinds of evidence are resolved independently and then compared, rather
than one being assumed to outrank the other.

**Geometry** places the coordinate inside an official boundary.

**Naming** resolves the State and Local Government Area text through the
state-constrained crosswalk.

Where both are available and agree, the assignment is confident. Where only one
is available, it is used and labelled. Where they disagree, the rule is:

- if the coordinate falls inside the state named in the file, the coordinate is
  kept, because the LGA text was probably a town or a misspelling;
- if the coordinate falls outside the state named in the file while the name
  resolves cleanly inside it, the name is kept and the coordinate treated as
  corrupt, because one bad number is likelier than two mutually consistent text
  fields both being wrong;
- if neither reading is self-consistent, the record is marked as a conflict and
  left for review rather than assigned.

A fourth pass handles records with no coordinate and no usable LGA name by
looking for a settlement name in the Location and narrative fields. The OCHA
administrative capitals and admin points layers give a name to coordinate
gazetteer, so a town name becomes a point and the point becomes an LGA.

Records that carry neither a coordinate, nor a resolvable LGA name, nor a
recognisable settlement are reported unresolved. They are not guessed at.
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

SNAP_TOLERANCE_DEGREES = 0.025          # roughly 2.5 km
NIGERIA_BBOX = (2.5, 3.9, 14.8, 14.1)   # min lon, min lat, max lon, max lat
MIN_GAZETTEER_NAME = 5                  # ignore very short names, too many false hits

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


def locate(point: Point, tree: STRtree, geometries, attributes):
    """Return the LGA containing the point, or the nearest within tolerance."""
    for candidate in tree.query(point):
        index = int(candidate)
        if geometries[index].contains(point):
            return attributes[index], "contained"
    nearest = int(tree.nearest(point))
    if geometries[nearest].distance(point) <= SNAP_TOLERANCE_DEGREES:
        return attributes[nearest], "snapped"
    return None, "outside"


def build_gazetteer(xlsx: Path) -> dict[str, tuple[float, float]]:
    """Settlement name to coordinate, from the OCHA capitals and points layers."""
    gazetteer: dict[str, tuple[float, float]] = {}
    for sheet in ("nga_admincapitals", "nga_adminpoints"):
        frame = pd.read_excel(xlsx, sheet_name=sheet)
        name_columns = [c for c in frame.columns if c.startswith("name")]
        for record in frame.itertuples():
            x = getattr(record, "x_coord", None)
            y = getattr(record, "y_coord", None)
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
    parser.add_argument("--incidents", default="data/raw/DATA_SOURCE.xlsx")
    parser.add_argument("--geometry", default="data/reference/nga_admin_boundaries.geojson.zip")
    parser.add_argument("--reference", default="data/reference/nga_admin_boundaries.xlsx")
    parser.add_argument("--crosswalk", default="data/reference/lga_crosswalk.csv")
    parser.add_argument("--out", default="data/processed/incidents_assigned.csv")
    args = parser.parse_args()

    incidents = pd.read_excel(args.incidents).reset_index(drop=True)
    incidents["row_id"] = incidents.index
    geometries, attributes = load_boundaries(Path(args.geometry))
    tree = STRtree(geometries)
    print(f"boundaries: {len(geometries)} LGA polygons")

    crosswalk = pd.read_csv(args.crosswalk)
    crosswalk = crosswalk[crosswalk["pcode"].notna()]
    name_lookup = {
        (state_key(r.state_in_file), normalise(r.label_in_file)):
            {"pcode": r.pcode, "lga": r.official_name, "state": r.official_state}
        for r in crosswalk.itertuples()
    }

    latitudes = pd.to_numeric(incidents["Latitude"], errors="coerce")
    longitudes = pd.to_numeric(incidents["Longitude"], errors="coerce")
    min_lon, min_lat, max_lon, max_lat = NIGERIA_BBOX
    has_coordinate = latitudes.between(min_lat, max_lat) & longitudes.between(min_lon, max_lon)

    rows = []
    tally = {"agree": 0, "geometry_only": 0, "name_only": 0,
             "conflict_kept_geometry": 0, "conflict_kept_name": 0,
             "conflict_unresolved": 0, "none": 0}

    for row_id in incidents.index:
        declared_state = state_key(incidents.at[row_id, "State"])

        geometry_hit = None
        if has_coordinate[row_id]:
            point = Point(float(longitudes[row_id]), float(latitudes[row_id]))
            geometry_hit, _ = locate(point, tree, geometries, attributes)

        name_hit = name_lookup.get(
            (declared_state, normalise(incidents.at[row_id, "Local Govt Area"]))
        )

        chosen, basis = None, None
        if geometry_hit and name_hit:
            if geometry_hit["pcode"] == name_hit["pcode"]:
                chosen, basis = geometry_hit, "agree"
            elif state_key(geometry_hit["state"]) == declared_state:
                chosen, basis = geometry_hit, "conflict_kept_geometry"
            elif state_key(name_hit["state"]) == declared_state:
                chosen, basis = name_hit, "conflict_kept_name"
            else:
                chosen, basis = None, "conflict_unresolved"
        elif geometry_hit:
            chosen, basis = geometry_hit, "geometry_only"
        elif name_hit:
            chosen, basis = name_hit, "name_only"
        else:
            basis = "none"

        tally[basis] += 1
        rows.append({"row_id": row_id, "basis": basis,
                     **(chosen or {"pcode": None, "lga": None, "state": None})})

    assigned = pd.DataFrame(rows).set_index("row_id")

    # ------------------------------------------------ pass 4, settlement names
    gazetteer = build_gazetteer(Path(args.reference))
    ordered = sorted(gazetteer.items(), key=lambda kv: -len(kv[0]))
    print(f"gazetteer: {len(gazetteer):,} settlement names")

    recovered = 0
    for row_id in assigned.index[assigned["pcode"].isna()]:
        declared_state = state_key(incidents.at[row_id, "State"])
        found = None
        for field in ("Location", "Details"):
            blob = normalise(incidents.at[row_id, field])
            if not blob:
                continue
            for name_key, (x, y) in ordered:
                if name_key in blob:
                    hit, _ = locate(Point(x, y), tree, geometries, attributes)
                    if hit and state_key(hit["state"]) == declared_state:
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
    print(f"\nRESOLVED {got:,} of {total:,} ({100*got/total:.2f}%)   unresolved {total-got:,}")
    print(f"distinct LGAs reached: {output['pcode'].nunique()} of 774")
    print("\nbasis of assignment:")
    for key, count in output["basis"].value_counts().items():
        print(f"   {key:26s} {count:>7,}")
    print(f"   (gazetteer recovered {recovered:,} in the final pass)")

    conflicts = output[output["basis"].str.startswith("conflict")]
    if len(conflicts):
        print(f"\nconflicts between coordinate and name: {len(conflicts):,}")
        for (a, b), n in conflicts.groupby(["State", "state"]).size().nlargest(8).items():
            print(f"   {n:>5,}  file says {a}, resolved to {b}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    keep = ["row_id", "Date of Incident", "Incident Type", "Incident Type 2", "State",
            "Local Govt Area", "Location", "Latitude", "Longitude", "Fatalities",
            "Kidnap", "pcode", "lga", "state", "basis", "resolved"]
    output[[c for c in keep if c in output.columns]].to_csv(args.out, index=False)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
