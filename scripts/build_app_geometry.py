"""Simplify the official boundaries into a GeoJSON small enough for the browser.

The Common Operational Dataset polygons are far more detailed than a national risk map
needs, and shipping them whole makes the interface slow to load on the kind of connection
it would actually be used over. Simplification is a display decision only. Nothing in the
modelling touches this file, and the adjacency graph is built from the full-resolution
polygons in `build_adjacency.py`, so a simplified border cannot change a neighbour
relation or a forecast.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

TOLERANCE = 0.008          # degrees, about 880 m at this latitude


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default="data/reference/nga_admin_boundaries.geojson.zip")
    parser.add_argument("--out", default="app/data/lga.geojson")
    args = parser.parse_args()

    with zipfile.ZipFile(args.zip) as archive:
        # The same member `build_adjacency.py` reads, so the map cannot show a
        # different set of areas from the one the graph was built on.
        name = "nga_admin2.geojson"
        raw = json.loads(archive.read(name))
    print(f"read {name}: {len(raw['features'])} features")

    out_features, dropped = [], 0
    for feature in raw["features"]:
        properties = feature["properties"]
        pcode = properties.get("ADM2_PCODE") or properties.get("adm2_pcode")
        lga = properties.get("ADM2_EN") or properties.get("adm2_name")
        state = properties.get("ADM1_EN") or properties.get("adm1_name")
        geometry = shape(feature["geometry"])
        simplified = geometry.simplify(TOLERANCE, preserve_topology=True)
        if simplified.is_empty:
            simplified = geometry            # keep the original rather than lose an area
            dropped += 1
        out_features.append({
            "type": "Feature",
            "properties": {"pcode": pcode, "lga": lga, "state": state},
            "geometry": mapping(simplified),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"type": "FeatureCollection", "features": out_features}, separators=(",", ":")))
    size = Path(args.out).stat().st_size
    print(f"wrote {args.out}: {len(out_features)} areas, {size / 1e6:.1f} MB"
          + (f", {dropped} kept at full resolution" if dropped else ""))
    if len({f["properties"]["pcode"] for f in out_features}) != 774:
        raise SystemExit("expected 774 distinct p-codes in the simplified boundary file")
    print("  774 distinct p-codes confirmed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
