"""Render a Nigeria risk map for the defence deck, from the app's own forecast data.

The colours are the deck template's, so the map sits inside the same visual language as
the slides. Everything plotted is read from app/data, so the picture cannot state a risk
the application does not.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Polygon as MplPolygon

APP = Path("app/data")
OUT = Path("dissertation")
GREEN, RED, GREY = "#1B4332", "#E4002B", "#565E6C"
# Pale to red, ending on the template's own accent.
CMAP = LinearSegmentedColormap.from_list("risk", ["#F5F7FA", "#F7D9C4", "#EE9A6A", RED])


def rings(geom):
    if geom["type"] == "Polygon":
        return [geom["coordinates"][0]]
    return [p[0] for p in geom["coordinates"]]


def main(horizon: int = 7) -> None:
    gj = json.loads((APP / "lga.geojson").read_text())
    fc = pd.read_csv(APP / "forecast.csv")
    fc = fc[fc.horizon_days == horizon].set_index("pcode")
    prob = fc["probability"].to_dict()

    fig, ax = plt.subplots(figsize=(11.4, 7.0))
    patches, values, missing = [], [], 0
    for feat in gj["features"]:
        code = feat["properties"]["pcode"]
        if code not in prob:
            missing += 1
            continue
        for ring in rings(feat["geometry"]):
            patches.append(MplPolygon(ring, closed=True))
            values.append(prob[code])
    if missing:
        raise SystemExit(f"{missing} areas in the boundary file have no forecast")

    pc = PatchCollection(patches, cmap=CMAP, edgecolor="white", linewidths=0.18)
    pc.set_array(pd.Series(values).to_numpy())
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.axis("off")

    # The highest-risk areas cluster in the north west, so labels placed on the polygons
    # collide. They are stacked down the right instead, each on a leader to its own area.
    xs = [p[0] for pt in patches for p in pt.get_xy()]
    ys = [p[1] for pt in patches for p in pt.get_xy()]
    x_lab = max(xs) + 0.55
    top = fc.nsmallest(5, "rank")
    y_hi, y_lo = max(ys) * 0.97, max(ys) * 0.62
    step = (y_hi - y_lo) / max(len(top) - 1, 1)
    for i, (code, row) in enumerate(top.iterrows()):
        feat = next(f for f in gj["features"] if f["properties"]["pcode"] == code)
        ring = rings(feat["geometry"])[0]
        cx = sum(p[0] for p in ring) / len(ring)
        cy = sum(p[1] for p in ring) / len(ring)
        ax.annotate(f"{row.lga}, {row.state}   {row.probability * 100:.1f}%",
                    xy=(cx, cy), xytext=(x_lab, y_hi - i * step),
                    fontsize=10.5, color=GREEN, fontweight="bold",
                    va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.9,
                                    connectionstyle="arc3,rad=-0.12"),
                    bbox=dict(facecolor="white", edgecolor="#D7DCE3",
                              boxstyle="round,pad=0.30"))
    ax.set_xlim(min(xs) - 0.3, x_lab + 5.6)

    cb = fig.colorbar(pc, ax=ax, fraction=0.022, pad=0.02)
    cb.set_label("Chance of at least one event in the next seven days",
                 fontsize=10.5, color=GREY)
    cb.ax.tick_params(labelsize=9.5, colors=GREY)
    cb.outline.set_visible(False)

    fig.savefig(OUT / "deck_risk_map.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {OUT / 'deck_risk_map.png'}  ({len(patches)} polygons, "
          f"{fc.shape[0]} areas at the {horizon}-day window)")


if __name__ == "__main__":
    main()
