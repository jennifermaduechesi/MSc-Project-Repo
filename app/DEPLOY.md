# Deploying this application to shinyapps.io

Account: **jenniferchidi**

## What is in this folder

| Path | What it is |
|---|---|
| `app.R` | The interface |
| `www/style.css` | The stylesheet, carried over from the earlier deployed system |
| `data/forecast.csv` | 774 areas scored at 7, 14 and 28 days |
| `data/drivers.csv` | Per-area feature contributions |
| `data/band_stats.csv` | Measured event rate of each tier at each window |
| `data/recent_incidents.csv` | Incidents recorded in the last eight weeks |
| `data/lga.geojson` | Simplified boundaries for the map |
| `data/meta.json` | Forecast week, anchors, meta-learner weights |

Total 2.1 MB, which is well inside the free tier.

## One decision before you deploy

`data/recent_incidents.csv` carries incident dates, locations, casualty counts and
perpetrator attribution from the Bulwark Intelligence corpus. Deploying publishes it.
Your earlier application already shows the same class of information, so this is probably
the position you have already taken, but it is your call rather than an assumption and it
is worth making deliberately.

If you would rather not publish it, delete that one file before deploying. The interface
handles its absence: the "Recent incidents" panel will say nothing is recorded, and
everything else works unchanged.

## Steps

You need R on a computer, not a phone.

```r
install.packages(c("rsconnect", "shiny", "leaflet", "DT", "dplyr",
                   "jsonlite", "readr", "sf", "tidyr"))

# Use the Copy to clipboard button on your shinyapps.io dashboard for this line,
# so the secret never has to be typed or pasted anywhere else.
rsconnect::setAccountInfo(name   = "jenniferchidi",
                          token  = "<from the dashboard>",
                          secret = "<from the dashboard>")

rsconnect::deployApp("path/to/this/app/folder",
                     appName = "lga-violence-risk-3window")
```

The first deployment takes several minutes, because shinyapps.io builds `sf` and
`leaflet` from source. Later deployments are quicker.

## If the build fails on sf

`sf` needs system geospatial libraries and occasionally times out on the free tier. If
that happens, the map is the only part that needs it, and the rest of the interface will
run without it. Tell me and I will swap the map for a version that does not depend on
`sf`.

## Keeping it current

The forecast is for one week. To refresh it after new data arrives:

```bash
python3 scripts/build_panel.py
python3 scripts/build_app_data.py
```

then redeploy. `scripts/build_app_geometry.py` only needs re-running if the boundary file
changes.
