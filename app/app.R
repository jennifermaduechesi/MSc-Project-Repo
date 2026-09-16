# Seven, fourteen and twenty-eight day violence risk, Nigerian Local Government Areas.
#
# The interface specified in section 3.8 of the dissertation. It reads a scored table
# produced by scripts/build_app_data.py and fits nothing itself, which keeps the deployed
# artefact from drifting away from the models the dissertation evaluates.
#
# Three design rules from section 3.8 are enforced here rather than stated in a caption.
# No band is rendered green and no label anywhere uses the words safe, clear or secure.
# Every area is reachable by direct lookup, so nothing is absent from which a user could
# infer an assessment. And the realised event rate for each band is displayed beside the
# band, so the reader sees what Low actually means rather than guessing.

library(shiny)
library(bslib)
library(leaflet)
library(DT)
library(dplyr)
library(jsonlite)
library(readr)
library(sf)

# ----------------------------------------------------------------------- data
DATA <- "data"
forecast   <- read_csv(file.path(DATA, "forecast.csv"), show_col_types = FALSE)
drivers    <- read_csv(file.path(DATA, "drivers.csv"), show_col_types = FALSE)
incidents  <- read_csv(file.path(DATA, "recent_incidents.csv"), show_col_types = FALSE)
history    <- read_csv(file.path(DATA, "history.csv"), show_col_types = FALSE)
meta       <- fromJSON(file.path(DATA, "meta.json"))
shapes     <- st_read(file.path(DATA, "lga.geojson"), quiet = TRUE)
band_stats <- read_csv(file.path(DATA, "band_stats.csv"), show_col_types = FALSE)

HORIZONS <- c("7 days" = 7, "14 days" = 14, "28 days" = 28)
BANDS <- c("Severe", "High", "Elevated", "Low")

# No green anywhere. A green band would read as an assurance the model cannot give.
BAND_FILL <- c(Severe = "#C1362F", High = "#E8894A",
               Elevated = "#F2C879", Low = "#B8C4D9")
BAND_TEXT <- c(Severe = "#FFFFFF", High = "#3A1F10",
               Elevated = "#4A3A12", Low = "#2B3240")

states <- sort(unique(forecast$state))

# --------------------------------------------------------------------- theme
# A system font stack rather than font_google(). Fetching a web font at start-up makes
# the application's first load depend on an outbound request, which hangs where that
# request is blocked and adds a failure mode on deployment for no benefit the reader
# would notice.
CLAY_FONT <- c("Nunito", "Segoe UI", "Helvetica Neue", "Arial", "sans-serif")
clay <- bs_theme(
  version = 5,
  bg = "#EEF1F6", fg = "#25303F",
  primary = "#5B6E8C", base_font = CLAY_FONT, heading_font = CLAY_FONT
)

CLAY_CSS <- "
.clay { background:#F7F9FC; border-radius:22px; padding:18px 20px;
        box-shadow: 8px 8px 18px #D3DAE6, -8px -8px 18px #FFFFFF; margin-bottom:18px; }
.clay-tight { padding:12px 16px; }
.band-pill { display:inline-block; padding:5px 14px; border-radius:14px;
             font-weight:700; font-size:0.95rem; }
.bignum { font-size:2.6rem; font-weight:800; line-height:1.05; }
.muted  { color:#66748A; font-size:0.88rem; }
.warn   { background:#FDF3E7; border-left:5px solid #E8894A; border-radius:14px;
          padding:12px 16px; margin-bottom:16px; font-size:0.92rem; }
.leaflet-container { border-radius:18px; }
"

pill <- function(band) {
  sprintf('<span class="band-pill" style="background:%s;color:%s">%s</span>',
          BAND_FILL[[band]], BAND_TEXT[[band]], band)
}

# ------------------------------------------------------------------------- ui
ui <- page_sidebar(
  theme = clay,
  title = "Seven to twenty-eight day violence risk, Nigerian Local Government Areas",
  tags$head(tags$style(HTML(CLAY_CSS))),

  sidebar = sidebar(
    width = 300,
    radioButtons("horizon", "Forecast window",
                 choices = HORIZONS, selected = 7),
    helpText(HTML(paste0(
      "Opens on seven days. The evaluation in the methodology chapter finds the ",
      "seven-day window best on every measure that is comparable across horizons. ",
      "A longer window scores higher on raw average precision only because the ",
      "target becomes commoner."))),
    hr(),
    selectInput("state", "State", choices = c("All states", states)),
    checkboxGroupInput("bands", "Risk band", choices = BANDS, selected = BANDS),
    hr(),
    selectizeInput("area", "Look up any area", choices = NULL,
                   options = list(placeholder = "Type an area name")),
    helpText(HTML(paste0("Every one of the ", meta$areas,
                         " areas can be looked up, including those in the Low band."))),
    hr(),
    div(class = "muted",
        HTML(paste0("Forecast week beginning <b>", meta$forecast_week, "</b>.<br>",
                    "Last completed week ", meta$last_completed_week, ".")))
  ),

  navset_card_tab(
    nav_panel(
      "Forecast",
      div(class = "warn", HTML(paste0(
        "<b>This is a prioritisation of finite patrol capacity, not a forecast of where ",
        "events will occur.</b> An area in the Low band has not been assessed as safe. ",
        "Across the five test years, 41.9 per cent of all recorded events happened in ",
        "areas this system had placed in the Low band, and in the weakest year that ",
        "share reached 68.5 per cent."))),
      layout_columns(
        col_widths = c(7, 5),
        div(class = "clay", leafletOutput("map", height = 560)),
        div(
          div(class = "clay clay-tight", uiOutput("band_summary")),
          div(class = "clay", h5("Ranked areas"),
              div(class = "muted", "Ordered by probability. The rank is national, not within the filter."),
              DTOutput("ranked"))
        )
      )
    ),

    nav_panel(
      "Compare horizons",
      div(class = "clay", htmlOutput("compare_intro")),
      layout_columns(
        col_widths = c(5, 7),
        div(class = "clay", h5("How the bands fill at each window"), DTOutput("band_compare")),
        div(class = "clay", h5("Areas whose band changes with the window"),
            div(class = "muted", style = "margin-bottom:14px;",
                paste("An area that is Low at seven days and High at twenty-eight is",
                      "one where the evidence points to elevated risk, but not",
                      "imminently.")),
            DTOutput("shift_table"))
      )
    ),

    nav_panel(
      "Area detail",
      uiOutput("detail")
    ),

    nav_panel(
      "How to read this",
      div(class = "clay", htmlOutput("method"))
    )
  )
)

# --------------------------------------------------------------------- server
server <- function(input, output, session) {

  updateSelectizeInput(session, "area",
                       choices = sort(unique(paste0(forecast$lga, ", ", forecast$state))),
                       selected = "", server = TRUE)

  current <- reactive({
    forecast |>
      filter(horizon_days == as.numeric(input$horizon)) |>
      arrange(rank)
  })

  filtered <- reactive({
    out <- current()
    if (input$state != "All states") out <- out |> filter(state == input$state)
    if (length(input$bands)) out <- out |> filter(band %in% input$bands)
    out
  })

  stats_now <- reactive({
    band_stats |> filter(horizon_days == as.numeric(input$horizon))
  })

  info_now <- reactive({
    meta[[as.character(input$horizon)]]
  })

  output$band_summary <- renderUI({
    s <- stats_now()
    info <- info_now()
    rows <- lapply(BANDS, function(b) {
      row <- s |> filter(band == b)
      n <- current() |> filter(band == b) |> nrow()
      realised <- if (nrow(row) && !is.na(row$realised_rate)) {
        sprintf("about 1 area-week in %.0f recorded an event", 1 / row$realised_rate)
      } else "not separately validated at this window"
      div(style = "margin-bottom:9px",
          HTML(pill(b)),
          span(style = "margin-left:10px; font-weight:700", n, "areas"),
          div(class = "muted", style = "margin-left:2px", realised))
    })
    severe_thin <- nrow(s) && !is.na(s$areas_per_week[s$band == "Severe"][1]) &&
      s$areas_per_week[s$band == "Severe"][1] < 1
    tagList(h5("Bands this week"), rows,
            if (isTRUE(severe_thin))
              div(class = "muted", style = "margin-bottom:8px",
                  HTML(paste("At this window the Severe cut is eight times a base rate",
                             "of roughly", sprintf("%.2f", s$anchor_rate[1]),
                             ", which asks for a probability near certainty. Almost no",
                             "area reaches it, so read High as the top band here."))),
            div(class = "muted",
                HTML(sprintf(paste("Boundaries sit at 2, 4 and 8 times the base rate of",
                                   "the %d completed weeks before the forecast that",
                                   "share its recording coverage, which is %.4f at this",
                                   "window. The window stops at the coverage change of",
                                   "1 January 2026 rather than running back a fixed 52",
                                   "weeks, because the rate either side of it is not",
                                   "the same quantity."),
                             info$anchor_weeks, s$anchor_rate[1]))),
            div(class = "muted", style = "margin-top:6px",
                HTML(paste("Realised rates are measured on the",
                           "calibrated linear member across all three windows, because",
                           "the horizon comparison holds the model fixed so that a",
                           "change between windows is a change of window and not of",
                           "algorithm. The probabilities above them come from the full",
                           "ensemble."))))
  })

  output$map <- renderLeaflet({
    leaflet() |>
      addProviderTiles(providers$CartoDB.PositronNoLabels) |>
      setView(lng = 8.7, lat = 9.1, zoom = 6)
  })

  observe({
    keep <- filtered()
    if (nrow(keep) == 0) {
      leafletProxy("map") |> clearShapes()
      return(invisible(NULL))
    }
    shp <- shapes[shapes$pcode %in% keep$pcode, ]
    shp <- merge(shp, keep[, c("pcode", "band", "probability", "rank", "lga", "state")],
                 by = "pcode", all.x = TRUE)
    labels <- sprintf(
      "<b>%s</b><br>%s<br>%s &middot; rank %d of %d<br>%.1f%% chance within %s days",
      shp$lga.y, shp$state.y, shp$band, shp$rank, meta$areas,
      100 * shp$probability, input$horizon)
    leafletProxy("map", data = shp) |>
      clearShapes() |>
      addPolygons(
        layerId = ~pcode,
        weight = 0.5, color = "#7C8899", opacity = 1,
        fillColor = unname(BAND_FILL[shp$band]), fillOpacity = 0.85,
        highlightOptions = highlightOptions(weight = 2, color = "#25303F",
                                            bringToFront = TRUE),
        label = lapply(labels, HTML))
  })

  output$ranked <- renderDT({
    filtered() |>
      transmute(Rank = rank, Area = lga, State = state, Band = band,
                Probability = sprintf("%.1f%%", 100 * probability)) |>
      datatable(rownames = FALSE, selection = "single",
                options = list(pageLength = 12, dom = "tp", scrollX = TRUE)) |>
      formatStyle("Band", backgroundColor = styleEqual(BANDS, unname(BAND_FILL[BANDS])),
                  color = styleEqual(BANDS, unname(BAND_TEXT[BANDS])),
                  fontWeight = "bold")
  })

  observeEvent(input$map_shape_click, {
    hit <- forecast |> filter(pcode == input$map_shape_click$id) |> slice(1)
    if (nrow(hit)) {
      updateSelectizeInput(session, "area",
                           selected = paste0(hit$lga, ", ", hit$state))
    }
  })

  output$compare_intro <- renderUI({
    HTML(paste0(
      "<h5>Why three windows</h5><p>The seven-day window is the one the system is built ",
      "around and the one it performs best on. The longer windows are shown because an ",
      "area's position can change with the horizon, and that change is itself ",
      "information. Each window is banded against its own base rate, which rises from ",
      "roughly 0.036 at seven days to 0.113 at twenty-eight, so a band means the same ",
      "thing at every window: this area is at least two, four or eight times the ",
      "ordinary rate.</p>",
      "<p class='muted'>Average precision rises with the window and lift over the base ",
      "rate falls. A longer window is not a better forecast, it is an easier target.</p>"))
  })

  output$band_compare <- renderDT({
    band_stats |>
      mutate(Window = paste0(horizon_days, " days"),
             Band = band,
             `Areas per week` = round(areas_per_week, 1),
             `Realised rate` = sprintf("%.3f", realised_rate),
             `Share of events` = sprintf("%.1f%%", 100 * share_of_events),
             `Base rate` = sprintf("%.4f", anchor_rate)) |>
      select(Window, Band, `Areas per week`, `Realised rate`,
             `Share of events`, `Base rate`) |>
      datatable(rownames = FALSE, options = list(pageLength = 12, dom = "t"))
  })

  output$shift_table <- renderDT({
    wide <- forecast |>
      select(pcode, lga, state, horizon_days, band, rank) |>
      tidyr::pivot_wider(names_from = horizon_days,
                         values_from = c(band, rank))
    idx <- function(b) match(b, rev(BANDS))
    wide |>
      filter(idx(band_7) != idx(band_28)) |>
      mutate(Direction = ifelse(idx(band_28) > idx(band_7), "rises by 28 days",
                                "falls by 28 days")) |>
      arrange(rank_7) |>
      transmute(Area = lga, State = state, `7 days` = band_7, `14 days` = band_14,
                `28 days` = band_28, `Rank at 7` = rank_7, `Rank at 28` = rank_28,
                Direction) |>
      datatable(rownames = FALSE, options = list(pageLength = 12, dom = "tp", scrollX = TRUE))
  })

  chosen <- reactive({
    req(input$area)
    parts <- strsplit(input$area, ", ", fixed = TRUE)[[1]]
    current() |> filter(lga == parts[1], state == parts[2]) |> slice(1)
  })

  output$detail <- renderUI({
    if (is.null(input$area) || input$area == "") {
      return(div(class = "clay",
                 h5("Look up an area"),
                 p(paste("Use the box in the sidebar. Every area returns a probability",
                         "and a band, including areas in the Low band. Nothing is",
                         "withheld, so an area's absence from the ranked list carries",
                         "no meaning."))))
    }
    a <- chosen()
    if (nrow(a) == 0) return(div(class = "clay", "No area matched."))
    d <- drivers |>
      filter(pcode == a$pcode, horizon_days == as.numeric(input$horizon)) |>
      arrange(desc(abs(contribution)))
    inc <- incidents |> filter(pcode == a$pcode)

    tagList(
      layout_columns(
        col_widths = c(4, 8),
        div(class = "clay",
            div(class = "muted", a$state),
            h4(a$lga),
            div(class = "bignum", sprintf("%.1f%%", 100 * a$probability)),
            div(class = "muted", sprintf("chance of at least one event in the next %s days",
                                         input$horizon)),
            br(), HTML(pill(a$band)),
            div(class = "muted", style = "margin-top:10px",
                sprintf("Ranked %d of %d nationally.", a$rank, meta$areas)),
            hr(),
            div(class = "muted",
                HTML("A band is not a verdict on this area's safety. It says how this
                      area compares with an ordinary area-week."))),
        div(class = "clay",
            h5("What the linear model is reading"),
            div(class = "muted",
                HTML(paste("These are the signed contributions of the penalised linear",
                           "member of the ensemble, which decompose exactly on the",
                           "log-odds scale. They explain that member's view, not the",
                           "whole ensemble's. Producing a faithful decomposition of an",
                           "ensemble containing a graph network is beyond what this",
                           "study attempts."))),
            br(),
            DTOutput("driver_table"))
      ),
      div(class = "clay",
          h5(sprintf("Recorded incidents here in the last 8 weeks (%d)", nrow(inc))),
          if (nrow(inc) == 0)
            div(class = "muted",
                HTML("None recorded. That is a statement about the record, not about
                      what happened."))
          else DTOutput("incident_table"))
    )
  })

  output$driver_table <- renderDT({
    a <- chosen()
    drivers |>
      filter(pcode == a$pcode, horizon_days == as.numeric(input$horizon)) |>
      arrange(desc(abs(contribution))) |>
      transmute(Reading = label, Value = round(value, 3),
                Direction = ifelse(contribution > 0, "raises risk", "lowers risk"),
                Weight = round(contribution, 3)) |>
      datatable(rownames = FALSE, options = list(dom = "t", pageLength = 10))
  })

  output$incident_table <- renderDT({
    a <- chosen()
    incidents |>
      filter(pcode == a$pcode) |>
      transmute(Date = date, Location = location, Killed = deaths,
                Abducted = kidnapped, Group = perpetrator) |>
      datatable(rownames = FALSE, options = list(dom = "tp", pageLength = 8, scrollX = TRUE))
  })

  output$method <- renderUI({
    HTML(paste0(
      "<h5>What this is</h5>",
      "<p>A weekly ranking of all ", meta$areas, " Nigerian Local Government Areas by the ",
      "probability that at least one kidnapping or banditry event is recorded there in ",
      "the window selected. It is built from an incident corpus compiled by Bulwark ",
      "Intelligence and supplied for this study, and from official administrative ",
      "boundaries.</p>",
      "<h5>What the bands mean</h5>",
      "<p>Boundaries are multiples of the base rate over the previous 52 completed weeks. ",
      "Elevated is at least twice the ordinary rate, High at least four times, Severe at ",
      "least eight. The anchor is recomputed for each window, so a band carries the same ",
      "meaning at seven days as at twenty-eight.</p>",
      "<p>A band may be empty. In one of the five test years no area anywhere reached ",
      "eight times the recent rate, and the interface shows an empty Severe band rather ",
      "than promoting the next areas to fill it. A display that always finds something ",
      "severe teaches the people using it to ignore it.</p>",
      "<h5>What it does not do</h5>",
      "<p>It does not predict individual incidents. It does not attribute a forecast to ",
      "a named armed group, because attribution in the record describes events that have ",
      "already happened. It does not report probabilities to a precision the evaluation ",
      "does not support.</p>",
      "<h5>The limitation that matters most</h5>",
      "<p>Across five years of held-out testing, 41.9 per cent of recorded events fell in ",
      "areas banded Low, ranging from 29.9 per cent in the best year to 68.5 per cent in ",
      "the worst. Low means the recorded evidence does not concentrate here this week. It ",
      "does not mean nothing will happen. Roughly 95 per cent of areas hold their band ",
      "from one week to the next, so most of what changes week to week is small.</p>",
      "<p class='muted'>Model: stacked ensemble of a penalised logistic regression, a ",
      "random forest, a gradient boosting ensemble and a recurrent graph network, ",
      "combined by an unweighted logistic meta-learner on the log-odds scale. Scores ",
      "generated ", meta$forecast_week, " from data complete to ",
      meta$last_completed_week, ".</p>"))
  })
}

shinyApp(ui, server)
