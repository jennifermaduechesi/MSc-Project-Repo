# Seven, fourteen and twenty-eight day violence risk, Nigerian local government areas.
#
# The interface specified in section 3.8 of the dissertation, built to the same layout,
# palette and typography as the deployed version of this project's earlier system, so the
# two read as one piece of work. The stylesheet in www/style.css is that system's own,
# carried over unchanged. Two things differ. The data is this project's corpus, and the
# forecast is offered over three windows rather than one.
#
# The application fits nothing. It reads a scored table produced by
# scripts/build_app_data.py, which is what keeps the deployed artefact from drifting away
# from the models the dissertation evaluates.
#
# Three rules from section 3.8 are enforced here rather than stated in a caption. No tier
# is rendered green and no label uses the words safe, clear or secure. Every area is
# reachable, by the ranked list, by the map and by direct lookup, so nothing is absent
# from which a reader could infer an assessment. And each tier carries its measured event
# rate beside it.

library(shiny)
library(leaflet)
library(DT)
library(dplyr)
library(jsonlite)
library(readr)
library(sf)
library(tidyr)

DATA <- "data"
forecast   <- read_csv(file.path(DATA, "forecast.csv"), show_col_types = FALSE)
drivers    <- read_csv(file.path(DATA, "drivers.csv"), show_col_types = FALSE)
incidents  <- read_csv(file.path(DATA, "recent_incidents.csv"), show_col_types = FALSE)
band_stats <- read_csv(file.path(DATA, "band_stats.csv"), show_col_types = FALSE)
meta       <- fromJSON(file.path(DATA, "meta.json"))
shapes     <- st_read(file.path(DATA, "lga.geojson"), quiet = TRUE)

TIERS <- c("Low", "Elevated", "High", "Severe")
TIER_FILL <- c(Low = "#4575B4", Elevated = "#91BFDB",
               High = "#FC8D59", Severe = "#D73027")
HORIZONS <- c("7 days" = 7, "14 days" = 14, "28 days" = 28)

states <- sort(unique(forecast$state))
areas_all <- sort(unique(paste0(forecast$lga, ", ", forecast$state)))

pill <- function(tier) {
  sprintf('<span class="pill pill-%s">%s</span>', tier, tier)
}

# ------------------------------------------------------------------------------ ui
ui <- fluidPage(
  tags$head(
    tags$link(rel = "stylesheet", type = "text/css", href = "style.css"),
    tags$title("Violence risk, Nigerian local government areas")
  ),

  # The cover sits in the initial HTML so it is on screen from the first paint, for the
  # reason the stylesheet gives: a cold instance spends several seconds waking before it
  # can send anything, and the page would otherwise sit blank.
  tags$div(
    id = "app-loading",
    tags$div(
      class = "loading-card",
      tags$div(class = "loading-title", "Violence risk"),
      tags$div(class = "loading-note", "Scoring 774 local government areas"),
      tags$div(class = "loading-bar", tags$div(class = "loading-bar-fill")),
      tags$div(class = "loading-hint",
               "The first visit of the day wakes the server, which takes a few seconds.")
    )
  ),
  tags$script(HTML(
    "$(document).on('shiny:idle', function(){
       var c = document.getElementById('app-loading');
       if (c && !c.classList.contains('is-hidden')) {
         c.classList.add('is-hidden');
         setTimeout(function(){ c.style.display = 'none'; }, 400);
       }
     });")),

  h2("Seven, fourteen and twenty-eight day violence risk, Nigerian local government areas"),

  fluidRow(
    column(
      3,
      div(
        class = "well",
        radioButtons("horizon", "Forecast window", choices = HORIZONS,
                     selected = 7, inline = TRUE),
        div(class = "footnote", style = "margin-top:-6px;margin-bottom:14px;",
            paste("Seven days is the window the system is built around and the one it",
                  "scores best on once the base rate is taken into account.")),
        selectInput("state", "State", choices = c("All states", states)),
        checkboxGroupInput("tiers", "Risk tier", choices = TIERS, selected = TIERS),
        sliderInput("top_n", "Areas listed", min = 10, max = 100, value = 25, step = 5),
        selectizeInput("area", "Look up any area", choices = NULL,
                       options = list(placeholder = "Type an area name")),
        tags$hr(),
        uiOutput("run_metadata")
      ),
      uiOutput("headline_metrics")
    ),

    column(
      6,
      div(class = "map-frame", leafletOutput("map", height = 520)),
      div(class = "footnote",
          "Blue marks the lowest risk and red the most severe. Every area is shaded, because
           every area is scored."),
      tags$hr(),
      h4("Ranked areas"),
      DTOutput("table"),
      tags$hr(),
      h4("How the three windows compare"),
      div(class = "footnote", style = "margin-bottom:10px;",
          "An area that is Low at seven days and High at twenty-eight is one where the
           evidence points to risk that is building rather than imminent."),
      DTOutput("shift_table")
    ),

    column(
      3,
      h4(textOutput("selected_title")),
      uiOutput("selected_summary"),
      h4("Why this score"),
      uiOutput("drivers"),
      h4("Actors recorded here"),
      uiOutput("actors"),
      h4("Recent incidents"),
      uiOutput("incidents")
    )
  ),

  tags$hr(),
  fluidRow(column(12, div(class = "footnote", uiOutput("closing_note"))))
)

# -------------------------------------------------------------------------- server
server <- function(input, output, session) {

  updateSelectizeInput(session, "area", choices = areas_all, selected = "", server = TRUE)

  current <- reactive({
    forecast |> filter(horizon_days == as.numeric(input$horizon)) |> arrange(rank)
  })

  stats_now <- reactive({
    band_stats |> filter(horizon_days == as.numeric(input$horizon))
  })

  info_now <- reactive(meta[[as.character(input$horizon)]])

  filtered <- reactive({
    out <- current()
    if (input$state != "All states") out <- out |> filter(state == input$state)
    if (length(input$tiers)) out <- out |> filter(band %in% input$tiers)
    out
  })

  chosen <- reactiveVal(NULL)
  observeEvent(input$area, {
    if (!is.null(input$area) && nzchar(input$area)) {
      parts <- strsplit(input$area, ", ", fixed = TRUE)[[1]]
      hit <- current() |> filter(lga == parts[1], state == parts[2]) |> slice(1)
      if (nrow(hit)) chosen(hit$pcode)
    }
  }, ignoreInit = TRUE)
  observeEvent(input$map_shape_click, chosen(input$map_shape_click$id))
  observeEvent(input$table_rows_selected, {
    rows <- filtered()
    if (length(input$table_rows_selected)) chosen(rows$pcode[input$table_rows_selected])
  })

  selected <- reactive({
    if (is.null(chosen())) return(current() |> slice(1))
    current() |> filter(pcode == chosen()) |> slice(1)
  })

  # ------------------------------------------------------------------- left column
  output$run_metadata <- renderUI({
    info <- info_now()
    HTML(sprintf(
      "<div class='footnote'>Forecast week beginning <b>%s</b>.<br>
       Last completed week %s.<br>
       Tier boundaries sit at 2, 4 and 8 times a base rate of %.4f, measured over the
       %d completed weeks before the forecast that share its recording coverage.</div>",
      meta$forecast_week, meta$last_completed_week, info$anchor_rate, info$anchor_weeks))
  })

  output$headline_metrics <- renderUI({
    d <- current(); s <- stats_now()
    metric <- function(value, caption, cls) {
      div(class = paste("metric", cls),
          div(class = "value", value), div(class = "caption", caption))
    }
    hit_rate <- function(tiers) {
      rows <- s[s$band %in% tiers, ]
      if (!nrow(rows) || all(is.na(rows$realised_rate))) return("")
      # Pool over the tiers the card counts, weighting each by the areas it holds in an
      # ordinary week. A card covering three tiers has to quote the rate for all three,
      # not for whichever one it happens to name.
      r <- sum(rows$areas_per_week * rows$realised_rate) / sum(rows$areas_per_week)
      paste(if (r < 0.1) sprintf("%.1f%%", 100 * r) else sprintf("%.0f%%", 100 * r),
            "recorded an event")
    }
    tagList(
      metric(sum(d$band == "Severe"), paste("severe.", hit_rate("Severe")), "metric-severe"),
      metric(sum(d$band == "High"), paste("high.", hit_rate("High")), "metric-high"),
      metric(sum(d$band %in% c("Severe", "High", "Elevated")),
             paste("elevated or above.", hit_rate(c("Severe", "High", "Elevated"))),
             "metric-total"),
      div(class = "footnote",
          sprintf("The remaining %d areas are Low, where %s.",
                  sum(d$band == "Low"), hit_rate("Low")))
    )
  })

  # -------------------------------------------------------------------------- map
  output$map <- renderLeaflet({
    leaflet() |>
      addProviderTiles(providers$CartoDB.PositronNoLabels) |>
      setView(lng = 8.7, lat = 9.1, zoom = 6)
  })

  observe({
    keep <- filtered()
    if (nrow(keep) == 0) { leafletProxy("map") |> clearShapes(); return(invisible(NULL)) }
    shp <- merge(shapes[shapes$pcode %in% keep$pcode, ],
                 keep[, c("pcode", "band", "probability", "rank", "lga", "state")],
                 by = "pcode", all.x = TRUE)
    labels <- sprintf(
      "<b>%s</b><br>%s<br>%s &middot; rank %d of %d<br>%.1f%% chance within %s days",
      shp$lga.y, shp$state.y, shp$band, shp$rank, meta$areas,
      100 * shp$probability, input$horizon)
    leafletProxy("map", data = shp) |>
      clearShapes() |>
      addPolygons(
        layerId = ~pcode, weight = 0.4, color = "#FFFFFF", opacity = 1,
        fillColor = unname(TIER_FILL[shp$band]), fillOpacity = 0.85,
        highlightOptions = highlightOptions(weight = 2, color = "#14161A",
                                            bringToFront = TRUE),
        label = lapply(labels, HTML),
        labelOptions = labelOptions(className = "lga-tooltip", html = TRUE))
  })

  # ---------------------------------------------------------------- ranked areas
  output$table <- renderDT({
    filtered() |>
      head(input$top_n) |>
      transmute(Rank = rank, Area = lga, State = state,
                Tier = pill(band), Probability = sprintf("%.1f%%", 100 * probability)) |>
      datatable(rownames = FALSE, escape = FALSE, selection = "single",
                options = list(pageLength = 12, dom = "tp", scrollX = TRUE,
                               columnDefs = list(list(className = "dt-right",
                                                      targets = c(0, 4)))))
  })

  output$shift_table <- renderDT({
    wide <- forecast |>
      select(pcode, lga, state, horizon_days, band, rank) |>
      pivot_wider(names_from = horizon_days, values_from = c(band, rank))
    idx <- function(b) match(b, TIERS)
    wide |>
      filter(idx(band_7) != idx(band_28)) |>
      mutate(Direction = ifelse(idx(band_28) > idx(band_7),
                                "rises by 28 days", "falls by 28 days")) |>
      arrange(rank_7) |>
      transmute(Area = lga, State = state,
                `7 days` = pill(band_7), `14 days` = pill(band_14),
                `28 days` = pill(band_28),
                `Rank at 7` = rank_7, `Rank at 28` = rank_28, Direction) |>
      datatable(rownames = FALSE, escape = FALSE, selection = "none",
                options = list(pageLength = 8, dom = "tp", scrollX = TRUE))
  })

  # ------------------------------------------------------------------ right column
  output$selected_title <- renderText({
    a <- selected(); if (!nrow(a)) return("")
    paste0(a$lga, ", ", a$state)
  })

  output$selected_summary <- renderUI({
    a <- selected(); if (!nrow(a)) return(NULL)
    s <- stats_now(); r <- s$realised_rate[s$band == a$band]
    HTML(sprintf(
      "<div class='metric'><div class='value'>%.1f%%</div>
       <div class='caption'>chance of at least one event within %s days</div></div>
       <p>%s &middot; ranked %d of %d nationally.</p>
       <p class='footnote'>%s A tier says how this area compares with an ordinary
       area-week. It is not a verdict on the area.</p>",
      100 * a$probability, input$horizon, pill(a$band), a$rank, meta$areas,
      if (length(r) && !is.na(r))
        sprintf("Across the test years, %s of area-weeks in this tier recorded an event.",
                if (r < 0.1) sprintf("%.1f%%", 100 * r) else sprintf("%.0f%%", 100 * r))
      else ""))
  })

  output$drivers <- renderUI({
    a <- selected(); if (!nrow(a)) return(NULL)
    d <- drivers |>
      filter(pcode == a$pcode, horizon_days == as.numeric(input$horizon)) |>
      arrange(desc(abs(contribution)))
    if (!nrow(d)) return(div(class = "footnote", "No drivers recorded for this area."))
    rows <- paste0(
      "<tr><td>", d$label, "</td><td>",
      ifelse(is.na(d$value), "<span class='footnote'>grouped</span>",
             formatC(d$value, format = "f", digits = 2)), "</td><td>",
      ifelse(d$contribution > 0, "raises", "lowers"), "</td></tr>", collapse = "")
    HTML(paste0(
      "<div id='drivers'><table class='table'><thead><tr><th>Reading</th>",
      "<th>Value</th><th>Effect</th></tr></thead><tbody>", rows, "</tbody></table></div>",
      "<p class='footnote'>These are the signed contributions of the penalised linear
       member of the ensemble, which decompose exactly on the log-odds scale. They
       explain that member's view rather than the whole ensemble's.</p>
       <p class='footnote'>Readings that measure nearly the same thing are grouped,
       because their separate weights are not interpretable even though their sum is.
       The windows that remain separate still overlap, since a four-week count contains
       the two-week count inside it, and in a small number of areas two overlapping
       readings point in opposite directions. Read the direction of the group rather
       than of any single line.</p>"))
  })

  output$actors <- renderUI({
    a <- selected(); if (!nrow(a)) return(NULL)
    act <- incidents |>
      filter(pcode == a$pcode, !is.na(perpetrator), nzchar(perpetrator)) |>
      count(perpetrator, sort = TRUE)
    if (!nrow(act))
      return(div(class = "footnote",
                 "No actor recorded here in the last eight weeks. That is a statement
                  about the record, not about who is present."))
    rows <- paste0("<tr><td>", act$perpetrator, "</td><td>", act$n, "</td></tr>",
                   collapse = "")
    HTML(paste0("<div id='actors'><table class='table'><thead><tr><th>Group</th>",
                "<th>Incidents</th></tr></thead><tbody>", rows,
                "</tbody></table></div>",
                "<p class='footnote'>Attribution describes events already recorded. It is
                 not carried into the forecast.</p>"))
  })

  output$incidents <- renderUI({
    a <- selected(); if (!nrow(a)) return(NULL)
    inc <- incidents |> filter(pcode == a$pcode) |> head(12)
    if (!nrow(inc))
      return(div(class = "footnote",
                 "None recorded in the last eight weeks. That is a statement about the
                  record, not about what happened."))
    rows <- paste0("<tr><td>", inc$date, "</td><td>", inc$location, "</td><td>",
                   inc$deaths, "</td><td>", inc$kidnapped, "</td></tr>", collapse = "")
    HTML(paste0("<div id='incidents'><table class='table'><thead><tr><th>Date</th>",
                "<th>Location</th><th>Killed</th><th>Taken</th></tr></thead><tbody>",
                rows, "</tbody></table></div>"))
  })

  output$closing_note <- renderUI({
    s <- stats_now(); info <- info_now()
    low <- s$share_of_events[s$band == "Low"]
    HTML(sprintf(
      "Probabilities are calibrated over a %s day horizon and are an aid to prioritisation,
       not a forecast of individual incidents. An area outside the list is not assessed as
       safe: across the test years %s of all recorded events fell in areas this system had
       placed in the Low tier. Tier boundaries are multiples of the base rate of the window
       shown, so a tier carries the same meaning at seven days as at twenty-eight. The
       measured rates beside each tier come from the calibrated linear member, because the
       comparison across windows holds the model fixed, while the probabilities come from
       the full ensemble. Those rates were measured in test weeks whose base rate was %.4f.
       The weeks anchoring this forecast sit at %.4f, a %s recording period, so read them
       as the ordering a tier implies rather than as a rate to expect this week.",
      input$horizon,
      if (length(low) && !is.na(low)) sprintf("%.1f per cent", 100 * low) else "a large share",
      s$anchor_rate[1], info$anchor_rate,
      if (info$anchor_rate < s$anchor_rate[1]) "quieter" else "busier"))
  })
}

shinyApp(ui, server)
