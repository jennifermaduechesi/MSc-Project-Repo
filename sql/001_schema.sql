-- Phase 1 schema: LGA reference, raw incidents, raw text mentions, weekly feature/label table.
-- Requires PostGIS and pgvector extensions (installed via docker/postgres/Dockerfile).

CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS vector;

-- Reference table for Nigeria's 774 LGAs. Populate from an authoritative boundary
-- source (e.g. OCHA Nigeria admin boundaries / NBS), not hand-entered — see
-- docs/data_access_notes.md for sourcing notes.
CREATE TABLE IF NOT EXISTS lga (
    lga_id      TEXT PRIMARY KEY,       -- admin boundary code from source dataset
    lga_name    TEXT NOT NULL,
    state_name  TEXT NOT NULL,
    geom        geometry(MultiPolygon, 4326) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_lga_geom ON lga USING GIST (geom);

-- Structured incident records, primarily from ACLED.
CREATE TABLE IF NOT EXISTS incidents (
    incident_id     TEXT PRIMARY KEY,      -- source event id (e.g. ACLED event_id_cnty)
    lga_id          TEXT REFERENCES lga(lga_id),
    event_date      DATE NOT NULL,
    event_type      TEXT NOT NULL,         -- e.g. 'Violence against civilians', 'Battles'
    event_subtype   TEXT,
    actor1          TEXT,
    actor2          TEXT,
    fatalities      INTEGER DEFAULT 0,
    source          TEXT,                  -- e.g. 'ACLED'
    source_scale    TEXT,
    notes           TEXT,
    geom            geometry(Point, 4326),
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_incidents_lga_date ON incidents (lga_id, event_date);
CREATE INDEX IF NOT EXISTS idx_incidents_geom ON incidents USING GIST (geom);

-- Unstructured text mentions from news / X (Twitter) / Telegram, post spaCy/SBERT processing.
CREATE TABLE IF NOT EXISTS text_mentions (
    mention_id      BIGSERIAL PRIMARY KEY,
    source          TEXT NOT NULL,         -- 'premium_times' | 'daily_trust' | 'humangle' | 'npf' | 'twitter' | 'telegram'
    published_at    TIMESTAMPTZ NOT NULL,
    lga_id          TEXT REFERENCES lga(lga_id),   -- nullable: geocoding may fail
    raw_text        TEXT NOT NULL,
    entities        JSONB,                 -- spaCy-extracted entities (actors, locations, groups)
    embedding       vector(384),           -- SBERT sentence embedding (all-MiniLM-L6-v2 dim)
    sentiment_score DOUBLE PRECISION,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_text_mentions_lga_date ON text_mentions (lga_id, published_at);

-- Weekly per-LGA feature/label table consumed by the model (built in later phases;
-- table created now so the ingestion pipeline has a defined target to land data in).
CREATE TABLE IF NOT EXISTS features_weekly (
    lga_id                  TEXT REFERENCES lga(lga_id),
    week_start              DATE NOT NULL,
    incident_count_7d       INTEGER,
    actor_centrality        DOUBLE PRECISION,
    festival_flag           BOOLEAN,
    military_ops_flag       BOOLEAN,
    ransom_payment_nearby   BOOLEAN,
    sentiment_score         DOUBLE PRECISION,
    spatial_lag_risk        DOUBLE PRECISION,
    entity_mentions         INTEGER,
    incident_prob_7d        DOUBLE PRECISION,   -- label, populated once outcome is known
    PRIMARY KEY (lga_id, week_start)
);
