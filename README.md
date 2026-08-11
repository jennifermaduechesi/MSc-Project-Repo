# Kidnapping & Banditry Hotspot Predictor (OSINT)

MSc Data Science project — see `Project_Brief_Kidnapping_Banditry_Hotspot_Predictor.md`
for the full brief. This README covers the Phase 1 data-foundation setup.

## Phase 1 scope

- PostgreSQL + PostGIS + pgvector schema (`sql/001_schema.sql`)
- Kafka (single-node, KRaft) for streaming ingestion
- ACLED backfill client → Kafka producer
- News/X/Telegram producers are stubbed; built out in Phase 2

## Quickstart

1. Copy `.env.example` to `.env` and fill in your ACLED credentials
   (register at https://acleddata.com/register/ — see `docs/data_access_notes.md`
   for auth-flow notes before relying on the client as-is).
2. Start the infrastructure:
   ```
   docker compose up -d
   ```
   This builds a custom Postgres image (PostGIS + pgvector) and runs the
   schema in `sql/` automatically on first start, plus a single-node Kafka
   broker.
3. Create a virtualenv and install dependencies:
   ```
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. Populate the `lga` table from an authoritative Nigeria admin-boundary
   source (see `docs/data_access_notes.md`) — the ACLED backfill below
   depends on `lga` already being populated for LGA-level geocoding.
5. Run the ACLED backfill and publish to Kafka:
   ```
   python -m src.ingestion.kafka_producer
   ```

## Repo layout

```
sql/                 Postgres/PostGIS/pgvector schema
docker/postgres/      Custom Postgres image (postgis + pgvector)
docker-compose.yml    Postgres + Kafka for local dev
src/config/           Environment-driven settings
src/ingestion/        ACLED client, Kafka producers
docs/                 Data access notes, constraints
PROJECT_LOG.md        Running log of struggles/errors/limitations/surprises
```
