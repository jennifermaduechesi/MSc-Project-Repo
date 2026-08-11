# Project Log — Struggles, Errors, Limitations, Mistakes, Surprises

Running log for the dissertation's reflection/limitations section. Add an
entry whenever something goes wrong, takes longer than expected, or turns
out different from the plan — not just the polished outcome.

Format: `Date | Phase | Category | Note`

---

**2026-08-11 | Phase 1 | Limitation** — The build environment's outbound
network policy blocks `api.acleddata.com` (proxy returns a 502 policy
denial). The ACLED client code was written from the documented API
contract but could not be executed/tested from this environment. It
needs a real run — with a registered ACLED key — from a machine that has
access, before Phase 2 relies on it.

**2026-08-11 | Phase 1 | Surprise** — `postgis/postgis` and `pgvector`
don't ship in one official image. Had to layer `postgresql-16-pgvector`
onto the PostGIS base image via a custom Dockerfile
(`docker/postgres/Dockerfile`) rather than pulling a single ready-made
image.

**2026-08-11 | Phase 1 | Limitation** — ACLED's API auth has reportedly
moved some accounts to OAuth bearer tokens instead of the classic
`key`+`email` params; which flow applies wasn't confirmed (couldn't
reach ACLED's docs/registration from this session to check). Flagged in
`docs/data_access_notes.md` as something to verify at registration.

**2026-08-11 | Phase 1 | Note** — The remote build environment is
ephemeral (container is reclaimed after inactivity) — Kafka/Postgres
here are for local/dev spin-up and schema validation, not a persistent
deployment. Actual data backfill and long-running ingestion will need to
run somewhere persistent (local machine or a hosted VM).

**2026-08-11 | Phase 1 | Limitation** — Could not actually run
`docker compose up` in the build sandbox: the Docker daemon can't start
inside this container (nested containers aren't permitted here), and
there's no standalone Postgres/PostGIS server installed to fall back to.
`docker-compose.yml` was validated with `docker compose config` (parses
and resolves cleanly) and `sql/001_schema.sql` was reviewed manually,
but neither has been executed end-to-end yet. First real task for
whoever runs this next: `docker compose up -d` on a machine with a
working Docker daemon, confirm the schema applies cleanly and the
`vector` extension installs without error.
