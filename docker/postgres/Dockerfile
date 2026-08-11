FROM postgis/postgis:16-3.4

# postgis/postgis already has the PGDG apt repo configured; pgvector ships
# there as postgresql-16-pgvector. Not bundled in the base image, so add it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends postgresql-16-pgvector \
    && rm -rf /var/lib/apt/lists/*
