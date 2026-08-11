"""ACLED API client for backfilling Nigeria security events.

Requires a free ACLED account (register at https://acleddata.com/register/).
ACLED's auth flow has changed over time — some accounts use the classic
`key` + `email` query params, newer registrations may be issued an OAuth
bearer token instead. Verify which applies to this account's credentials
against the current ACLED API docs before relying on this client; see
docs/data_access_notes.md.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests

from src.config.settings import settings

ACLED_READ_URL = "https://api.acleddata.com/acled/read"
PAGE_SIZE = 500


def fetch_nigeria_events(start_date: date, end_date: date) -> pd.DataFrame:
    """Pull all ACLED events for Nigeria in [start_date, end_date], paginated."""
    if not settings.acled_api_key or not settings.acled_email:
        raise RuntimeError(
            "ACLED_API_KEY / ACLED_EMAIL not set. Register at "
            "https://acleddata.com/register/ and populate .env (see .env.example)."
        )

    all_rows: list[dict] = []
    page = 1
    while True:
        params = {
            "key": settings.acled_api_key,
            "email": settings.acled_email,
            "country": "Nigeria",
            "event_date": f"{start_date.isoformat()}|{end_date.isoformat()}",
            "event_date_where": "BETWEEN",
            "limit": PAGE_SIZE,
            "page": page,
        }
        response = requests.get(ACLED_READ_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        rows = payload.get("data", [])
        if not rows:
            break

        all_rows.extend(rows)
        if len(rows) < PAGE_SIZE:
            break

        page += 1
        time.sleep(1)  # be polite to the API

    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    df = fetch_nigeria_events(date(2015, 1, 1), date.today())
    print(f"Fetched {len(df)} events")
    df.to_csv("data/raw/acled_nigeria_backfill.csv", index=False)
