"""Publish ACLED events onto Kafka, one message per event.

Phase 1 covers ACLED only. News/X/Telegram producers are Phase 2 work
(spaCy/SBERT text pipeline) and are stubbed below so the topic layout
is fixed now.
"""
from __future__ import annotations

import json
from datetime import date

from kafka import KafkaProducer

from src.config.settings import settings
from src.ingestion.acled_client import fetch_nigeria_events


def get_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def publish_acled_backfill(start_date: date, end_date: date) -> int:
    producer = get_producer()
    events = fetch_nigeria_events(start_date, end_date)

    count = 0
    for _, event in events.iterrows():
        producer.send(settings.kafka_topic_acled, value=event.to_dict())
        count += 1

    producer.flush()
    return count


def publish_news_stream() -> None:
    """Phase 2: Premium Times / Daily Trust / HumAngle / NPF press releases."""
    raise NotImplementedError("News scraping pipeline is Phase 2 work")


def publish_twitter_stream() -> None:
    """Phase 2: X/Twitter security analyst accounts, subject to API access tier."""
    raise NotImplementedError("Twitter ingestion is Phase 2 work; see docs/data_access_notes.md")


def publish_telegram_stream() -> None:
    """Phase 2: Telegram channels, subject to ToS constraints."""
    raise NotImplementedError("Telegram ingestion is Phase 2 work; see docs/data_access_notes.md")


if __name__ == "__main__":
    n = publish_acled_backfill(date(2015, 1, 1), date.today())
    print(f"Published {n} ACLED events to '{settings.kafka_topic_acled}'")
