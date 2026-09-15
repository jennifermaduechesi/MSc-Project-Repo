"""Merge the Bulwark Intelligence incident log with the kidnapping specialist file.

The two supplied files overlap and cannot simply be concatenated. Measured on the
window they share, 1,340 of the additional file's rows describe an event already
present in the main log, which is 58.2% of the rows falling in that window. Appending would double count those events, inflate the
target, and distort any rate estimated from event times.

They also disagree about the period they cover and about how much they record.

    main log         2021-01-01 to 2025-12-31   23,407 rows, all incident types
    additional file  2011-07-03 to 2026-08-02    4,616 rows, kidnap and banditry only

Three decisions were taken by the student with the evidence in front of her and
are implemented here.

**Period.** The full fifteen years is retained and labelled, because it is real
data and supports the long-run description in Chapters One and Four. Only 2021 to
2025 is marked for modelling. The reason is coverage: 2011 to 2020 rests on the
additional file alone and averages about 194 events a year, against roughly 1,900
a year once both sources are present. A model trained across that boundary would
read a tenfold change in collection effort as a tenfold change in violence.

**Duplicates.** A row in the additional file is treated as a duplicate when it
shares a date and state with a main-log record and at least 60% of its title words
appear in that record's narrative. The threshold was set by reading matched pairs
rather than by picking a number: at 60% the pairs are unambiguously the same
event, with titles like "Gunmen Kidnap Polytechnic Lecturer In Rivers" matching
verbatim. The main log's record is kept as the primary, since it carries better
coordinates (93.6% against 56.3%).

**Deaths.** Where the main log is blank the additional file's Total Deaths is used,
which fills 851 values. Where both carry a figure the main log is kept. The two
disagree in 29.6% of the cases where both are populated, with the main log higher
more often than not, so this rule adds information without overwriting what the
primary compiler recorded. Every filled value is flagged so the effect can be
tested.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

TITLE_OVERLAP_THRESHOLD = 0.60
MODEL_START, MODEL_END = pd.Timestamp("2021-01-01"), pd.Timestamp("2025-12-31")

STATE_ALIASES = {
    "fct": "federalcapitalterritory",
    "fctabuja": "federalcapitalterritory",
    "abuja": "federalcapitalterritory",
}

STOPWORDS = set(
    "the a an in of at on and to for with by is was were has have been said after "
    "as from that this it its".split()
)


def normalise(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.replace("’", "'").replace("‘", "'").lower().strip()
    return "".join(ch for ch in text if ch.isalnum())


def state_key(value: object) -> str:
    key = normalise(value)
    return STATE_ALIASES.get(key, key)


def content_words(value: object) -> set[str]:
    return {w for w in re.sub(r"[^a-z ]", " ", str(value).lower()).split()
            if w and w not in STOPWORDS}


def load_main(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    subtype = frame["Incident Type 2"].astype(str).str.strip().str.lower()
    primary = frame["Incident Type"].astype(str).str.strip().str.lower()
    out = pd.DataFrame({
        "date": pd.to_datetime(frame["Date of Incident"]),
        "state_raw": frame["State"],
        "lga_raw": frame["Local Govt Area"],
        "location": frame["Location"],
        "latitude": pd.to_numeric(frame["Latitude"], errors="coerce"),
        "longitude": pd.to_numeric(frame["Longitude"], errors="coerce"),
        "narrative": frame["Details"],
        "deaths": pd.to_numeric(frame["Fatalities"], errors="coerce"),
        "kidnapped": pd.to_numeric(frame["Kidnap"], errors="coerce"),
        "perpetrator": None,
        "source_file": "main",
    })
    out["is_target"] = subtype.isin(["banditry", "kidnap incident"])
    out["category"] = subtype.where(out["is_target"], primary)
    # Security force operations are the leading-indicator predictor class.
    out["is_operation"] = primary.eq("gsf operation")
    return out


def load_additional(path: Path) -> pd.DataFrame:
    frame = pd.read_excel(path)
    category = frame["Category"].astype(str).str.strip().str.lower()
    # "Banditary" is the file's spelling throughout.
    category = category.replace({"banditary": "banditry", "kidnap": "kidnap incident"})
    return pd.DataFrame({
        "date": pd.to_datetime(frame["Date"]),
        "state_raw": frame["State"],
        "lga_raw": frame["LGA"],
        "location": frame["Community (city,town, ward)"],
        "latitude": pd.to_numeric(frame["Latitude"], errors="coerce"),
        "longitude": pd.to_numeric(frame["Longitude"], errors="coerce"),
        "narrative": frame["Title"],
        "deaths": pd.to_numeric(frame["Total Deaths"], errors="coerce"),
        "kidnapped": pd.to_numeric(frame["Kidnapee (V)"], errors="coerce"),
        "perpetrator": frame["Kidnapper (P)"],
        "source_file": "additional",
        "category": category,
        "is_target": True,          # the file contains only kidnap and banditry
        "is_operation": False,
    })


def find_duplicates(additional: pd.DataFrame, main: pd.DataFrame) -> dict[int, int]:
    """Map additional-file row index to the main-log row it duplicates."""
    index: dict[tuple[str, str], list[int]] = {}
    for row in main.itertuples():
        key = (row.date.strftime("%Y-%m-%d"), state_key(row.state_raw))
        index.setdefault(key, []).append(row.Index)

    matches: dict[int, int] = {}
    for row in additional.itertuples():
        key = (row.date.strftime("%Y-%m-%d"), state_key(row.state_raw))
        candidates = index.get(key, [])
        if not candidates:
            continue
        title_words = content_words(row.narrative)
        if not title_words:
            continue
        best_index, best_score = None, 0.0
        for candidate in candidates:
            narrative_words = content_words(main.at[candidate, "narrative"])
            if not narrative_words:
                continue
            score = len(title_words & narrative_words) / len(title_words)
            if score > best_score:
                best_index, best_score = candidate, score
        if best_score >= TITLE_OVERLAP_THRESHOLD:
            matches[row.Index] = best_index
    return matches


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", default="data/raw/DATA_SOURCE.xlsx")
    parser.add_argument("--additional", default="data/raw/Additional_data.xlsx")
    parser.add_argument("--out", default="data/processed/incidents_merged.csv")
    args = parser.parse_args()

    main_log = load_main(Path(args.main))
    additional = load_additional(Path(args.additional))
    print(f"main log        : {len(main_log):,} rows, {main_log['date'].min().date()} to {main_log['date'].max().date()}")
    print(f"additional file : {len(additional):,} rows, {additional['date'].min().date()} to {additional['date'].max().date()}")

    duplicates = find_duplicates(additional, main_log)
    print(f"\nduplicates identified: {len(duplicates):,} of {len(additional):,} additional rows")

    # ------------------------------------------------- enrich the primary record
    main_log["deaths_filled_from_additional"] = False
    main_log["kidnapped_filled_from_additional"] = False
    main_log["perpetrator"] = main_log["perpetrator"].astype("object")

    filled_deaths = filled_kidnapped = filled_perp = 0
    for additional_index, main_index in duplicates.items():
        donor = additional.loc[additional_index]

        if pd.isna(main_log.at[main_index, "deaths"]) and pd.notna(donor["deaths"]):
            main_log.at[main_index, "deaths"] = donor["deaths"]
            main_log.at[main_index, "deaths_filled_from_additional"] = True
            filled_deaths += 1

        if pd.isna(main_log.at[main_index, "kidnapped"]) and pd.notna(donor["kidnapped"]):
            main_log.at[main_index, "kidnapped"] = donor["kidnapped"]
            main_log.at[main_index, "kidnapped_filled_from_additional"] = True
            filled_kidnapped += 1

        if pd.isna(main_log.at[main_index, "perpetrator"]) and pd.notna(donor["perpetrator"]):
            main_log.at[main_index, "perpetrator"] = donor["perpetrator"]
            filled_perp += 1

    print(f"  deaths filled from additional     : {filled_deaths:,}")
    print(f"  kidnapped filled from additional  : {filled_kidnapped:,}")
    print(f"  perpetrator added from additional : {filled_perp:,}")

    # ----------------------------------------------------------- combine records
    kept = additional.drop(index=list(duplicates.keys()))
    merged = pd.concat([main_log, kept], ignore_index=True, sort=False)
    merged["in_model_period"] = merged["date"].between(MODEL_START, MODEL_END)

    for column in ("deaths_filled_from_additional", "kidnapped_filled_from_additional"):
        merged[column] = merged[column].fillna(False).astype(bool)

    print(f"\nmerged total: {len(merged):,} records "
          f"({merged['date'].min().date()} to {merged['date'].max().date()})")
    print(f"  from main log                 : {int((merged['source_file']=='main').sum()):,}")
    print(f"  new from additional           : {int((merged['source_file']=='additional').sum()):,}")
    print(f"  target events (all periods)   : {int(merged['is_target'].sum()):,}")
    print(f"  target events in model period : {int((merged['is_target'] & merged['in_model_period']).sum()):,}")
    print(f"  security operations           : {int(merged['is_operation'].sum()):,}")

    print("\ntarget events per year:")
    per_year = merged[merged["is_target"]].groupby(
        [merged["date"].dt.year, "source_file"]).size().unstack(fill_value=0)
    print(per_year.to_string())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out, index=False)
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
