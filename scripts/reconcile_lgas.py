"""Map the incident file's Local Government Area labels onto the official 774.

The supplied incident data records the LGA as free text, and holds 905 distinct
values where Nigeria has 774 units. The surplus is a mixture of capitalisation
variants, curly against straight apostrophes, hyphen against slash conventions,
misspellings, and place names that are towns rather than LGAs. None of the
modelling can begin until each label resolves to one official unit, because the
panel is indexed by LGA.

The reference list is the OCHA Common Operational Dataset for Nigerian
administrative boundaries, `nga_admin2`, which carries 774 rows with a p-code, the
state, and up to three alternate spellings per unit. It is reference geography
rather than incident data, and is used only to resolve names.

**Matching is constrained by state, and that is not optional.** Six LGA names
occur in two different states: Bassa in Kogi and Plateau, Ifelodun and Irepodun in
both Kwara and Osun, and Nasarawa in both Kano and Nasarawa state. Matching on
name alone would silently assign events to the wrong part of the country, and the
resulting map would look plausible.

Three passes run in order, from strictest to loosest, and every match records
which pass produced it so the loose ones can be audited:

1. Exact match on the normalised name within the same state, including the
   official alternate spellings.
2. Fuzzy match within the same state, above a similarity threshold.
3. Anything still unmatched is reported rather than guessed. A label that cannot
   be resolved is left null, and the incident is excluded from the panel, which
   is the honest treatment.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

# Below this similarity a fuzzy match is not trustworthy enough to accept without
# a human looking at it. Chosen so that "Birnin-kudu" against "Birnin Kudu"
# passes while genuinely different names do not.
FUZZY_THRESHOLD = 0.88

STATE_ALIASES = {
    "fct": "federal capital territory",
    "fct abuja": "federal capital territory",
    "abuja": "federal capital territory",
}

# Labels that no automatic rule can resolve, each checked by hand against the
# official state list and recorded here with the reason. Keyed by (state, label)
# after normalisation. Three distinct causes are represented.
#
# Renames. Yewa North and Yewa South are the current names of the LGAs the
# reference list still calls Egbado North and Egbado South.
#
# Errors in the reference list rather than in the incident data. Kaduna's LGA is
# Makarfi; the OCHA list transposes it to "Markafi". Imo's is commonly Onuimo;
# the OCHA list has "Unuimo". In both cases the incident file is right and the
# reference is wrong, which is worth stating in the methodology.
#
# Headquarters towns recorded in place of their LGA. Gamboru is the seat of Ngala,
# and Koton Karfe of Kogi LGA.
MANUAL_OVERRIDES = {
    ("federalcapitalterritory", "municipalarea"): "Abuja Municipal",
    ("bayelsa", "yenagoa"): "Yenegoa",
    ("ogun", "yewanorth"): "Egbado North",
    ("ogun", "yewasouth"): "Egbado South",
    ("kebbi", "danko"): "Wasagu/Danko",
    ("kebbi", "dankowasagu"): "Wasagu/Danko",
    ("kebbi", "arewa"): "Arewa-Dandi",
    ("zamfara", "chafe"): "Tsafe",
    ("kaduna", "makarfi"): "Markafi",
    ("imo", "onuimo"): "Unuimo",
    ("borno", "gamborungala"): "Ngala",
    ("kogi", "kotonkarfe"): "Kogi",
}


def normalise(value: object) -> str:
    """Casefold and strip everything that varies between spellings of one name.

    Unicode normalisation matters here: the file contains both the typographic
    apostrophe in Jama'are and the ASCII one in Jama'are, and they are different
    code points that would otherwise never match.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.replace("’", "'").replace("‘", "'")
    text = text.lower().strip()
    return re.sub(r"[^a-z0-9]", "", text)


def normalise_state(value: object) -> str:
    key = " ".join(str(value or "").lower().split())
    return normalise(STATE_ALIASES.get(key, key))


def build_reference(path: Path) -> pd.DataFrame:
    """One row per official LGA, with every known spelling flattened out."""
    frame = pd.read_excel(path, sheet_name="nga_admin2")
    name_columns = [c for c in frame.columns if re.fullmatch(r"adm2_name\d?", c)]
    rows = []
    for record in frame.itertuples():
        state = normalise_state(getattr(record, "adm1_name"))
        for column in name_columns:
            name = getattr(record, column, None)
            if name is None or (isinstance(name, float) and pd.isna(name)):
                continue
            rows.append(
                {
                    "state_key": state,
                    "name_key": normalise(name),
                    "pcode": getattr(record, "adm2_pcode"),
                    "official_name": getattr(record, "adm2_name"),
                    "official_state": getattr(record, "adm1_name"),
                }
            )
    reference = pd.DataFrame(rows).drop_duplicates(subset=["state_key", "name_key"])
    return reference


def reconcile(incidents: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Return one row per distinct (state, LGA label) with its resolution."""
    labels = (
        incidents[["State", "Local Govt Area"]]
        .dropna(subset=["Local Govt Area"])
        .astype(str)
        .drop_duplicates()
        .reset_index(drop=True)
    )
    labels["state_key"] = labels["State"].map(normalise_state)
    labels["name_key"] = labels["Local Govt Area"].map(normalise)

    exact = reference.set_index(["state_key", "name_key"])
    by_state: dict[str, list[str]] = {}
    for state_key, group in reference.groupby("state_key"):
        by_state[state_key] = group["name_key"].tolist()

    by_official = reference.drop_duplicates(subset=["state_key", "official_name"]).set_index(
        ["state_key", "official_name"]
    )

    results = []
    for row in labels.itertuples():
        key = (row.state_key, row.name_key)

        override = MANUAL_OVERRIDES.get(key)
        if override is not None:
            hit = by_official.loc[(row.state_key, override)]
            results.append((row.State, getattr(row, "_2"), override,
                            hit["official_state"], hit["pcode"], "manual", 1.0))
            continue

        if key in exact.index:
            hit = exact.loc[key]
            results.append((row.State, getattr(row, "_2"), hit["official_name"],
                            hit["official_state"], hit["pcode"], "exact", 1.0))
            continue

        candidates = by_state.get(row.state_key, [])
        best_name, best_score = None, 0.0
        for candidate in candidates:
            score = difflib.SequenceMatcher(None, row.name_key, candidate).ratio()
            if score > best_score:
                best_name, best_score = candidate, score
        if best_name is not None and best_score >= FUZZY_THRESHOLD:
            hit = exact.loc[(row.state_key, best_name)]
            results.append((row.State, getattr(row, "_2"), hit["official_name"],
                            hit["official_state"], hit["pcode"], "fuzzy",
                            round(best_score, 4)))
        else:
            results.append((row.State, getattr(row, "_2"), None, None, None,
                            "unmatched", round(best_score, 4)))

    return pd.DataFrame(
        results,
        columns=["state_in_file", "label_in_file", "official_name",
                 "official_state", "pcode", "match_type", "score"],
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incidents", default="data/raw/DATA_SOURCE.xlsx")
    parser.add_argument("--reference", default="data/reference/nga_admin_boundaries.xlsx")
    parser.add_argument("--out", default="data/reference/lga_crosswalk.csv")
    args = parser.parse_args()

    incidents = pd.read_excel(args.incidents)
    reference = build_reference(Path(args.reference))
    crosswalk = reconcile(incidents, reference)

    counts = incidents.groupby(["State", "Local Govt Area"]).size().rename("records")
    crosswalk = crosswalk.merge(
        counts.reset_index().rename(
            columns={"State": "state_in_file", "Local Govt Area": "label_in_file"}
        ),
        on=["state_in_file", "label_in_file"], how="left",
    )
    crosswalk["records"] = crosswalk["records"].fillna(0).astype(int)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    crosswalk.sort_values(["match_type", "records"], ascending=[True, False]).to_csv(
        args.out, index=False
    )

    total_labels = len(crosswalk)
    total_records = int(crosswalk["records"].sum())
    print(f"distinct (state, label) pairs: {total_labels:,}")
    print(f"records covered:               {total_records:,}\n")
    for kind in ("manual", "exact", "fuzzy", "unmatched"):
        part = crosswalk[crosswalk["match_type"] == kind]
        print(f"  {kind:10s} labels={len(part):>4,}  records={int(part['records'].sum()):>7,} "
              f"({100 * part['records'].sum() / total_records:5.2f}% of records)")

    resolved = crosswalk[crosswalk["pcode"].notna()]
    print(f"\ndistinct official LGAs reached: {resolved['pcode'].nunique():,} of 774")

    unmatched = crosswalk[crosswalk["match_type"] == "unmatched"].nlargest(20, "records")
    if not unmatched.empty:
        print("\nlargest unmatched labels (these need a human decision):")
        for row in unmatched.itertuples():
            print(f"   {row.records:>5,}  {row.state_in_file:<18} {row.label_in_file!r}"
                  f"   closest score {row.score}")

    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
