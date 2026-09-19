"""Verify every reference against its Crossref record, field by field.

Run with --refresh to re-fetch from Crossref and rewrite the cache; otherwise the cached
records are used, so the check is reproducible without a network.

The year compared is the year of the issue, not the date Crossref calls `issued`. For a
journal that publishes online first, `issued` is the online date and can be one to three
years earlier than the issue the article finally appeared in. APA 6 cites the version of
record, so the order of preference is the printed issue date, then the journal issue
date, then `issued`. Ten entries in this list differ from `issued` for exactly that
reason and would look like errors under a naive comparison.

Entries without a digital object identifier are reports, datasets and legal instruments.
They are listed explicitly below so that a new one cannot appear unnoticed.
"""
from __future__ import annotations
import argparse, json, re, sys, time, unicodedata, urllib.request
from pathlib import Path

REFS = Path("dissertation/references_consolidated.md")
CACHE = Path("data/reference/crossref_cache.json")
MAILTO = "maduechesijennifer@gmail.com"

NO_DOI_EXPECTED = {
    "Bulwark Intelligence", "Constitution of the Federal Republic of Nigeria",
    "Human Rights Watch", "SBM Intelligence",
    "United Nations Office for the Coordination of Humanitarian Affairs",
}

# Departures from the Crossref record, each with the reason it is deliberate.
DEPARTURES = {
    "10.22456/2448-3923.93808":
        "Crossref holds the 2020 deposit year; the issue is v. 4, n. 8, Jul./Dec. 2019",
    "10.2307/2094589": "Crossref holds only the first page",
    "10.2307/1909582": "Crossref holds only the first page",
    "10.2307/1269547": "Crossref holds only the first page",
}

ENTRY = re.compile(r"^(?:[^\Wa-z\d_]|(?:d[aeiou]|van|von|del|della|dos|la|le|ten|ter)\s)")
fails, passes = [], 0


def check(label, expected, got):
    global passes
    ok = expected == got
    if ok:
        passes += 1
    else:
        fails.append(label)
    print(f"  [{'pass' if ok else 'FAIL'}] {label:<62} {expected!r:>10}  {got!r}")


# Letters that are not a base letter plus a combining mark, so NFKD leaves them whole and
# stripping non-ASCII would delete them outright. APA 6.25 alphabetises them as the plain
# letter, which puts Rod for Rod and not "Rd".
STANDALONE = str.maketrans({
    "\u00f8": "o", "\u00d8": "O", "\u00e6": "ae", "\u00c6": "Ae",
    "\u0153": "oe", "\u0152": "Oe", "\u00f0": "d", "\u00d0": "D",
    "\u00fe": "th", "\u00de": "Th", "\u0142": "l", "\u0141": "L",
    "\u0111": "d", "\u0110": "D", "\u00df": "ss",
})


def fold(s: str) -> str:
    """Compare names without being defeated by diacritics or punctuation."""
    s = unicodedata.normalize("NFKD", s.translate(STANDALONE))
    return re.sub(r"[^a-z]", "", "".join(c for c in s if not unicodedata.combining(c)).lower())


# An entry's author block always ends in a full stop before the year. The preamble notes
# mention years too ("Page ranges for Cohen and Felson (1979) ..."), so requiring that
# full stop is what separates an entry from a note about one.
HEAD = re.compile(r"^.+?\.\s*\((?:\d{4}[a-z]?|n\.d\.)\)")


def entries() -> list[str]:
    out = []
    for line in REFS.read_text(encoding="utf-8").split("\n"):
        s = line.strip()
        if s and ENTRY.match(s) and HEAD.match(s):
            out.append(s)
    return out


def sort_key(e: str) -> tuple:
    """APA 6.25 orders letter by letter on surname, then initials, then year.

    Folding the whole author block into one string breaks that, because it runs the
    surname into the initials and puts "Jing, C." before "Jin, G., Wang, Q.".
    """
    head = e.split(" (")[0]
    surname = head.split(",")[0]
    year = re.search(r"\((\d{4}|n\.d\.)", e).group(1)
    return (fold(surname), fold(head), year)


def issue_year(m: dict) -> int | None:
    for key in ("published-print",):
        v = m.get(key)
        if v:
            return (v.get("date-parts") or [[None]])[0][0]
    ji = m.get("journal-issue", {})
    for key in ("published-print", "published-online"):
        v = ji.get(key)
        if v:
            return (v.get("date-parts") or [[None]])[0][0]
    v = m.get("issued")
    return (v.get("date-parts") or [[None]])[0][0] if v else None


def refresh(dois: list[str]) -> dict:
    cache = {}
    for doi in dois:
        with urllib.request.urlopen(
                f"https://api.crossref.org/works/{doi}?mailto={MAILTO}", timeout=30) as r:
            m = json.load(r)["message"]
        cache[doi] = {
            "year": issue_year(m),
            "authors": [(a.get("family") or a.get("name") or "") for a in m.get("author", [])],
            "volume": m.get("volume"), "issue": m.get("issue"), "page": m.get("page"),
            "title": (m.get("title") or [""])[0],
            "venue": (m.get("container-title") or [""])[0],
        }
        time.sleep(0.1)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False))
    return cache


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    refs = entries()
    with_doi = {re.search(r"doi:(\S+)", e).group(1): e for e in refs if "doi:" in e}
    without = [e for e in refs if "doi:" not in e]

    print(f"-- {len(refs)} entries, {len(with_doi)} with a digital object identifier ---------")
    cache = refresh(sorted(with_doi)) if args.refresh or not CACHE.exists() \
        else json.loads(CACHE.read_text())
    check("every identifier has a cached Crossref record", [],
          sorted(set(with_doi) - set(cache)))

    print("\n-- fields against the Crossref record --------------------------------------")
    years, vols, isss, pages, firsts, counts = [], [], [], [], [], []
    for doi, e in with_doi.items():
        rec = cache.get(doi)
        if not rec:
            continue
        label = e[:40]
        stated = int(re.search(r"\((\d{4})[a-z]?\)", e).group(1))
        if rec["year"] and stated != rec["year"] and doi not in DEPARTURES:
            years.append((label, stated, rec["year"]))
        mv = re.search(r"\*,\s*\*(\d+)\*(?:\(([\d-]+)\))?,\s*([0-9]+(?:-[0-9]+)?|e\d+)", e)
        if mv:
            vol, iss, pg = mv.groups()
            if rec["volume"] and vol != str(rec["volume"]):
                vols.append((label, vol, rec["volume"]))
            if iss and rec["issue"] and iss != str(rec["issue"]):
                isss.append((label, iss, rec["issue"]))
            if rec["page"] and "-" in str(rec["page"]) and pg != str(rec["page"]) \
                    and doi not in DEPARTURES:
                pages.append((label, pg, rec["page"]))
        if rec["authors"]:
            if fold(e.split(",")[0]) != fold(rec["authors"][0]):
                firsts.append((label, e.split(",")[0], rec["authors"][0]))
            n = len(re.findall(r",\s*(?:[A-Z]\.\s*)+", e.split("(")[0]))
            if " . . . " not in e and n != len(rec["authors"]):
                counts.append((label, n, len(rec["authors"])))
    check("years match the issue year", [], years)
    check("volumes match", [], vols)
    check("issue numbers match", [], isss)
    check("page ranges match", [], pages)
    check("first author surnames match", [], firsts)
    check("author counts match, where not abbreviated", [], counts)

    print("\n-- list integrity ----------------------------------------------------------")
    keys = [sort_key(e) for e in refs]
    check("entries are in alphabetical order", sorted(keys), keys)
    check("no duplicate entries", len(refs), len(set(refs)))
    check("entries without an identifier are the expected reports and instruments",
          set(), {e.split(".")[0].split(",")[0] for e in without} - NO_DOI_EXPECTED)
    check("every departure recorded is still in the list", [],
          [d for d in DEPARTURES if d not in with_doi])
    # APA 6 takes no full stop after a retrieval address, so a URL is a valid ending.
    check("every entry ends in a full stop, an identifier or a retrieval address", [],
          [e[:40] for e in refs if not re.search(r"(\.|doi:\S+|https?://\S+)$", e)])

    print("=" * 96)
    print(f"{passes} passed, {len(fails)} failed, {passes + len(fails)} reference checks")
    print("=" * 96)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
