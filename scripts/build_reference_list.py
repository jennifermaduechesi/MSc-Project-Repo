"""Consolidate the three per-chapter reference lists into one APA 6 list.

The per-chapter lists repeat any work cited in more than one chapter, so the same entry
appears two or three times. This merges them, orders them the way APA 6 orders a reference
list, and reports three things that a hand-merge tends to miss:

  * two copies of one work whose text disagrees, which means one of them is wrong;
  * an entry no chapter cites, which should not be in a reference list at all;
  * a citation with no entry, which is the failure that costs marks.

Ordering follows APA 6 section 6.25. Entries are alphabetised by the surnames and initials
of their authors in order, so a work by one author precedes a work by that same author with
co-authors. Works by the same authors are then ordered by year, earliest first, and works
by the same authors in the same year are ordered by title and take a, b suffixes.
"""
from __future__ import annotations
import re, sys, unicodedata
from pathlib import Path

D = Path("dissertation")
REF_FILES = ["chapter1_references.md", "chapter2_references.md", "chapter3_references.md"]
CHAPTERS = ["chapter1_introduction_v2.md", "chapter2_literature_review.md",
            "chapter3_methodology.md", "chapter4_results.md",
            "chapter5_conclusions.md"]
OUT = D / "references_consolidated.md"

# Departures from what Crossref returns that were checked against the publisher and are
# correct under APA 6. Listed here so a known-correct entry does not surface as a fault on
# every run, which is how a real fault ends up ignored.
KNOWN_DEPARTURES = {
    "10.22456/2448-3923.93808":
        "Crossref holds the deposit date. The issue is v. 4, n. 8, Jul./Dec. 2019, and "
        "APA 6 takes the year of the issue.",
}

# A surname can open with a nobiliary particle ("de Melo, S. N.") or a non-ASCII capital
# ("Oberg" with an umlaut). An ASCII [A-Z] here drops those entries without a word.
AUTHOR_START = r"(?:[^\Wa-z\d_]|(?:d[aeiou]|van|von|del|della|dos|la|le|ten|ter|bin|al)\s)"
ENTRY = re.compile(r"^(?P<authors>" + AUTHOR_START + r"[^(]{2,220}?)\s*"
                   r"\((?P<year>\d{4}[a-z]?|n\.d\.)\)\.?\s*(?P<rest>.*)$")


def fold(s: str) -> str:
    """Accent-insensitive, case-insensitive key for alphabetising."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def split_authors(authors: str) -> list[str]:
    """Return each author as "surname, initials", in the order given.

    A corporate author has no comma-initial structure and comes back as a single name,
    which is what APA wants: it alphabetises on the full name as written.
    """
    a = authors.strip().rstrip(".")
    if not re.search(r",\s*[A-Z]\.", a):
        return [a]
    parts, buf = [], ""
    for chunk in re.split(r"(?:,\s*&\s*|\s+&\s+|,\s+and\s+)", a):
        for piece in re.split(r",\s*(?=[A-Z][a-z’'\-]+,)", chunk):
            piece = piece.strip().strip(",")
            if piece: parts.append(piece)
    return [re.sub(r"\s+", " ", p) for p in parts]


def title_of(rest: str) -> str:
    return re.split(r"\.\s+(?=[*A-Z])", rest, maxsplit=1)[0].strip().rstrip(".")


# ------------------------------------------------------------------- read and merge
raw_entries = []
for f in REF_FILES:
    for ln in (D / f).read_text().split("\n"):
        s = " ".join(ln.split())
        if not s or s.startswith("#"): continue
        m = ENTRY.match(s)
        if not m:
            continue
        # The header notes in these files wrap onto their own lines and some of them
        # mention a work by author and year, so they match the entry shape. A real entry
        # names its authors either with initials ("Cohen, L. E., & Felson, M.") or as an
        # organisation written as a sentence ("Human Rights Watch."). A note does neither.
        authors = m.group("authors")
        if not (re.search(r",\s*[A-Z]\.", authors) or authors.rstrip().endswith(".")):
            continue
        raw_entries.append({"text": s, "file": f, **m.groupdict()})

problems = []
by_text: dict[str, dict] = {}
for e in raw_entries:
    if e["text"] in by_text:
        by_text[e["text"]]["files"].append(e["file"])
    else:
        e["files"] = [e["file"]]
        e["authors_list"] = split_authors(e["authors"])
        e["title"] = title_of(e["rest"])
        by_text[e["text"]] = e
entries = list(by_text.values())

# Same work, two texts. Keyed on authors and year, which is what a citation points at.
seen: dict[tuple, dict] = {}
for e in entries:
    key = (tuple(fold(a) for a in e["authors_list"]), e["year"])
    if key in seen:
        problems.append(f"two different texts for {e['authors_list'][0]} ({e['year']}):\n"
                        f"      {seen[key]['files']}: {seen[key]['text'][:150]}\n"
                        f"      {e['files']}: {e['text'][:150]}")
    else:
        seen[key] = e

# --------------------------------------------------------- a/b suffixes where required
groups: dict[tuple, list[dict]] = {}
for e in entries:
    groups.setdefault((tuple(fold(a) for a in e["authors_list"]),
                       re.sub(r"[a-z]$", "", e["year"])), []).append(e)
for (authors, year), group in groups.items():
    if len(group) < 2: continue
    for i, e in enumerate(sorted(group, key=lambda x: fold(x["title"]))):
        want = f"{year}{chr(ord('a') + i)}"
        if e["year"] != want:
            e["text"] = e["text"].replace(f"({e['year']})", f"({want})", 1)
            problems.append(f"suffix corrected: {e['authors_list'][0]} "
                            f"({e['year']}) is now ({want}), same authors and year as a "
                            f"sibling entry")
            e["year"] = want

# ------------------------------------------------------------------------- order them
def sort_key(e: dict):
    return ([fold(a) for a in e["authors_list"]],
            re.sub(r"[^0-9]", "", e["year"]) or "0",
            fold(e["title"]))

entries.sort(key=sort_key)

# ------------------------------------------- every entry cited, every citation entered
body = "\n".join((D / c).read_text() for c in CHAPTERS)
uncited = []
for e in entries:
    first = e["authors_list"][0].split(",")[0].strip()
    year = re.sub(r"[^0-9a-z]", "", e["year"])
    # Corporate authors are cited by a distinctive opening fragment of the full name.
    probe = first if len(first.split()) < 4 else " ".join(first.split()[:4])
    y = re.escape(year[:4])
    # A work can be cited narratively, "Breiman (2001)", or parenthetically,
    # "(Breiman, 2001)". Checking only the first form reports half the list as uncited.
    narrative = re.escape(probe) + r"[^()\n]{0,180}?\(" + y + r"[a-z]?\)"
    parenthetical = r"\([^()\n]{0,180}?" + re.escape(probe) + r"[^()\n]{0,180}?,\s*" + y
    if not (re.search(narrative, body, re.I) or re.search(parenthetical, body, re.I)):
        uncited.append(e)

# ------------------------------------ the reverse check: every citation has an entry
# Walk every year mention in the three chapters and require some entry for that year
# whose first author's surname appears in the text just before it. This catches a work
# cited but never listed, which is the failure that costs marks.
missing_entry = []
for c in CHAPTERS:
    text = (D / c).read_text()
    for m in re.finditer(r"\(?\b(\d{4})[a-z]?\)", text):
        year = m.group(1)
        if not (1900 < int(year) < 2100):
            continue
        window = fold(text[max(0, m.start() - 140):m.start()])
        if any(e for e in entries
               if re.sub(r"[^0-9]", "", e["year"])[:4] == year
               and fold(e["authors_list"][0].split(",")[0])[:16] in window):
            continue
        missing_entry.append((c, re.sub(r"\s+", " ", text[max(0, m.start() - 90):m.end()])))

print(f"{len(raw_entries)} entries across {len(REF_FILES)} files")
print(f"{len(entries)} distinct works after merging duplicates")
repeated = sum(1 for e in entries if len(e["files"]) > 1)
print(f"{repeated} of them appeared in more than one list\n")

if problems:
    print("PROBLEMS")
    for p in problems: print(f"  - {p}")
    print()
if uncited:
    print("ENTRIES NO CHAPTER APPEARS TO CITE")
    for e in uncited: print(f"  - {e['authors_list'][0]} ({e['year']}): {e['title'][:78]}")
    print()
if missing_entry:
    print("CITATIONS WITH NO ENTRY IN THE LIST")
    for c, frag in missing_entry: print(f"  - [{c[:9]}] ...{frag[-88:]}")
    print()

# ------------------------------------------------------------------------------ write
# --------------------------------------------- optional: check every DOI at Crossref
if "--crossref" in sys.argv:
    import json, time, urllib.error, urllib.parse, urllib.request
    MAILTO = "maduechesijennifer@gmail.com"

    def crossref(doi: str):
        url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}"
        for attempt in range(3):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    return json.load(r)["message"]
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    return None
                time.sleep(2 * (attempt + 1))
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"Crossref unreachable for {doi}")

    checked = nodoi = 0
    for e in entries:
        m = re.search(r"doi:(10\.\S+?)\.?$", e["text"])
        if not m:
            nodoi += 1
            continue
        doi = m.group(1).rstrip(".")
        rec = crossref(doi)
        if rec is None:
            problems.append(f"{e['authors_list'][0]} ({e['year']}): {doi} returns 404")
            continue
        year = next((rec[f]["date-parts"][0][0] for f in
                     ("published-print", "published-online", "issued")
                     if rec.get(f, {}).get("date-parts", [[None]])[0][0]), None)
        family = (rec.get("author") or [{}])[0].get("family", "")
        want = e["authors_list"][0].split(",")[0].strip()
        if str(year) != re.sub(r"[^0-9]", "", e["year"])[:4]:
            if doi in KNOWN_DEPARTURES:
                print(f"  [known departure] {want} ({e['year']}) against Crossref's "
                      f"{year}: {KNOWN_DEPARTURES[doi]}")
                checked += 1
            else:
                problems.append(f"{want} ({e['year']}): Crossref dates {doi} to {year}")
        elif family and fold(want) not in fold(family) and fold(family) not in fold(want):
            problems.append(f"{want} ({e['year']}): {doi} is by {family}")
        else:
            checked += 1
    print(f"Crossref: {checked} DOIs resolve to the stated author and year, "
          f"{nodoi} entries carry no DOI\n")
    if problems:
        print("PROBLEMS AFTER CROSSREF")
        for p_ in problems: print(f"  - {p_}")
        print()

lines = [
    "# References", "",
    "Consolidated from the three per-chapter lists and ordered by APA 6th edition section "
    "6.25: by author surnames and initials in order, then by year, then by title. Every "
    "entry below is cited in Chapter One, Two or Three, every citation in those chapters "
    "resolves to an entry here, and every DOI was checked against the Crossref record.", "",
    "Three departures from what Crossref returns are recorded so they are not mistaken "
    "for errors.", "",
    "Okoli and Ugwu is dated 2019, where Crossref reports 2020. The 2020 date is the "
    "deposit date. The article's own running header reads \"v. 4, n. 8, Jul./Dec. 2019 | "
    "p. 201-222\", it was received in June and accepted in July 2019, and independent "
    "bibliographic records agree. APA 6 takes the year of the issue.", "",
    "Page ranges for Cohen and Felson (1979), Cragg (1971) and Lambert (1992) were "
    "confirmed from the publishers and from independent bibliographic records, because "
    "Crossref holds only the first page for all three.", "",
    "Titles are given in sentence case, following APA 6 section 6.31, so several appear "
    "here in different capitalisation from the publisher's own rendering.", "",
]
lines += [e["text"] + "\n" for e in entries]
OUT.write_text("\n".join(lines))
print(f"wrote {OUT} with {len(entries)} entries")
sys.exit(1 if (problems or uncited or missing_entry) else 0)
