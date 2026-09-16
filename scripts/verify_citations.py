"""Check every work Chapter Three cites: it has a reference entry, the entry is
unambiguous, the APA 6 first-occurrence rule is met, and the DOI resolves at Crossref.

Written after a looser check matched "Reinhart and Greenhouse (2018)" to the reference
for "Reinhart (2018)", which is a different paper by the same first author in the same
year. Matching on surname and year alone is not matching.
"""
from __future__ import annotations
import json, re, sys, time, urllib.error, urllib.parse, urllib.request
from pathlib import Path

D = Path("dissertation")
CHAPTER = D / "chapter3_methodology.md"
REF_FILES = ["chapter1_references.md", "chapter2_references.md", "chapter3_references.md"]
EARLIER = ["chapter1_introduction_v2.md", "chapter2_literature_review.md"]
MAILTO = "maduechesijennifer@gmail.com"

PASS, FAIL, WARN = [], [], []
def ok(label, detail=""):   PASS.append((label, detail))
def bad(label, detail=""):  FAIL.append((label, detail))
def warn(label, detail=""): WARN.append((label, detail))


def surnames(authors: str) -> list[str]:
    a = re.sub(r"\(Ed[s]?\.\)", "", authors)
    out = []
    for p in re.split(r",\s*(?=[A-Z]\.)|,?\s*&\s*|,\s*and\s+", a):
        p = p.strip().strip(",.")
        if p and not re.fullmatch(r"[A-Z]\.?( [A-Z]\.?)*", p):
            sn = p.split(",")[0].strip()
            if len(sn) > 1: out.append(sn)
    return out


# ------------------------------------------------------------------ the reference list
entries = []
for f in REF_FILES:
    for ln in (D / f).read_text().split("\n"):
        s = ln.strip()
        if not s or s.startswith("#"): continue
        m = re.match(r"^([A-Z][^(]{2,180}?)\s*\((\d{4}[a-z]?)\)", s)
        if m:
            entries.append({"authors": m.group(1).strip(), "year": m.group(2),
                            "raw": s, "file": f, "surnames": surnames(m.group(1))})

# The per-chapter lists repeat any work cited in more than one chapter. Those repeats are
# the same entry, not two candidate references, so collapse them before matching. A repeat
# whose text differs is a real inconsistency and is reported rather than silently merged.
by_text, dupes = {}, []
for e in entries:
    prev = by_text.get(e["raw"])
    if prev is None:
        by_text[e["raw"]] = e
    else:
        prev["file"] += f", {e['file']}"
seen_key = {}
for e in by_text.values():
    k = (tuple(sn.lower() for sn in e["surnames"]), e["year"])
    if k in seen_key and seen_key[k]["raw"] != e["raw"]:
        dupes.append((seen_key[k], e))
    seen_key.setdefault(k, e)
entries = list(by_text.values())
for a, b in dupes:
    bad(f"consistent entry for {a['surnames'][0]} ({a['year']})",
        f"{a['file']} and {b['file']} give different text")

# Works Chapter Three cites, written as (surname list, year). An "et al." citation is
# stored with the surnames the reference entry must begin with.
CITED = [
 (["Barton", "Lennox"], "2022"), (["Bird", "King"], "2018"), (["Breiman"], "2001"),
 (["Chatzimparmpas"], "2021"), (["Clarke"], "2023"), (["Cohen", "Felson"], "1979"),
 (["Cragg"], "1971"), (["Davis", "Goadrich"], "2006"), (["Dawkins"], "2021"),
 (["Dietrich", "Eck"], "2020"), (["Fagan"], "2007"), (["Feng"], "2021"),
 (["Hajihosseinlou"], "2024"),
 (["United Nations Office for the Coordination of Humanitarian Affairs"], "2025"),
 (["Johnson"], "2008"), (["Kadar"], "2019"), (["King", "Zeng"], "2001"),
 (["Kounadi"], "2020"), (["Lewis"], "2012"), (["Mullahy"], "1986"),
 (["Niculescu-Mizil", "Caruana"], "2005"), (["Ratcliffe", "Rengert"], "2008"),
 (["Reinhart", "Greenhouse"], "2018"), (["Rose"], "2006"),
 (["Saito", "Rehmsmeier"], "2015"), (["Tekin", "Kozat"], "2023"), (["Wolpert"], "1992"),
]

ET_AL = {"Chatzimparmpas", "Fagan", "Hajihosseinlou", "Kadar", "Kounadi", "Lewis", "Rose"}
resolved = {}
for want, year in CITED:
    key = f"{' & '.join(want)} ({year})"
    cands = [e for e in entries if e["year"] == year
             and [s.lower() for s in e["surnames"][:len(want)]] == [w.lower() for w in want]]
    if want[0] in ET_AL:
        # An "et al." citation names only the first author, so the entry is the one whose
        # first surname matches and which actually has three or more authors.
        cands = [e for e in entries if e["year"] == year
                 and e["surnames"][:1] == want[:1] and len(e["surnames"]) >= 3]
    elif len(want) == 1:
        # A single-surname citation must match an entry with exactly that one author,
        # otherwise it is ambiguous against a same-year paper by the same first author.
        cands = [e for e in cands if len(e["surnames"]) == 1]
    if len(cands) == 1:
        resolved[key] = cands[0]
        ok(f"reference for {key}", cands[0]["file"])
    elif not cands:
        bad(f"reference for {key}", "no entry matches this author list and year")
    else:
        bad(f"reference for {key}", f"{len(cands)} entries match, citation is ambiguous")

# ------------------------------------ APA 6: full author list on first occurrence in doc
earlier_text = "\n".join((D / f).read_text() for f in EARLIER)
for first in sorted(ET_AL):
    entry = next((e for e in resolved.values() if e["surnames"][:1] == [first]), None)
    if entry is None: continue
    full = entry["surnames"]
    pattern = re.escape(full[0]) + r"[^.)]{0,120}?" + re.escape(full[-1]) + r"\s*\(\d{4}\)"
    if re.search(pattern, earlier_text):
        ok(f"APA6 first occurrence in full for {first} et al.", " and ".join(full))
    else:
        bad(f"APA6 first occurrence in full for {first} et al.",
            "Chapter Three shortens it but no earlier chapter spells it out")

# --------------------------------------- an initial is required when surnames collide
chapter_text = CHAPTER.read_text()
def first_initials(authors: str) -> str:
    m = re.match(r"^[^,]+,\s*((?:[A-Z]\.\s*)+)", authors)
    return re.sub(r"\s+", "", m.group(1)) if m else ""

by_surname = {}
for e in entries:
    if e["surnames"]:
        by_surname.setdefault(e["surnames"][0].lower(), set()).add(first_initials(e["authors"]))
for sn, variants in by_surname.items():
    # Two entries by the same person, one solo and one with co-authors, are not a
    # collision. Only distinct initials mean distinct people sharing a surname.
    if len(variants) < 2: continue
    if not re.search(rf"\b{re.escape(sn)}\b", chapter_text, re.I): continue
    hits = re.findall(rf"([A-Z]\.(?: [A-Z]\.)*\s+)?{re.escape(sn)}\s*\(", chapter_text, re.I)
    if any(h.strip() for h in hits):
        ok(f"initials used for colliding surname {sn.title()}", f"{len(variants)} authors share it")
    else:
        bad(f"initials used for colliding surname {sn.title()}",
            f"{len(variants)} different authors share this surname")

# -------------------------------------------------------------- DOIs against Crossref
def crossref(doi: str) -> dict | None:
    url = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}?mailto={MAILTO}"
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)["message"]
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
            time.sleep(2 * (attempt + 1))
        except Exception:
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Crossref unreachable for {doi}")

for key, e in sorted(resolved.items()):
    m = re.search(r"doi:(10\.\S+?)\s*$", e["raw"])
    if not m:
        warn(f"DOI present for {key}", "no DOI in the entry")
        continue
    doi = m.group(1).rstrip(".")
    rec = crossref(doi)
    if rec is None:
        bad(f"Crossref resolves {key}", f"{doi} returns 404")
        continue
    got_year = None
    for f in ("published-print", "published-online", "issued"):
        if rec.get(f, {}).get("date-parts", [[None]])[0][0]:
            got_year = rec[f]["date-parts"][0][0]; break
    got_first = (rec.get("author") or [{}])[0].get("family", "")
    want_first = e["surnames"][0] if e["surnames"] else ""
    year_ok = str(got_year) == re.sub(r"[a-z]", "", e["year"])
    name_ok = (not want_first) or (not got_first) or \
              want_first.lower().replace("-", "") in got_first.lower().replace("-", "")
    title = (rec.get("title") or [""])[0]
    if year_ok and name_ok:
        ok(f"Crossref {key}", f"{doi}  {title[:58]}")
    else:
        bad(f"Crossref {key}",
            f"{doi} -> {got_first} {got_year}; entry says {want_first} {e['year']}")

# ---------------------------------------------------------------------------- report
w = max(len(l) for l, _ in PASS + FAIL + WARN) + 2
for l, d in PASS: print(f"  [pass] {l:<{w}} {d}")
for l, d in WARN: print(f"  [warn] {l:<{w}} {d}")
for l, d in FAIL: print(f"  [FAIL] {l:<{w}} {d}")
print("=" * 110)
print(f"{len(PASS)} passed, {len(WARN)} warnings, {len(FAIL)} failed, "
      f"{len(PASS)+len(WARN)+len(FAIL)} citation checks")
print("=" * 110)
sys.exit(1 if FAIL else 0)
