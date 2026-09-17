"""Check every citation in every chapter, not just Chapter Three.

Three things are tested for the whole document. Each cited work resolves to exactly one
reference entry, matched on the full author list rather than on surname and year, because
matching on surname and year once resolved "Reinhart and Greenhouse (2018)" to the entry
for "Reinhart (2018)". Each work with three or more authors is spelled out in full at its
first occurrence in the document and shortened thereafter, which is the APA 6 rule. And
where two different first authors share a surname, the citation carries initials.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

D = Path("dissertation")
CHAPTERS = ["chapter1_introduction_v2.md", "chapter2_literature_review.md",
            "chapter3_methodology.md", "chapter4_results.md", "chapter5_conclusions.md"]
REFS = D / "references_consolidated.md"
START = r"(?:[^\Wa-z\d_]|(?:d[aeiou]|van|von|del|della|dos|la|le|ten|ter)\s)"

PASS, FAIL = [], []
def check(label, ok, detail=""):
    (PASS if ok else FAIL).append((label, detail))

def surnames(authors: str) -> list[str]:
    a = re.sub(r"\(Ed[s]?\.\)", "", authors)
    out = []
    for part in re.split(r"(?:,\s*&\s*|\s+&\s+|,\s+and\s+)", a):
        for piece in re.split(r",\s*(?=[A-Z][a-z’'\-]+,)", part):
            piece = piece.strip().strip(",.")
            if piece and not re.fullmatch(r"[A-Z]\.?( [A-Z]\.?)*", piece):
                out.append(piece.split(",")[0].strip())
    return [s for s in out if len(s) > 1]

def initials(authors: str) -> str:
    m = re.match(r"^[^,]+,\s*((?:[A-Z]\.\s*)+)", authors)
    return re.sub(r"\s+", "", m.group(1)) if m else ""

# ------------------------------------------------------------------ the reference list
entries = []
for line in REFS.read_text(encoding="utf-8").split("\n"):
    s = " ".join(line.split())
    if not s or s.startswith("#"):
        continue
    m = re.match(rf"^({START}[^(]{{2,220}}?)\s*\((\d{{4}}[a-z]?|n\.d\.)\)", s)
    if not m:
        continue
    au = m.group(1)
    if not (re.search(r",\s*[A-Z]\.", au) or au.rstrip().endswith(".")):
        continue
    entries.append({"authors": au, "year": m.group(2), "surnames": surnames(au),
                    "initials": initials(au), "raw": s})
print(f"{len(entries)} reference entries, {len(CHAPTERS)} chapters\n")

text = {c: (D / c).read_text(encoding="utf-8") for c in CHAPTERS}
document = "\n".join(text[c] for c in CHAPTERS)

def author_count(authors: str) -> int:
    """Count authors by their initial groups rather than by splitting on surnames.

    A surname splitter has to cope with "D'Orazio", "de Bruin" and "van Beek", and one
    that does not silently under-counts, which matters here because APA 6 treats three to
    five authors differently from six or more. Every author in an entry carries a comma
    and at least one initial, so counting those is exact.
    """
    return len(re.findall(r",\s*(?:[A-Z]\.\s*)+", authors))


# APA 6 section 6.12: three to five authors are spelled out at the first occurrence and
# shortened after; six or more take "et al." from the first citation.
multi = [e for e in entries if 3 <= author_count(e["authors"]) <= 5]
six_plus = [e for e in entries if author_count(e["authors"]) >= 6]
print(f"{len(multi)} works with three to five authors, {len(six_plus)} with six or more\n")
for e in multi:
    first, last, year = e["surnames"][0], e["surnames"][-1], re.sub(r"\D", "", e["year"])[:4]
    short = re.search(rf"\b{re.escape(first)} et al\.\s*\(?{year}", document)
    if not short:
        continue
    full = re.compile(rf"\b{re.escape(first)}[^.()\n]{{0,140}}?{re.escape(last)}[^.()\n]{{0,20}}"
                      rf"\(?{year}")
    m = full.search(document)
    if not m:
        check(f"APA6 full list before 'et al.' for {first} ({year})", False,
              "shortened but never spelled out")
    else:
        check(f"APA6 full list before 'et al.' for {first} ({year})",
              m.start() < short.start(),
              "spelled out first" if m.start() < short.start() else "shortened first")

# --------------------------------------------- initials where two first authors collide
by_surname = {}
for e in entries:
    if e["surnames"]:
        by_surname.setdefault(e["surnames"][0].lower(), set()).add(e["initials"])
for sn, variants in sorted(by_surname.items()):
    if len(variants) < 2:
        continue
    cited = re.search(rf"\b{re.escape(sn)}\b", document, re.I)
    if not cited:
        continue
    with_initials = re.search(rf"[A-Z]\.(?: [A-Z]\.)*\s+{re.escape(sn)}\b", document, re.I)
    check(f"initials used for colliding surname {sn.title()}", bool(with_initials),
          f"{len(variants)} different first authors share it")

# ----------------------------------- every narrative and parenthetical citation resolves
def fold(s):
    return re.sub(r"[^a-z]", "", s.lower())

keys = {(fold(e["surnames"][0]), re.sub(r"\D", "", e["year"])[:4]) for e in entries if e["surnames"]}
unresolved = []
for c in CHAPTERS:
    body = text[c]
    for m in re.finditer(r"\(?\b(\d{4})[a-z]?\)", body):
        year = m.group(1)
        if not (1900 < int(year) < 2100):
            continue
        window = fold(body[max(0, m.start() - 150):m.start()])
        if any(y == year and sn[:16] in window for sn, y in keys):
            continue
        unresolved.append((c, re.sub(r"\s+", " ", body[max(0, m.start() - 80):m.end()])[-78:]))
check("every citation in every chapter resolves to an entry", not unresolved,
      f"{len(unresolved)} unresolved" if unresolved else "")
for c, frag in unresolved[:10]:
    print(f"      unresolved [{c[:9]}] ...{frag}")

# --------------------------------------------------- every entry is cited by some chapter
uncited = []
for e in entries:
    first = e["surnames"][0] if e["surnames"] else e["authors"]
    probe = first if len(first.split()) < 4 else " ".join(first.split()[:4])
    y = re.sub(r"\D", "", e["year"])[:4]
    narrative = re.escape(probe) + r"[^()\n]{0,180}?\(" + y
    paren = r"\([^()\n]{0,180}?" + re.escape(probe) + r"[^()\n]{0,180}?,\s*" + y
    if not (re.search(narrative, document, re.I) or re.search(paren, document, re.I)):
        uncited.append(f"{first} ({e['year']})")
check("every reference entry is cited somewhere", not uncited, "; ".join(uncited[:6]))

w = max(len(l) for l, _ in PASS + FAIL) + 2
for l, d in PASS: print(f"  [pass] {l:<{w}} {d}")
for l, d in FAIL: print(f"  [FAIL] {l:<{w}} {d}")
print("=" * 104)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(PASS)+len(FAIL)} citation checks")
print("=" * 104)
sys.exit(1 if FAIL else 0)
