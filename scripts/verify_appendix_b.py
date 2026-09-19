"""Check the Appendix B review matrix against the Chapter Two reference list.

The matrix exists to answer a supervisor's question about how many studies were
reviewed, so the count it implies has to be the count that is actually there. A row
naming a work that is not in the reference list would be an invention, and a reference
with no row would be an omission. Both are checked here, in both directions, along with
the totals the appendix states in its own opening paragraph.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

APX = Path("dissertation/appendix_b_review_matrix.md")
REFS = Path("dissertation/chapter2_references.md")
CH2 = Path("dissertation/chapter2_literature_review.md")

ENTRY = re.compile(r"^(?:[^\Wa-z\d_]|(?:d[aeiou]|van|von|del|della|dos|la|le|ten|ter)\s)")
fails: list[str] = []
passes = 0


def check(label, expected, got) -> None:
    global passes
    ok = expected == got
    if ok:
        passes += 1
    else:
        fails.append(label)
    print(f"  [{'pass' if ok else 'FAIL'}] {label:<62} expected {expected!r:>28}  got {got!r}")


def ref_entries() -> list[str]:
    text = REFS.read_text(encoding="utf-8")
    body = text[text.index("Adepeju"):]
    return [l.strip() for l in body.split("\n") if l.strip() and ENTRY.match(l.strip())]


def short_form(entry: str, n_authors: int) -> set[str]:
    """The acceptable in-text labels for a reference, ignoring any leading initials."""
    first = entry.split(",")[0].strip()
    year = re.search(r"\((\d{4})[a-z]?\)", entry).group(1)
    if n_authors == 1:
        return {f"{first} ({year})"}
    if n_authors == 2:
        m = re.search(r"&\s*([^,]+),", entry)
        second = m.group(1).strip() if m else ""
        return {f"{first} and {second} ({year})"}
    return {f"{first} et al. ({year})"}


def author_count(entry: str) -> int:
    """Count authors by counting ", Initials" groups; the last author carries one too."""
    head = entry.split("(")[0].strip()
    if " . . . " in head:
        return 6
    return len(re.findall(r",\s*(?:[A-Z]\.\s*)+", head))


def main() -> None:
    apx = APX.read_text(encoding="utf-8")
    rows = [l for l in apx.split("\n")
            if l.startswith("| ") and not l.startswith("| Study ")
            and not set(l.replace("|", "").strip()) <= set("-: ")]
    labels = [r.strip("|").split("|")[0].strip() for r in rows]

    check("every row has three columns", [], [r for r in rows if len(r.strip("|").split("|")) != 3])
    check("no duplicate rows", len(labels), len(set(labels)))
    check("no em dashes", 0, apx.count("—"))
    check("no empty cells", [], [r for r in rows if any(not c.strip() for c in r.strip("|").split("|"))])

    entries = ref_entries()
    wanted: dict[str, str] = {}
    for e in entries:
        for form in short_form(e, author_count(e)):
            wanted[form] = e

    # A label may carry a disambiguating initial, e.g. "X. Zhang et al. (2022)".
    def strip_initials(s: str) -> str:
        return re.sub(r"^(?:[A-Z]\.\s*)+", "", s)

    matched, unmatched = {}, []
    for lab in labels:
        key = lab if lab in wanted else strip_initials(lab)
        if key in wanted:
            matched[key] = lab
        else:
            unmatched.append(lab)

    check("every matrix row names a work in the reference list", [], unmatched)
    check("every reference in the list has a matrix row", [],
          sorted({wanted[k][:44] for k in wanted if k not in matched}))
    check("matrix row count equals reference count", len(entries), len(rows))

    stated = re.search(r"([A-Z][a-z]+(?:-[a-z]+)?(?:\s+[a-z-]+)*)\s+works are listed", apx)
    words = {"seventy-nine": 79, "seventy-four": 74, "five": 5}
    check("appendix states the total in words", 79, words.get((stated.group(1) if stated else "").lower()))
    m = re.search(r"([A-Za-z-]+) are peer-reviewed\s+journal articles and ([A-Za-z-]+) are conference", apx)
    conf = [e for e in entries
            if re.search(r"In \*Proceedings|\(pp\. |Piscataway|New York, NY: Association|Cambridge, England", e)]
    check("stated peer-reviewed count matches the list", len(entries) - len(conf),
          words.get(m.group(1).lower()) if m else None)
    check("stated other-venue count matches the list", len(conf),
          words.get(m.group(2).lower()) if m else None)

    # A label carrying initials is disambiguating a shared surname. The initials have to be
    # the first author's own, and the surname has to be one that is genuinely shared.
    surnames = [e.split(",")[0].strip() for e in entries]
    wrong_initials, not_shared = [], []
    for lab in labels:
        m = re.match(r"^((?:[A-Z]\.\s*)+)(.+?) (?:et al\.|and |\()", lab)
        if not m:
            continue
        init, sur = m.group(1).strip(), m.group(2).strip()
        if surnames.count(sur) < 2:
            not_shared.append(lab)
        entry = wanted.get(strip_initials(lab), "")
        given = re.match(r"[^,]+,\s*((?:[A-Z]\.\s*)+)", entry)
        if not given or given.group(1).strip() != init:
            wrong_initials.append(lab)
    check("initials on a label are the first author's own", [], wrong_initials)
    check("initials are only used for surnames shared by two works", [], not_shared)

    # Nothing in the matrix may contradict how Chapter Two cites the same work.
    ch2 = CH2.read_text(encoding="utf-8")
    check("every labelled surname appears in Chapter Two", [],
          [l for l in labels if strip_initials(l).split(" ")[0] not in ch2])

    # Chapter Two now states the counts in words. They have to be the real ones.
    NUM = {"seventy-nine": 79, "seventy-four": 74, "five": 5,
           "fifty-nine": 59, "forty-five": 45}
    years = [int(re.search(r"\((\d{4})[a-z]?\)", e).group(1)) for e in entries]
    body = ch2[:ch2.index("## Conflict Event Data")]
    def stated(pattern):
        m = re.search(pattern, body, re.I)
        return NUM.get(m.group(1).lower()) if m else None
    check("Chapter Two states the total reviewed", len(entries),
          stated(r"([A-Za-z-]+) works are reviewed"))
    check("Chapter Two states the peer-reviewed count", len(entries) - len(conf),
          stated(r"([A-Za-z-]+) are peer-reviewed journal"))
    check("Chapter Two states the other-venue count", len(conf),
          stated(r"and ([A-Za-z-]+) are conference papers"))
    check("Chapter Two states the count published since 2015", sum(y >= 2015 for y in years),
          stated(r"([A-Za-z-]+) of the seventy-nine date from 2015"))
    check("Chapter Two states the count published since 2020", sum(y >= 2020 for y in years),
          stated(r"and ([A-Za-z-]+) from 2020 onward"))
    check("Chapter Two points the reader to the appendix", True, "Appendix B" in ch2)
    check("Table 2 is described as a subset, not the whole review", True,
          "subset of the literature reviewed in this chapter" in ch2)

    print("=" * 110)
    print(f"{passes} passed, {len(fails)} failed, {passes + len(fails)} checks")
    print("=" * 110)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
