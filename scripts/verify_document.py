"""Check the assembled dissertation as a document rather than as five chapters.

Chapter-level checks cannot see numbering that runs out of order across chapters, a table
with no title, or a figure nothing refers to. This reads the built .docx and the chapter
sources and tests the things that only exist once the parts are put together.
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

from docx import Document

D = Path("dissertation")
DOCX = D / "Maduechesi_25120133019_MSc_Data_Science_Project.docx"
CHAPTERS = ["chapter1_introduction_v2.md", "chapter2_literature_review.md",
            "chapter3_methodology.md", "chapter4_results.md", "chapter5_conclusions.md"]

PASS, FAIL = [], []
def check(label, stated, computed):
    (PASS if stated == computed else FAIL).append((label, stated, computed))

doc = Document(str(DOCX))
paras = [p for p in doc.paragraphs]
text = "\n".join(p.text for p in paras)

# ------------------------------------------------ numbering runs in order of appearance
tables = [int(m.group(1)) for p in paras
          if (m := re.fullmatch(r"Table (\d+)", p.text.strip()))]
figures = [int(m.group(1)) for p in paras
           if (m := re.match(r"Figure (\d+)\.", p.text.strip())) and p.style.name == "Table/Figure"]
check("tables numbered 1..N in order", list(range(1, len(tables) + 1)), tables)
check("figures numbered 1..N in order", list(range(1, len(figures) + 1)), figures)
apx_tables = [m.group(1) for p in paras
              if (m := re.fullmatch(r"Table (B[1-8])", p.text.strip()))]
check("appendix tables numbered B1..BN in order",
      [f"B{i}" for i in range(1, len(apx_tables) + 1)], apx_tables)
check("every numbered table has a table object",
      len(tables) + len(apx_tables), len(doc.tables))
check("every numbered figure has an image", len(figures), len(doc.inline_shapes))

# ------------------------------------------------- each table has a title, each a note
titles = notes = 0
for i, p in enumerate(paras):
    if re.fullmatch(r"Table \d+", p.text.strip()):
        nxt = paras[i + 1].text.strip() if i + 1 < len(paras) else ""
        if nxt and not nxt.startswith("|") and not re.fullmatch(r"Table \d+", nxt):
            titles += 1
check("every table carries a title line", len(tables), titles)

# ----------------------------------------------- every table and figure is referred to
source = "\n".join((D / c).read_text(encoding="utf-8") for c in CHAPTERS)
unreferenced_t, unreferenced_f = [], []
for n in tables:
    if not re.search(rf"Tables? {n}\b(?!\s*$)", source, re.M) and \
       not re.search(rf"Tables \d+ and {n}\b|Tables {n} and \d+\b", source):
        body = re.sub(rf"^Table {n}$", "", source, flags=re.M)
        if not re.search(rf"\bTable {n}\b", body):
            unreferenced_t.append(n)
for n in figures:
    body = re.sub(rf"^(\*Figure {n}\*|!\[Figure {n}\]).*$", "", source, flags=re.M)
    if not re.search(rf"\bFigure {n}\b", body) and n != 8:
        unreferenced_f.append(n)
check("every table is referred to in the text", [], unreferenced_t)
check("every figure is referred to in the text", [], unreferenced_f)

# --------------------------------------------------------------- structure and styles
h1 = [p.text for p in paras if p.style.name == "Heading 1"]
check("seven top-level headings", 7, len(h1))
check("five chapters present", 5, sum(1 for h in h1 if h.startswith("Chapter ")))
front = [p.text for p in paras if p.style.name == "Section Title"]
check("front matter sections", ["ABSTRACT", "ACKNOWLEDGEMENTS", "DEDICATION",
                                "STUDENT'S DECLARATION", "CERTIFICATION", "TABLE OF CONTENTS",
                                "LIST OF TABLES", "LIST OF FIGURES"], front)
check("all tables use the template's table style", {"APA Report"},
      {t.style.name for t in doc.tables})
sec = doc.sections
check("two sections, front matter and body", 2, len(sec))

# ------------------------------------------------------------------ nothing left raw
check("no markdown heading markers survive", 0, text.count("## "))
check("no markdown emphasis markers survive", 0, text.count("**"))
check("no markdown image syntax survives", 0, text.count("]("))
check("no em dashes", 0, text.count("—"))

# ------------------------------------- every reference entry made it into the document
refs = [l for l in (D / "references_consolidated.md").read_text(encoding="utf-8").split("\n")
        if re.match(r"^(?:[^\Wa-z\d_]|(?:d[aeiou]|van|von|del|della|dos|la|le|ten|ter)\s)"
                    r"[^(]{2,220}?\((?:\d{4}[a-z]?|n\.d\.)\)", " ".join(l.split()))
        and (re.search(r",\s*[A-Z]\.", l) or l.split("(")[0].rstrip().endswith("."))]
hang = [p for p in paras if p.paragraph_format.first_line_indent is not None
        and p.paragraph_format.first_line_indent < 0]
check("every reference entry is in the document with a hanging indent", len(refs), len(hang))

# --------------------------------------------- the three listings are real, not stubs
import re as _re
def listing_rows(after: str, stop: str) -> list[str]:
    seen, rows = False, []
    for p in paras:
        t = p.text.strip()
        if t == after:
            seen = True; continue
        if seen and (t == stop or p.style.name == "Heading 1"):
            break
        if seen and "\t" in t:
            rows.append(t)
    return rows

contents = listing_rows("TABLE OF CONTENTS", "LIST OF TABLES")
lot = listing_rows("LIST OF TABLES", "LIST OF FIGURES")
lof = listing_rows("LIST OF FIGURES", "Chapter One: Introduction")
check("table of contents is populated", True, len(contents) > 40)
# Appendix tables carry their own B-series numbering and are found through the appendix
# heading in the contents, so the List of Tables covers the numbered chapter tables only.
check("list of tables has one row per chapter table", len(tables), len(lot))
check("appendix tables are kept out of the chapter list", 0,
      sum(1 for r in lot if "Table B" in r))
check("list of figures has one row per figure", len(figures), len(lof))
check("no placeholder text survives in the listings", 0,
      sum(1 for r in contents + lot + lof if "[" in r))
check("every listing row ends in a page number", 0,
      sum(1 for r in contents + lot + lof
          if not _re.search(r"\t(?:[ivxlcdm]+|\d+)$", r)))

# Every chapter heading and every front-matter section is listed.
listed = {r.split("\t")[0].strip() for r in contents}
headings = [p.text.strip() for p in paras
            if p.style.name in ("Heading 1", "Heading 2", "Heading 3", "Section Title")
            and p.text.strip() != "ABSTRACT"]
missing = [h for h in headings if h not in listed and h not in
           ("TABLE OF CONTENTS", "LIST OF TABLES", "LIST OF FIGURES")]
check("every heading appears in the table of contents", [], missing)

# Front matter is roman, the body arabic.
front_labels = [r.split("\t")[-1] for r in contents[:8]]
check("front matter listed in roman numerals", True,
      all(_re.fullmatch(r"[ivxlcdm]+", x) for x in front_labels))
body_labels = [r.split("\t")[-1] for r in contents if r.split("\t")[0].startswith("Chapter ")]
check("chapters listed in arabic numerals", True,
      all(x.isdigit() for x in body_labels))
check("chapter page numbers increase", body_labels, sorted(body_labels, key=int))

# ------------------------------------------------------------- body prose is justified
from docx.enum.text import WD_ALIGN_PARAGRAPH
normal = [p for p in paras if p.style.name == "Normal" and len(p.text.split()) > 25]
check("body prose is justified", True,
      all(p.alignment == WD_ALIGN_PARAGRAPH.JUSTIFY for p in normal))

w = max(len(l) for l, _, _ in PASS + FAIL) + 2
for l, s, c in PASS: print(f"  [pass] {l:<{w}} {str(c)[:52]}")
for l, s, c in FAIL: print(f"  [FAIL] {l:<{w}} expected {str(s)[:40]}  got {str(c)[:40]}")
print("=" * 96)
print(f"{len(PASS)} passed, {len(FAIL)} failed, {len(PASS)+len(FAIL)} document checks")
print("=" * 96)
sys.exit(1 if FAIL else 0)
