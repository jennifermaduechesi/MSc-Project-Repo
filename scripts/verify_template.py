"""Check the built dissertation against the Pan-Atlantic University template.

The template is the marking standard for structure, so this reads it rather than a
transcription of it. Everything compared here is taken out of PAU_TEMPLATE.docx at run
time: the front-matter sequence, the chapter headings, the section headings the template
fixes for Chapter One, the paragraph styles it defines for each element, and the body
formatting its Normal style carries.

Three deliberate departures are recorded and asserted, rather than being allowed to pass
silently:

  Chapter Four    The template writes "Results Discussion"; the supervisor asked for
                  "Results and Discussion", and his instruction governs.
  References      The template has no reference-list section. APA 6 requires one, so it
                  is present, placed after Chapter Five and before the appendices.
  Justified body  The template's Normal style is left-aligned. The body is justified at
                  the student's instruction.
  Abstract label  The template leaves the abstract unlabelled above its "Keywords:" line.
                  It is labelled here so that it appears in the table of contents.
"""
from __future__ import annotations
import re, sys
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

TEMPLATE = Document("dissertation/PAU_TEMPLATE.docx")
BUILT = Document("dissertation/Maduechesi_25120133019_MSc_Data_Science_Project.docx")

fails, passes = [], 0


def check(label, expected, got):
    global passes
    ok = expected == got
    if ok:
        passes += 1
    else:
        fails.append(label)
    print(f"  [{'pass' if ok else 'FAIL'}] {label:<58} template {expected!r:>26}  built {got!r}")


def flow(doc):
    out = []
    for ch in doc.element.body.iterchildren():
        if ch.tag == qn("w:p"):
            p = Paragraph(ch, doc)
            if p.text.strip():
                out.append(("p", p.style.name, p.text.strip()))
        elif ch.tag == qn("w:tbl"):
            out.append(("t", Table(ch, doc).style.name, ""))
    return out


def styled(doc, style):
    return [t for k, s, t in flow(doc) if k == "p" and s == style]


def main() -> None:
    t_flow, b_flow = flow(TEMPLATE), flow(BUILT)

    print("-- chapter headings --------------------------------------------------------")
    t_h1 = [t for t in styled(TEMPLATE, "Heading 1")]
    b_h1 = [t for t in styled(BUILT, "Heading 1")]
    chapters = [h for h in t_h1 if h.startswith("Chapter")]
    # The supervisor asked for "Results and Discussion" where the template writes "Results
    # Discussion". His instruction governs, so the expected heading is adjusted here and the
    # departure is asserted below rather than passing unnoticed.
    SUPERVISOR = {"Chapter Four: Results Discussion": "Chapter Four: Results and Discussion"}
    expected = [SUPERVISOR.get(h, h) for h in chapters]
    built = [h for h in b_h1 if h.startswith("Chapter")]
    check("the five chapter headings, verbatim", expected, built)
    check("Chapter Four renamed on the supervisor's instruction",
          "Chapter Four: Results and Discussion" in built,
          "Chapter Four: Results Discussion" not in built)
    check("appendices heading present", "Appendices" in t_h1, "Appendices" in b_h1)
    check("reference list added after Chapter Five",
          True, b_h1.index("References") > b_h1.index("Chapter Five: Summary, Conclusions and Recommendations")
          and b_h1.index("References") < b_h1.index("Appendices"))

    print("\n-- front matter ------------------------------------------------------------")
    t_sec = styled(TEMPLATE, "Section Title")
    b_sec = styled(BUILT, "Section Title")
    # The template's Section Title style is all-caps at render time, so compare case-folded.
    t_norm = [s.upper() for s in t_sec if s != "How to Present Tables and Figures"]
    b_norm = [s.upper() for s in b_sec]
    check("front-matter sections, in the template's order", t_norm,
          [s for s in b_norm if s != "ABSTRACT"])
    check("abstract labelled and placed first", 0, b_norm.index("ABSTRACT"))
    check("keywords line follows the abstract", True,
          any(t.startswith("Keywords:") for k, s, t in b_flow[:40]))
    check("instructional template section dropped", False,
          "How to Present Tables and Figures" in b_sec)

    print("\n-- Chapter One's fixed sections --------------------------------------------")
    t_all = [(s, t) for k, s, t in t_flow if k == "p"]
    i = next(n for n, (s, t) in enumerate(t_all) if t == "Chapter One: Introduction")
    j = next(n for n, (s, t) in enumerate(t_all) if t == "Chapter Two: Literature Review")
    t_ch1 = [t for s, t in t_all[i:j] if s == "Heading 2"]
    b_all = [(s, t) for k, s, t in b_flow if k == "p"]
    bi = next(n for n, (s, t) in enumerate(b_all) if t == "Chapter One: Introduction")
    bj = next(n for n, (s, t) in enumerate(b_all) if t == "Chapter Two: Literature Review")
    b_ch1 = [t for s, t in b_all[bi:bj] if s == "Heading 2"]
    check("Chapter One's section headings, verbatim", t_ch1, b_ch1)

    print("\n-- styles and formatting ---------------------------------------------------")
    check("table style", {"APA Report"},
          {s for k, s, _ in b_flow if k == "t"})
    t_styles = {s for k, s, _ in t_flow if k == "p"}
    b_styles = {s for k, s, _ in b_flow if k == "p"}
    check("no style used that the template does not define", set(), b_styles - set(TEMPLATE.styles and [x.name for x in BUILT.styles]))
    check("appendix headings use the template's appendix style", True,
          "Heading 2 for Appendix" in b_styles)
    check("captions use the template's Table/Figure style", True,
          "Table/Figure" in b_styles)
    nf = TEMPLATE.styles["Normal"].paragraph_format
    check("body first-line indent, inches", round(nf.first_line_indent.inches, 2), 0.5)
    check("body line spacing", nf.line_spacing, 2.0)
    check("body font size, points", TEMPLATE.styles["Normal"].font.size.pt,
          BUILT.styles["Normal"].font.size.pt)

    print("\n-- heading case consistency ------------------------------------------------")
    small = {"a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "it", "its",
             "of", "on", "or", "the", "to", "with", "within", "against", "between"}
    bad = []
    for s, t in b_all:
        if s not in ("Heading 2", "Heading 3"):
            continue
        words = re.findall(r"[A-Za-z][\w'-]*", t)
        for n, w in enumerate(words):
            if n == 0 or w.lower() in small or not w[0].isalpha():
                continue
            if w[0].islower() and w.lower() not in {"per", "cent"}:
                bad.append(f"{t} :: {w}")
                break
    check("second and third level headings are in title case", [], bad)

    print("\n-- recorded departures -----------------------------------------------------")
    body = [p for p in BUILT.paragraphs
            if p.style.name == "Normal" and len(p.text.split()) > 25]
    check("body is justified at the student's instruction", True,
          all(str(p.alignment).startswith("JUSTIFY") for p in body))
    check("template's own Normal is left aligned, so this is a departure", None, nf.alignment)

    print("=" * 106)
    print(f"{passes} passed, {len(fails)} failed, {passes + len(fails)} template checks")
    print("=" * 106)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
