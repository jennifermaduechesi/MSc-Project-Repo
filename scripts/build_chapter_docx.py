"""Build a dissertation chapter as a Word file in the Pan-Atlantic template.

Used for every chapter so the whole document is set identically. It handles three
things beyond plain prose:

  * markdown pipe tables become native Word tables ruled the APA way, which means
    a rule above the header, a rule below the header and a rule below the last row,
    and no vertical lines anywhere;
  * "Table N" sits on its own line above an italic title, and the note below opens
    with an italic "Note";
  * an image reference becomes a centred picture, with the "Figure N." label in
    italics at the start of the caption beneath it.

Body text follows the Pan-Atlantic MSc project template: US Letter, one inch margins,
Times New Roman 12pt, double spaced, first line indent on body paragraphs.
Table text is set one point smaller and single spaced, which APA permits and which
keeps a five column table on the page.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

BODY_PT = 12
TABLE_PT = 10
INDENT = Inches(0.5)
DOUBLE = 480          # w:line, twentieths of a point
SINGLE = 240
FIGURE_WIDTH = Inches(6.0)


# --------------------------------------------------------------------------- base
def _set_spacing(paragraph, line: int, before: int = 0, after: int = 0) -> None:
    pf = paragraph.paragraph_format._element.get_or_add_pPr()
    spacing = pf.find(qn("w:spacing"))
    if spacing is None:
        spacing = OxmlElement("w:spacing")
        pf.append(spacing)
    spacing.set(qn("w:line"), str(line))
    spacing.set(qn("w:lineRule"), "auto")
    spacing.set(qn("w:before"), str(before))
    spacing.set(qn("w:after"), str(after))


def _style(document) -> None:
    normal = document.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(BODY_PT)
    rpr = normal.element.get_or_add_rPr().get_or_add_rFonts()
    for attr in ("w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"):
        rpr.set(qn(attr), "Times New Roman")
    section = document.sections[0]
    section.page_width, section.page_height = Inches(8.5), Inches(11)
    for side in ("top", "bottom", "left", "right"):
        setattr(section, f"{side}_margin", Inches(1))


def _runs(paragraph, text: str, size: int = BODY_PT) -> None:
    """Render **bold** and *italic* inline spans."""
    for part in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = paragraph.add_run(part[2:-2]); run.bold = True
        elif part.startswith("*") and part.endswith("*"):
            run = paragraph.add_run(part[1:-1]); run.italic = True
        else:
            run = paragraph.add_run(part)
        run.font.name = "Times New Roman"
        run.font.size = Pt(size)


# -------------------------------------------------------------------------- tables
def _border(element, edge: str, size: int) -> None:
    borders = element.find(qn("w:tcBorders"))
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        element.append(borders)
    tag = borders.find(qn(f"w:{edge}"))
    if tag is None:
        tag = OxmlElement(f"w:{edge}")
        borders.append(tag)
    tag.set(qn("w:val"), "single" if size else "nil")
    tag.set(qn("w:sz"), str(size))
    tag.set(qn("w:color"), "000000")


def _apa_table(document, rows: list[list[str]]) -> None:
    """A table ruled the APA way: horizontal rules only, none vertical."""
    table = document.add_table(rows=len(rows), cols=len(rows[0]))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    last = len(rows) - 1
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            cell = table.cell(r, c)
            cell.text = ""
            paragraph = cell.paragraphs[0]
            _set_spacing(paragraph, SINGLE, before=20, after=20)
            _runs(paragraph, text, size=TABLE_PT)
            if r == 0:
                for run in paragraph.runs:
                    run.bold = False          # APA uses plain, not bold, column heads
            tc = cell._tc.get_or_add_tcPr()
            _border(tc, "left", 0)
            _border(tc, "right", 0)
            _border(tc, "top", 6 if r in (0, 1) else 0)
            _border(tc, "bottom", 6 if r == last else 0)


def _flush_table(document, buffer: list[str]) -> None:
    rows = []
    for line in buffer:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-{2,}:?", c) for c in cells):
            continue                          # the markdown separator row
        rows.append(cells)
    if rows:
        _apa_table(document, rows)
    buffer.clear()


# --------------------------------------------------------------------------- build
def build(source: Path, out_path: Path) -> None:
    document = Document()
    _style(document)
    lines = source.read_text().splitlines()
    table_buffer: list[str] = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("|"):
            table_buffer.append(stripped)
            continue
        if table_buffer:
            _flush_table(document, table_buffer)

        if not stripped:
            continue

        # figure image
        image = re.fullmatch(r"!\[[^\]]*\]\(([^)]+)\)", stripped)
        if image:
            target = (source.parent / image.group(1)).resolve()
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _set_spacing(paragraph, SINGLE, before=120, after=120)
            paragraph.add_run().add_picture(str(target), width=FIGURE_WIDTH)
            continue

        if stripped.startswith("# "):
            paragraph = document.add_paragraph()
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            _set_spacing(paragraph, DOUBLE)
            run = paragraph.add_run(stripped[2:].upper())
            run.bold = True
            run.font.name, run.font.size = "Times New Roman", Pt(14)
            continue

        if stripped.startswith("### "):
            paragraph = document.add_paragraph()
            _set_spacing(paragraph, DOUBLE)
            run = paragraph.add_run(stripped[4:])
            run.bold, run.italic = False, True
            run.font.name, run.font.size = "Times New Roman", Pt(BODY_PT)
            continue

        if stripped.startswith("## "):
            paragraph = document.add_paragraph()
            _set_spacing(paragraph, DOUBLE)
            run = paragraph.add_run(stripped[3:])
            run.bold = True
            run.font.name, run.font.size = "Times New Roman", Pt(BODY_PT)
            continue

        paragraph = document.add_paragraph()
        _set_spacing(paragraph, DOUBLE)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

        # "Table 2", the italic table title, the note and the figure caption all sit
        # flush left; ordinary body paragraphs take a first line indent.
        flush_left = (
            re.fullmatch(r"Table \d+", stripped)
            or stripped.startswith("*Note*")
            or stripped.startswith("*Figure ")
            or (stripped.startswith("*") and stripped.endswith("*") and "*" not in stripped[1:-1])
        )
        if not flush_left:
            paragraph.paragraph_format.first_line_indent = INDENT
        _runs(paragraph, stripped)

    if table_buffer:
        _flush_table(document, table_buffer)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    document.save(out_path)
    tables = len(document.tables)
    print(f"wrote {out_path} ({len(document.paragraphs)} paragraphs, {tables} tables)")


if __name__ == "__main__":
    build(Path(sys.argv[1]), Path(sys.argv[2]))


# Kept in the repository rather than in a scratch directory, because an earlier copy was
# lost when a container was recycled and the chapters then had no way to be rebuilt.
