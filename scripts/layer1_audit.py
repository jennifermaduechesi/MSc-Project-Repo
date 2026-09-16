"""Mechanical Layer 1 audit of a chapter, per doc-antidetect Part I.

Reads the chapter, not a transcription of it. Every figure below is measured.
Layer 2 checks (A11 typos, A3f/A3g slips, A8 citation errors, A15 voice markers)
are deliberately absent.
"""
import re, sys
from collections import Counter

src = open(sys.argv[1], encoding="utf-8").read()
lines = src.split("\n")

# Prose only: drop headings, table rows, table numbers, italic captions, notes, images.
def is_prose(ln: str) -> bool:
    s = ln.strip()
    if not s: return False
    if s.startswith(("#", "|", "!", ">")): return False
    if re.fullmatch(r"(Table|Figure) \d+\.?", s): return False
    if s.startswith("*") and s.endswith("*"): return False
    return True

paras = [ln.strip() for ln in lines if is_prose(ln)]
prose = "\n".join(paras)

def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text) if s.strip()]

print(f"{'LAYER 1 AUDIT':^78}")
print("=" * 78)
print(f"prose paragraphs {len(paras)},  words {len(prose.split())}")

# --- A17 em dashes, A3c quotes ------------------------------------------------
print(f"\nA17 em dashes                          {src.count(chr(8212))}   (target 0)")
print(f"A3c curly quotes                       {sum(src.count(c) for c in '“”‘’')}   (target 0)")
print(f"    non-ascii characters               {sum(1 for c in src if ord(c) > 127)}   (target 0)")

# --- A5 banned vocabulary, A18 copula, A19 restatement ------------------------
A5 = ["delve","tapestry","multifaceted","holistic","robust","seamless","groundbreaking",
      "transformative","leverage","utilize","utilise","facilitate","in order to",
      "in terms of","with regard to","a wide range of","a variety of","a wealth of",
      "it could be argued","one might suggest","bolstered","garner","enduring",
      "intricate","interplay","align with","enhance","vibrant","showcas","highlight",
      "valuable","meticulous","boasts","furthermore","moreover","additionally",
      "it is worth noting","it is important to note","this underscores","this demonstrates",
      "firstly","secondly","thirdly","pivotal","crucial"]
A18 = ["serves as","stands as","functions as","acts as","features","offers","maintains",
       "is home to","holds the distinction"]
A19 = ["in summary","in conclusion","to summarize","to summarise","overall,","taken together",
       "collectively,","this section has shown","as demonstrated above"]
for name, lst in (("A5  banned vocabulary", A5), ("A18 copula substitutes", A18),
                  ("A19 closing restatements", A19)):
    hits = {w: len(re.findall(re.escape(w), prose, re.I)) for w in lst}
    hits = {w: c for w, c in hits.items() if c}
    print(f"\n{name:<38}{sum(hits.values())}   (target 0)")
    for w, c in sorted(hits.items(), key=lambda x: -x[1]): print(f"      {w!r}: {c}")

# --- A16 repeated descriptors -------------------------------------------------
STOP = set("""the a an and or of in to is was that it this for on with as be are were at from by
we i our they have had has not but which their been all also more can would will one when than
if its so no each my into he she do did use used data model models results table week weeks
area areas event events chapter section figure number rows row over under after before between
same other than then there here what how why because while during per such these those first
second third both any some most many much less least new only own just about above below
against again further once both few own same too very s t don now d ll m o re ve y ain
does doing done get got make made take taken go going come came see seen say said know known
think thought want wanted give given find found tell told become became leave left put
feel felt bring brought begin began keep kept hold held write written stand stood hear heard
let mean meant set sit lose lost pay met run""".split())
words = re.findall(r"\b[a-z]{4,}\b", prose.lower())
flag = [(w, c) for w, c in Counter(words).most_common(400) if c >= 4 and w not in STOP]
HIGH = {"significant","significantly","robust","key","important","importantly","overall",
        "further","specific","specifically","various","demonstrate","demonstrates",
        "highlight","highlights","ensure","impact","notable","considerable"}
risky = [(w, c) for w, c in flag if w in HIGH]
print(f"\nA16 high-risk descriptors over 3        {len(risky)}   (target 0)")
for w, c in risky: print(f"      {w}: {c}")

# --- A4 paragraph shape -------------------------------------------------------
plen = [len(sentences(p)) for p in paras]
one, seven = sum(1 for n in plen if n == 1), sum(1 for n in plen if n >= 7)
print(f"\nA4  paragraph sentence counts          min {min(plen)}, median "
      f"{sorted(plen)[len(plen)//2]}, max {max(plen)}")
print(f"    single-sentence paragraphs         {one}   (need >=1)")
print(f"    paragraphs of 7+ sentences         {seven}   (need >=1)")
print(f"    two-sentence paragraphs            {sum(1 for n in plen if n == 2)}   (need >=2)")

# --- A2 sentence length and openings ------------------------------------------
sents = sentences(prose)
lens = [len(s.split()) for s in sents]
print(f"\nA2  sentences {len(sents)}, words per sentence min {min(lens)}, "
      f"median {sorted(lens)[len(lens)//2]}, max {max(lens)}")
runs = sum(1 for i in range(len(lens) - 2)
           if max(lens[i:i+3]) - min(lens[i:i+3]) <= 2)
print(f"    runs of 3 near-equal-length        {runs}")
openers = Counter(s.split()[0] for s in sents if s.split())
print(f"    most common sentence opener        {openers.most_common(1)[0][0]!r} "
      f"x{openers.most_common(1)[0][1]} of {len(sents)}")
print(f"    distinct openers                   {len(openers)}")

# --- A9 section asymmetry -----------------------------------------------------
secs, cur, name = [], [], None
for ln in lines:
    if ln.startswith("## "):
        if name: secs.append((name, len(" ".join(cur).split())))
        name, cur = ln[3:].strip(), []
    elif is_prose(ln): cur.append(ln)
if name: secs.append((name, len(" ".join(cur).split())))
if secs:
    avg = sum(w for _, w in secs) / len(secs)
    lo, hi = min(secs, key=lambda x: x[1]), max(secs, key=lambda x: x[1])
    print(f"\nA9  sections {len(secs)}, mean {avg:.0f} words")
    print(f"    longest  {hi[0][:42]:<44}{hi[1]}  ({hi[1]/avg:.2f}x)")
    print(f"    shortest {lo[0][:42]:<44}{lo[1]}  ({lo[1]/avg:.2f}x)")

# --- A3c semicolons -----------------------------------------------------------
print(f"\nA3c semicolons in prose                {prose.count(';')}")
print("=" * 78)
