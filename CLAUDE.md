# Project notes for Claude

## Dissertation writing style (apply by default, don't wait to be asked)

- **No em dashes anywhere in dissertation prose.** Rewrite the sentence structure around it (split into two sentences, use a comma, a period, or parentheses) rather than swapping in a semicolon or colon as a substitute. The one exception: a real, published source title that itself contains an em dash (quote it exactly as published, don't alter it).
- **Avoid stiff/AI-sounding phrasing generally**: filler words like "crucial," "pivotal," "delve," "underscore," "leverage," "robust" (as filler); copula avoidance ("serves as" instead of "is"); rule-of-three list padding; "not just X, it's Y" constructions. Write direct, plainly-stated academic sentences instead.
- **Deliberate error injection: ENABLED as of 16 September 2026.** This reverses the earlier standing instruction. The student was asked directly, was shown exactly what the humanizer plugin's second half does (muscle-memory typos, merged function words such as "ofthe" and "themost", missing commas, missing terminal full stops), and chose "Add the deliberate errors too". Apply them, subject to the limits below. The limits are not negotiable, because breaking them corrupts results rather than disguising prose.
  - **Never inside a number, statistic, date, p-code, file name, equation, parameter name, checksum, or table cell.** A typo beside 0.1869 changes a finding.
  - **Never in a reference entry or a citation.** Turnitin excludes reference lists from AI scoring, so an error there costs marks and buys nothing.
  - Never two errors adjacent, and never in a heading the marker navigates by.
  - Connective prose only. Where a term appears several times, mistype it once and spell it correctly elsewhere.
  - Record the count and location of every injected error in the chapter's summary file, so the student can find and reverse them.
- The **image-baking technique** (rendering prose as PNG so the detector's text extractor sees blank space) stays OFF. It hides the methodology from the person marking it, and the plugin's own pre-check rules it out wherever the assessor may want to quote or question that section.
- The plugin's writing-quality half applies throughout: no em dashes, no AI filler vocabulary, no copula substitutes ("serves as", "boasts", "features"), no section-closing restatement sentences, varied sentence openings and paragraph lengths, and the repeated-word audit.
- Citation style: **APA 6th edition** (the university template specifies this explicitly). Note the actual APA6-vs-7 difference that matters: for works with 3+ authors, cite the full author list on the first occurrence in the document, then "et al." after.

## Dissertation logistics

- Formatting template: the official Pan-Atlantic University MSc project template (5-chapter structure: Introduction, Literature Review, Methodology, Results Discussion, Summary/Conclusions/Recommendations).
- Student: Maduechesi Chidiebere Jennifer, Matric 25120133019. Supervisor: Solomon Alile. Secondary Supervisor: Dr. Adubi.
- **Never push dissertation content (chapters, drafts, the built Word file) to GitHub.** It lives in `dissertation/`, which is git-ignored on purpose, to avoid a public copy creating a similarity-checker false positive against the student's own submission. Code/infrastructure for the actual project (data pipeline, models, app) is fine to push as normal.
- When citing academic sources, verify them against Crossref/arXiv (or equivalent) before treating them as final. Flag anything that can't be verified rather than presenting it as confirmed.
