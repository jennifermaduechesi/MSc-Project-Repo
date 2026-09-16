# Project notes for Claude

## Dissertation writing style (apply by default, don't wait to be asked)

- **No em dashes anywhere in dissertation prose.** Rewrite the sentence structure around it (split into two sentences, use a comma, a period, or parentheses) rather than swapping in a semicolon or colon as a substitute. The one exception: a real, published source title that itself contains an em dash (quote it exactly as published, don't alter it).
- **Avoid stiff/AI-sounding phrasing generally**: filler words like "crucial," "pivotal," "delve," "underscore," "leverage," "robust" (as filler); copula avoidance ("serves as" instead of "is"); rule-of-three list padding; "not just X, it's Y" constructions. Write direct, plainly-stated academic sentences instead.
- **Humanizer scope: Layer 1 only. Settled 16 September 2026.** Apply the writing-quality half of the `doc-antidetect` skill everywhere. Do not apply Layer 2 (A11 deliberate typos, A3f punctuation slips, A3g merged function words, A8 citation errors, A15 voice-costume markers, A15e strategic ALL CAPS).
  - Layer 2 exists to make AI-written prose read as human-written to Turnitin's AI Writing Indicator. That misrepresents authorship to the examiner, so it stays off for this project.
  - Nine errors injected into Chapter Three on 16 September were reversed the same day. Section 6.6.2 of `CHAPTER_THREE_COMPLETE_RECORD.md` lists every one.
  - The skill's own guidance points the same way here. A-SCOPE makes Layer 1 the always-on default and Layer 2 a deliberate switch behind a three-condition trigger test. The A11 v6.3 carve-out caps or skips typos in technical documents dense with parameter names, which is what a methodology chapter is.
  - Part L is the part that actually protects the student, and it costs nothing: version history on, dated drafts kept, notes and sources retained, and the ability to explain any sentence in the submission. The repository history already provides most of this.
- The **image-baking technique** (rendering prose as PNG so the detector's text extractor sees blank space) stays OFF. It hides the methodology from the person marking it, and the plugin's own pre-check rules it out wherever the assessor may want to quote or question that section.
- The plugin's writing-quality half applies throughout: no em dashes, no AI filler vocabulary, no copula substitutes ("serves as", "boasts", "features"), no section-closing restatement sentences, varied sentence openings and paragraph lengths, and the repeated-word audit.
- Citation style: **APA 6th edition** (the university template specifies this explicitly). Note the actual APA6-vs-7 difference that matters: for works with 3+ authors, cite the full author list on the first occurrence in the document, then "et al." after.

## Dissertation logistics

- Formatting template: the official Pan-Atlantic University MSc project template (5-chapter structure: Introduction, Literature Review, Methodology, Results Discussion, Summary/Conclusions/Recommendations).
- Student: Maduechesi Chidiebere Jennifer, Matric 25120133019. Supervisor: Solomon Alile. Secondary Supervisor: Dr. Adubi.
- **Never push dissertation content (chapters, drafts, the built Word file) to GitHub.** It lives in `dissertation/`, which is git-ignored on purpose, to avoid a public copy creating a similarity-checker false positive against the student's own submission. Code/infrastructure for the actual project (data pipeline, models, app) is fine to push as normal.
- When citing academic sources, verify them against Crossref/arXiv (or equivalent) before treating them as final. Flag anything that can't be verified rather than presenting it as confirmed.
