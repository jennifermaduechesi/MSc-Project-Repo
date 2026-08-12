# Project notes for Claude

## Dissertation writing style (apply by default, don't wait to be asked)

- **No em dashes anywhere in dissertation prose.** Rewrite the sentence structure around it (split into two sentences, use a comma, a period, or parentheses) rather than swapping in a semicolon or colon as a substitute. The one exception: a real, published source title that itself contains an em dash (quote it exactly as published, don't alter it).
- **Avoid stiff/AI-sounding phrasing generally**: filler words like "crucial," "pivotal," "delve," "underscore," "leverage," "robust" (as filler); copula avoidance ("serves as" instead of "is"); rule-of-three list padding; "not just X, it's Y" constructions. Write direct, plainly-stated academic sentences instead.
- **This is a writing-quality standard, not detection evasion.** Do not insert deliberate typos, merged words, or punctuation errors to defeat AI-writing detectors (e.g. Turnitin's AI Writing Indicator). That was explicitly declined for this project and stays declined unless the user directly says otherwise in the moment.
- Citation style: **APA 6th edition** (the university template specifies this explicitly). Note the actual APA6-vs-7 difference that matters: for works with 3+ authors, cite the full author list on the first occurrence in the document, then "et al." after.

## Dissertation logistics

- Formatting template: the official Pan-Atlantic University MSc project template (5-chapter structure: Introduction, Literature Review, Methodology, Results Discussion, Summary/Conclusions/Recommendations).
- Student: Maduechesi Chidiebere Jennifer, Matric 25120133019. Supervisor: Solomon Alile. Secondary Supervisor: Dr. Adubi.
- **Never push dissertation content (chapters, drafts, the built Word file) to GitHub.** It lives in `dissertation/`, which is git-ignored on purpose, to avoid a public copy creating a similarity-checker false positive against the student's own submission. Code/infrastructure for the actual project (data pipeline, models, app) is fine to push as normal.
- When citing academic sources, verify them against Crossref/arXiv (or equivalent) before treating them as final. Flag anything that can't be verified rather than presenting it as confirmed.
