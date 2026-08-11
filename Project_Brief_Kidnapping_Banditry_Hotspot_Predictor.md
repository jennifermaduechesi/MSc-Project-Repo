# Kidnapping & Banditry Hotspot Predictor (OSINT)
## MSc Data Science Project Brief
Student: Maduechesi Chidiebere Jennifer
Matric: 25120133019
Supervisor: Solomon Alile
Secondary Supervisor: Dr Adubi
University: Pan Atlantic University
Programme: MSc Data Science
Track: Security & Civic — Project 08
Module: DAT 608 (Topics 4, 5, 6, 7, 11)

## The Big Idea
ACLED, SBM Intelligence, and NEXTIER together log over 4,000 violent incidents a year in Nigeria, but all three describe the past — a historical heat-map of where kidnapping and banditry *have* happened. State governments and security councils need the opposite: a forward-looking, per-LGA (Local Government Area) probability score for where an incident is *likely* in the next 7 days, along with a plain-language explanation of why (a nearby festival, an active military operation, a recent ransom payment in a neighboring LGA). This project builds that forecast, closing the gap between "here's what happened" and "here's what to watch this week."

## Personal / Domain Context
The student works in the IHS Security department, giving direct operational exposure to how physical-security risk is actually assessed and acted on in Nigeria — not just an academic reading of ACLED data. This provides a ground-truth check on whether the model's flagged drivers (festivals, troop movements, ransom activity) match what security teams already watch for informally, and access to domain judgment for validating whether the model's outputs would be operationally usable rather than just statistically accurate.

## What Makes This Project Original
1. **Forecasting, not reporting.** ACLED, SBM, and NEXTIER publish retrospective incident logs and periodic narrative reports; this project outputs a live 7-day-ahead probability per LGA — nobody in this space currently forecasts forward at LGA granularity.
2. **Named drivers, not a black-box score.** Alongside the probability, the model surfaces *why* — festival, military operation, or nearby ransom payment — via an explainability layer, instead of a bare risk number.
3. **Fuses three signal types existing tools use separately.** Structured incident history (ACLED), unstructured real-time text (news + Twitter/X + Telegram via spaCy/SBERT), and relational actor-network data (who is targeting whom, via graph features) are combined in one pipeline; existing tools typically use only one of these.
4. **Decision-ready output, not a dataset.** The end product is a live map a state security council can act on directly, not a spreadsheet or PDF report.

## Research Questions
Primary:
> Can a space-time, graph-enhanced gradient boosting model predict the probability of a kidnapping or banditry incident occurring in a given Nigerian LGA within the next 7 days, using multi-source OSINT data?

Secondary:
1. Which contextual drivers (festivals, military operations, ransom payments in neighboring LGAs) contribute most to short-term risk, and can the model expose them as a reliable explainability layer?
2. Does adding actor-network (graph) features — who is targeting whom — improve predictive performance over incident-history and text features alone?
3. How does a forward-looking 7-day probability score compare, in usefulness and lead time, to ACLED's retrospective heat-map approach?

## Target Variable
**incident_prob_7d** — the probability that at least one kidnapping or banditry incident occurs in a given LGA within the next 7 days. Framed as binary classification with a continuous probability output (0–1), then bucketed into Low / Medium / High risk bands for the map display. A probability-of-occurrence framing was chosen over count regression because kidnapping/banditry events are rare and sparse at LGA-week level, making occurrence more learnable — and more operationally actionable — than predicting exact counts.

## Features (Model Inputs)
### Historical Incident Features (ACLED)
- Rolling incident counts and incident-type mix per LGA (7/14/30-day windows)
- Time since last incident in LGA
### Text / NLP-Derived Signals (Premium Times, Daily Trust, HumAngle, NPF press releases, Twitter/X, Telegram — via spaCy + SBERT)
- Armed-group entity mention counts
- Aggregated threat/sentiment score from news and social text
### Graph Features (PyTorch Geometric)
- Actor-network centrality (who is targeting whom, group affiliation clustering)
### Contextual / Event Features
- Festival or major religious-event flag in the prediction window
- Active military operation flag in/near LGA
- Ransom payment reported in a neighboring LGA (past 14 days)
### Geospatial Features (PostGIS)
- Spatial-lag risk score (weighted average of neighboring LGA risk)
- Distance to nearest high-risk LGA
### Engineered Features
- 7-day rolling incident rate per LGA
- Spatial lag of neighbor risk (from PostGIS)
- Event co-occurrence counts (e.g., military operation + prior incident in same window)

## Data
Source: Streaming multi-source OSINT ingested via Kafka — not a single static file. ACLED Nigeria is the structured backbone; Premium Times, Daily Trust, HumAngle, NPF press releases, X/Twitter, and Telegram supply unstructured text, converted into structured features via spaCy/SBERT.
Size: ACLED Nigeria security events (recommend pulling full history from ~2015 to present) across all 774 LGAs; unstructured text volume grows continuously through the Kafka stream.
Period: Rolling window, recommended start 2015–present for the structured backbone, refreshed continuously for text/social signals.
Known issues: geocoding is limited to LGA centroid precision; underreporting is likely in remote/high-risk LGAs; duplicate reporting occurs across news sources; social text carries language/dialect variance (Hausa, Pidgin, English); Telegram access is subject to ToS/API constraints; positive labels (an incident occurring) are rare per LGA-week, so class imbalance must be handled explicitly.

Feature dictionary (replaces a fixed Excel column list, since this is a streaming feature pipeline rather than a static spreadsheet):
| Column | Meaning | Type |
|---|---|---|
| lga_id | Unique identifier for each of Nigeria's 774 LGAs | Categorical |
| incident_count_7d | Rolling count of ACLED-logged incidents in past 7 days | Numeric |
| incident_prob_7d | Predicted probability of ≥1 incident in next 7 days | Numeric (0–1), **target** |
| actor_centrality | Graph centrality score of dominant armed actor near LGA | Numeric |
| festival_flag | Major festival/religious event falls within prediction window | Boolean |
| military_ops_flag | Announced military operation active in/near LGA | Boolean |
| ransom_payment_nearby | Reported ransom payment in a neighboring LGA (past 14 days) | Boolean |
| sentiment_score | Aggregated SBERT threat/sentiment score from news + social mentions | Numeric |
| spatial_lag_risk | Weighted average risk score of neighboring LGAs (PostGIS) | Numeric |
| entity_mentions | Count of spaCy-extracted armed-group entity mentions (past 7 days) | Numeric |

## Model Plan
Algorithms to compare:
1. **Baseline** — logistic regression on incident history only (mirrors ACLED's current descriptive approach)
2. **Main** — XGBoost gradient boosting on space-time + text + contextual features
3. **Graph-enhanced** — PyTorch Geometric actor-network embeddings feeding into the gradient boosting model
Evaluation metrics: **PR-AUC** as the primary metric (handles the class imbalance of rare incidents better than ROC-AUC), **Brier score** for probability calibration, **F1** at a chosen operating threshold, and **Recall@Top-K LGAs** to measure operational usefulness for a security council with limited attention.
Validation strategy: **rolling-origin time-series cross-validation** — train on past weeks, validate on future weeks, never random shuffling, to avoid leaking future information — plus a **spatial holdout** on a subset of LGAs to test geographic generalization. Metrics and thresholds will be iterated until PR-AUC and Recall@Top-K both stabilize on the holdout weeks.

## App / Output Design
An R Shiny dashboard in a **Claymorphism** visual style, designed for non-STEM users (state officials, security council members) rather than data scientists. Inputs are kept minimal: pick a state, then an LGA (dropdown or map click) — the date window defaults automatically to "next 7 days," no parameter tuning exposed. Output: a color-coded LGA risk map (Low / Medium / High), the underlying probability score, and a plain-language driver explanation generated via Claude (e.g., "Risk is elevated due to an upcoming festival and a recent ransom payment in a neighboring LGA") instead of raw feature-importance or SHAP plots.

## Project Phases
Active build is 3 weeks (Aug 11 – Sep 1); the remaining time to the Sep 15 deadline is held as buffer for supervisor feedback, rework, and final submission prep rather than counted as build time.

| Phase | Focus | Dates |
|---|---|---|
| 1 | Data foundation: ACLED API access, PostgreSQL+PostGIS+pgvector schema, Kafka pipeline skeleton, confirm Twitter/Telegram access constraints, ground-truth check against IHS Security domain knowledge | Aug 11 – 14 |
| 2 | NLP & feature engineering: spaCy/SBERT entity extraction, actor-network graph construction, festival/military-ops event calendars, PostGIS spatial-lag features | Aug 15 – 18 |
| 3 | Baseline & core model: logistic regression baseline, XGBoost space-time model, rolling-origin CV harness, MLflow experiment tracking | Aug 19 – 22 |
| 4 | Graph-enhanced model & explainability: PyTorch Geometric embeddings, driver-explainability layer, Claude narrative generation | Aug 23 – 26 |
| 5 | App build & validation: R Shiny Claymorphism dashboard, connect model to map, usability pass with a non-STEM proxy user, finalize PR-AUC / Brier / Recall@K on holdout | Aug 27 – 29 |
| 6 | Write-up, ethics/limitations, final QA | Aug 30 – Sep 1 |
| — | **Buffer**: supervisor review, revisions, final submission prep | Sep 2 – 15 |

## Dissertation Title Options
1. Forecasting Kidnapping and Banditry Risk in Nigeria: A Space-Time Graph-Boosting Approach
2. From Heat-Maps to Forecasts: LGA-Level Security Risk Prediction from Multi-Source OSINT
3. An Explainable OSINT Model for Short-Term Security Risk Forecasting in Nigeria

## Key Differentiators for Examiner
- **Forward-looking, not retrospective**: outputs a 7-day-ahead probability per LGA, where ACLED/SBM/NEXTIER report only what has already happened.
- **Explainable by design**: names the specific drivers behind each score (festival, military operation, nearby ransom payment) instead of a black-box number.
- **Multi-signal fusion**: combines structured incident history, real-time unstructured text, and actor-network relationships in one pipeline — existing tools use at most one of these.
- **Actionable at the point of use**: delivers a decision-ready map for a state security council, not a raw dataset or a periodic PDF report.

## Notes for Claude in New Chat
- Student works in IHS Security — treat this as a real operational grounding point, not just an academic framing device.
- Nigeria has 774 LGAs; this is the spatial unit of analysis throughout.
- The data pipeline is streaming/multi-source, not a single static Excel file — think in terms of a feature-store schema, not a spreadsheet.
- Handle OSINT/security data sensitively: do not surface raw actor identities or unaggregated threat intel in any public-facing output.
- Telegram/Twitter scraping is subject to ToS/API limits — treat this as a methodology risk to document, not a solved input.
- Deadline is September 15, 2026. Follow the 6-phase workflow defined above; keep phase order intact when helping with implementation.
- **Do not push to GitHub once the project reaches the dissertation-writing phase** — the student wants to avoid triggering similarity/plagiarism detection against a public repo history. Code, pipeline, and brief artifacts are fine to push; dissertation prose is not.
- The student wants an ongoing `.md` log of struggles, errors, limitations, and surprises for use in the dissertation's reflection/limitations section — start this once implementation work (Phase 1 onward) begins, not before.
