# CA system baseline (DAT608 Project 8) and what it means for the dissertation

Notes taken from the DAT608 Continuous Assessment build (Project 8, Group 4) supplied
on 2026-09-08, comprising the source zip and `HANDOFF.md`. Every figure below was
verified directly against the shipped artifacts in that archive, not taken from the
handoff text.

## 1. Verification status

All headline numbers in `HANDOFF.md` reproduce exactly from
`data/artifacts/model_comparison.csv`, `data/artifacts/model/metadata.json`,
`data/artifacts/hawkes_parameters.json` and `data/warehouse.db`. The handoff is a
reliable document.

Warehouse contents as shipped:

| Table | Rows |
|---|---|
| lga_registry | 774 |
| lga_adjacency | 4,677 |
| incidents | 6,875 (3,640 `banditry_kidnapping`, 3,235 `state_operation`) |
| predictions | 774 |
| prediction_drivers | 3,870 |
| threat_actor_edges | 588 |
| documents | 73 |
| alerts | 26 |
| chatter | 0 (no credentials) |
| lga_week_features | 0 (panel regenerated, not persisted) |

Shipped model: `hawkes-xgb-2026-08-15`, run `run-20260815T214454Z`, 39 features,
trained to 2021-12-31, tested on 2023-01-02 to 2024-12-30 (81,270 rows, 400 positives).

## 2. The finding that matters most: the data does not measure the stated problem

The proposal's Introduction states, citing SBM Intelligence (2025), that the North-West
accounts for over 60% of kidnapping incidents, led by Zamfara (1,203 abductions),
Kaduna (629), Katsina (566) and Sokoto (358).

The UCDP-trained system points somewhere else entirely.

UCDP `banditry_kidnapping` labels by state (top 6 of the training corpus):

| State | Events |
|---|---|
| Borno | 1,156 |
| Plateau | 400 |
| Benue | 376 |
| Taraba | 191 |
| Adamawa | 188 |
| Kaduna | 156 |

Zamfara, Katsina and Sokoto do not appear in the top 12.

Where the shipped model places its High and Severe tiers:

| State | High/Severe LGAs |
|---|---|
| Borno | 10 |
| Benue | 7 |
| Plateau | 5 |
| Oyo, Lagos, Enugu, Anambra | 1 each |

Zamfara, Katsina, Sokoto and Kaduna: **zero**.

The cause is documented in `HANDOFF.md` section 4.3: UCDP only records events that
produced fatalities. A kidnapping resolved by ransom payment with no deaths is close to
invisible to it, and that is precisely the event class this project targets. The model
is therefore predicting lethal organised violence (Boko Haram in Borno, communal and
farmer-herder killing in Benue and Plateau), not the kidnap-for-ransom economy.

This is a construct validity problem, not a data quality problem. UCDP is a canonical,
peer-reviewed source; it is simply the wrong instrument for this outcome.

It also reframes the supervisor's first correction. "Look for dataset with a more
verified source" is best answered not by finding a more credible source than UCDP, but
by adding one that codes abduction without a fatality threshold. ACLED does exactly
that, which is why `HANDOFF.md` 4.3 calls an ACLED key "the single highest-value change
available". The model needs retraining on the wider base, not rebuilding.

## 2b. Baseline measured, and the ACLED work now written

The finding above was turned into a repeatable check,
`scripts/check_northwest_coverage.py`, which prints the numbers rather than leaving
them to be recalled. Against the warehouse as shipped:

| Measure | UCDP only |
|---|---|
| North West share of banditry and kidnapping events | 6.2% |
| North West LGAs with any recorded event | 41 of 186 |
| North West LGAs at High or Severe | 0 |
| Zamfara / Katsina / Sokoto / Kebbi events | 16 / 3 / 6 / 2 |

The North East, by contrast, holds 46.6% of the corpus, 75% of its LGAs have a
recorded event, and Borno alone carries 10 High or Severe LGAs.

Four changes were made to the CA project to close this. They are described in full in
that project's `docs/acled_north_west.md`.

1. The ACLED connector was rewritten. ACLED retired the `key` plus `email` query
   parameter scheme in favour of OAuth at `acleddata.com/oauth/token`, and the data
   host moved to `acleddata.com/api/acled/read`. The connector as inherited would have
   returned 401 and looked like a credential fault. Credentials are now `ACLED_EMAIL`
   and `ACLED_PASSWORD` against a free myACLED account.
2. ACLED events are classed on the same rule as UCDP, so the merged corpus carries one
   definition of the target rather than two.
   `Abduction/forced disappearance` enters the label class.
3. Cross-source deduplication was added (`src/pau_risk/dedupe.py`), applied when
   incidents are read rather than when they are written, so the raw layer stays
   auditable. UCDP and ACLED both code the same lethal attacks, and `event_id` is built
   per source, so without this one attack would count twice. The weekly label survives
   double counting; the lagged count features and the Hawkes branching ratio do not.
4. A verification script that answers the North West question directly, before and
   after, and writes the answer to `data/artifacts/northwest_coverage.md`.

The ACLED endpoint was not reachable from the environment this was written in, so the
matcher was validated against a synthetic merged corpus built from the real UCDP data.
It removed 3,404 of 3,436 planted duplicates (99.1%) and wrongly removed none of the
3,000 planted ACLED-only non-fatal events, which is the property that matters, since
those records are the whole reason for adding the source. `tests/test_dedupe.py` pins
both directions of the rule and the full CA suite (74 tests) passes.

What remains is hers to run: supply the myACLED credentials, ingest, then rebuild the
panel, retrain and rescore. The shipped tier thresholds and Hawkes parameters belong to
the narrower corpus, so the operational headline (top 20 LGAs reaching 36% of the areas
attacked that week) has to be recomputed rather than carried over.

## 3. Architecture divergence from the revised proposal

The built system and the corrected proposal describe different models.

| | Built (CA system) | Proposal after supervisor corrections |
|---|---|---|
| Sparsity handling | Hawkes point process + isotonic calibration; explicitly no resampling, `scale_pos_weight = 1.0` | Two-stage Hurdle / zero-inflated model |
| Relational structure | Multivariate Hawkes over the adjacency graph, feeding features into XGBoost | Spatio-Temporal GNN via PyTorch Geometric Temporal |
| Combination | Single hybrid (Hawkes features into XGBoost) | Stacking ensemble, 3 base learners + meta-learner |
| Horizons | 7 days only | 7, 14 and 30 day sensitivity comparison |

Measured result from the built system, worth carrying into any write-up because it cuts
against the assumption that the tree layer does the work:

| Model | ROC AUC | Recall@20 |
|---|---|---|
| Persistence baseline | 0.549 | 0.109 |
| Historical rate | 0.811 | 0.322 |
| Hawkes alone | 0.910 | 0.345 |
| XGBoost without Hawkes | 0.883 | 0.327 |
| Hybrid, calibrated (shipped) | 0.923 | 0.355 |

The Hawkes process alone outperforms gradient boosting alone. The tree layer contributes
calibration and per-row explainability rather than raw ranking. `HANDOFF.md` instructs
that this be reported honestly rather than hidden, which is the right call.

Fitted Hawkes parameters: branching ratio 0.504, self share 0.815, decay 0.0315 per day
(half life 22.0 days), log likelihood -14,351.79 over 2,445 events and 774 dimensions.

Operational headline: covering the top 20 LGAs (2.6% of the country) reaches 36% of the
areas attacked that week, against 11% for the current practice of re-covering last
week's incident list.

## 4. Open decisions

1. Which architecture does the dissertation describe: the built Hawkes hybrid, the
   corrected proposal's hurdle + ST-GNN + ensemble, or the built system extended toward
   the corrections? This needs the supervisor's view before Chapter 3 is finalised.
2. ~~Is ACLED being added?~~ Answered: yes, and the code is written (section 2b). What
   is still open is whether the retrained geography is reported as the headline result
   or as a robustness check alongside the UCDP-only model. Reporting both is the
   stronger thesis, because the gap between them is itself the finding: the choice of
   source, not the choice of model, decides where the system says the risk is.
3. Reuse and attribution. The CA build is group coursework (Project 8, Group 4, seven
   named contributors, deployed under a different account). The dissertation declaration
   asserts sole authorship and no prior submission. Confirm with the supervisor what may
   be carried over and how it must be declared.

## 5. Convention conflicts to watch

- The CA project's house style specifies **APA 7th edition**. The Pan-Atlantic MSc
  template and this repository's `CLAUDE.md` specify **APA 6th**. Do not mix them.
- Both agree on no em dashes, British English, and no fabricated sources.

## 6. Data

`data/raw/ged251-csv/GEDEvent_v25_1.csv` is UCDP GED v25.1, roughly 239 MB. It exceeds
GitHub's 100 MB per-file limit and is reproducible from
`https://ucdp.uu.se/downloads/ged/ged251-csv.zip`, so it stays out of the repository and
is fetched by the ingestion code. Per `HANDOFF.md` 4.2, the UCDP REST API returns 401
without a token; the bulk CSV needs no credential.
