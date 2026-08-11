# Data Access Notes (Phase 1)

Working notes on what's actually needed to get each data source flowing,
and the constraints/risks each one carries. Update this as access is
confirmed or constraints change.

## ACLED
- Free registration required: https://acleddata.com/register/
- `src/ingestion/acled_client.py` implements the classic `key` + `email`
  query-param auth against `api.acleddata.com/acled/read`. ACLED has
  moved some newer accounts to an OAuth bearer-token flow — **confirm
  which applies to this account at registration** and update the client
  if needed; this was not verified from the build environment (outbound
  access to `api.acleddata.com` is blocked by this session's network
  policy — see `PROJECT_LOG.md`).
- Rate limits and historical depth (how far back Nigeria data goes)
  should be confirmed once a key is issued.

## Premium Times / Daily Trust / HumAngle / NPF press releases
- No formal API for any of these — Phase 2 will need RSS feeds where
  available, or scraping with attention to each site's `robots.txt`
  and terms of use.
- HumAngle in particular covers security/conflict reporting closely
  aligned with this project's scope — worth checking for a data
  partnership or export before building a scraper.

## X / Twitter
- Requires a developer account and API tier; free tier has tight rate
  limits and limited historical search, which may not be sufficient
  for a project needing security-analyst accounts' full posting history.
- Action item: confirm which API tier is actually accessible/affordable
  before Phase 2 design assumes real-time streaming access.

## Telegram
- Telegram's Bot API and MTProto client API can both read public
  channels, but scraping content and bulk-storing it engages Telegram's
  ToS — review before building the Phase 2 producer, and avoid storing
  or displaying raw identifying content from private/semi-private
  channels.

## IHS Security ground-truth check (action item — not something an AI
assistant can do)
The brief's personal/domain-context rationale depends on validating the
model's flagged drivers (festival, military operation, nearby ransom
payment) against what IHS Security's own team already watches for
informally. This needs the student to have that conversation directly —
recommend doing it early (Phase 1 or 2) so it can inform feature
engineering rather than just validate it after the fact at the end.

## LGA reference data
Do not hand-type the 774 LGA list — source it from an authoritative
boundary dataset (e.g. OCHA Nigeria admin boundaries / NBS shapefiles)
and load it into the `lga` table via a one-off import script. Hand-typed
admin lists are a common source of silent geocoding errors.
