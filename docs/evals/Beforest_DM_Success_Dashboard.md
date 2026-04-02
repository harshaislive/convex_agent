# Beforest DM Success Audit And Dashboard Plan

Audit date: 2026-04-02

## Why this exists

The Beforest DM agent already does more than answer questions. It routes people into collectives, stays, experiences, products, creator conversations, and partnerships. To measure whether it benefits the brand, we need a dashboard that combines:

- reply quality
- operational reliability
- lead generation
- downstream conversion
- brand safety

This document separates what already exists in the codebase from what still needs instrumentation.

## Current audit

### What is already working in the product

- `POST /beforest/reply` is the main DM-friendly reply endpoint.
- Outline is the primary knowledge source for factual brand answers.
- Live Beforest properties are searchable for current page-level routing.
- Current-experience answers have an explicit freshness guard. If the reply includes past dates, the service rewrites the answer to a safe route to `https://experiences.beforest.co/`.
- Replies are clamped to Instagram-DM style limits: short, usually 1 to 2 sentences, and capped to roughly 220 characters where possible.
- The service stores DM events in Convex, including inbound message, reply text, thread metadata, session state, and automation state.
- Session tracking already classifies conversations into `creator`, `partnership`, `event`, `stay`, `experience`, `product`, `collective`, or `general`.
- Human handover is already supported through `bot`, `human`, and `paused` automation states.
- There is already a lightweight Beforest Ops inbox at `/admin/beforest` for search and handover control.
- ManyChat delivery is already supported, including buttonized links and UTM enrichment on Beforest and Typeform links.
- Optional Langfuse tracing and LangSmith feedback hooks already exist at the service layer.

### Important shipped changes in the current repo history

Recent commits on 2026-04-01 and 2026-04-02 show a clear product direction:

| Date | Commit | Change |
| --- | --- | --- |
| 2026-04-02 | `d4c987a` | Hardened experiences sync failure path |
| 2026-04-01 | `0024c4d` | Sped up experiences Outline sync |
| 2026-04-01 | `15043ae` | Added ManyChat tracking for Beforest page links |
| 2026-04-01 | `4736f65` | Synced Beforest experiences into Outline |
| 2026-04-01 | `d49ce5a` / `c89095e` / `af92ea9` / `562ae6e` / `c3fcfbe` / `cf5fe50` / `37bda4c` | Built and refined the Beforest admin inbox and assets |
| 2026-04-01 | `7630710` | Fixed favicon delivery |

### Uncommitted artifacts present in the workspace

These files exist in the working tree but are not committed yet:

- `docs/evals/beforest_dm_smart10_2026-04-01.csv`
- `scripts/run_beforest_smart10_eval.py`
- `docs/Beforest_AI_DM_System_Explainer.html`
- `9ea3af8a-4270-4a65-bf03-cd37b758822a.jpeg`

The first two are directly useful for measurement because they add a targeted live scorecard.

## Baseline from existing eval assets

### Smoke gate on 2026-04-01

The 5-query smoke run found one launch-blocking issue:

- 1 `P0`: the bot described experiences as "currently live" while listing dates in December 2025 and January 2026.

That exact failure is documented in `Beforest_DM_Instagram_Live_Runbook.md` and is the right kind of blocker to track permanently in the dashboard.

### Smart10 live scorecard on 2026-04-01

The targeted 10-query live run shows the current quality baseline:

| Metric | Result |
| --- | --- |
| Overall PASS rows | 7 / 10 |
| `P0` rows | 1 |
| `P1` rows | 2 |
| Groundedness | 90% |
| Routing | 80% |
| Hallucination-safe | 90% |
| DM style | 90% |
| Actionability | 90% |
| Tone | 90% |
| Link discipline | 90% |

Observed failures:

- Query 2 returned HTTP `500` for "What is the next experience?"
- Query 4 did not route a low-intent Bhopal ask to the Bhopal collective page.
- Query 5 sent the correct Typeform but missed UTM tracking in the reply text being evaluated.

### 100-query live run on 2026-04-01

The 100-query CSV is not manually scored yet, but it is still useful as an operational baseline:

| Metric | Result |
| --- | --- |
| Rows | 100 |
| HTTP `200` | 99 |
| HTTP `500` | 1 |
| Blank replies | 1 |
| Median latency | 9034.8 ms |
| P95 latency | 23024.6 ms |
| Average reply length | 163.5 chars |
| P95 reply length | 213 chars |
| Average links per reply | 0.50 |
| Max links in a reply | 2 |

The only logged `500` in that run was on query 100, "Tell me everything about Beforest in detail."

## What we can measure today without new backend work

The repo already gives us enough signal for a first dashboard with derived metrics:

- inbound DM volume by day from Convex events
- unique contacts by day or week from `contactId`
- reply rate from `agentReplied`
- suppression rate when automation is `human` or `paused`
- handover mix by status
- session mix by inferred `session_type`
- median and P95 latency from eval runs
- average reply length and link count from eval runs
- knowledge health from `/health/knowledge`
- ManyChat route share from links embedded in replies

This is enough for an operations dashboard, but not enough for a brand-impact dashboard.

## What is missing for true brand-impact measurement

Right now the service mostly stores conversation events, not business outcomes. These gaps block a serious ROI view:

- No stored reply-level metric for whether a route was sent to `experiences`, `hospitality`, `Bewild`, `contact`, or a collective page.
- No stored boolean for whether the freshness rewrite fired.
- No stored ManyChat delivery status, fallback-to-plain-text status, or delivery failure reason.
- No stored CTA click, form-start, form-submit, booking-start, or booking-confirmed event.
- No stored lead qualification status such as `new`, `qualified`, `disqualified`, `won`, or `lost`.
- No stored human handover reason taxonomy.
- No stored explicit model/tool usage summary per DM.
- No aggregate table for eval results over time.
- The 100-query live CSV on 2026-04-01 has raw outputs but no pass/fail scoring columns filled in.

Without those fields, we can measure activity and quality, but not actual brand lift, revenue lift, or workload reduction with confidence.

## Dashboard to build

The right answer is not one chart-heavy page. It should be four focused views.

### 1. Executive summary

Audience: founder, growth, brand lead

Primary questions:

- Is the agent helping the brand this week?
- Is it generating qualified demand?
- Is it reducing manual work without damaging quality?

Top KPIs:

| KPI | Definition |
| --- | --- |
| Conversations handled | Count of inbound DM events |
| Unique contacts | Distinct `contactId` values |
| Bot-handled rate | Share of conversations not suppressed by `human` or `paused` |
| Qualified lead rate | Share of conversations that reached a tracked lead outcome |
| Assisted conversion rate | Share of contacts that clicked and later completed a form, booking, or contact action |
| Human-hours saved | Estimated from bot-resolved conversations multiplied by average manual handling time |
| Brand safety incidents | Count of `P0` eval failures plus production routing/freshness incidents |

### 2. Demand and brand impact

Audience: growth, partnerships, creator team, hospitality team

Primary questions:

- What demand is the bot creating?
- Which parts of the brand are benefiting most?
- Which CTAs actually move people deeper into the funnel?

Charts and tables:

- leads by intent: `creator`, `partnership`, `event`, `stay`, `experience`, `product`, `collective`
- route destination volume: `experiences`, `hospitality`, `bewild`, `contact`, `collective page`, `typeform`
- tracked link CTR by destination
- Typeform start rate and completion rate by collective
- contact-form submit rate
- hospitality booking-start rate
- repeat engagement rate within 7 days
- top-performing entry questions by downstream conversion

North-star metrics for brand benefit:

- qualified leads generated
- cost per qualified lead compared with manual DM handling
- assisted bookings or assisted inquiries attributable to the DM bot
- creator and partnership inquiries sourced by the bot
- share of users routed to the correct property on the first bot answer

### 3. Quality and trust

Audience: product, prompt owner, QA

Primary questions:

- Is the bot accurate, fresh, concise, and on-brand?
- Which intents are degrading?
- Are we at risk of sending users to the wrong place?

Charts and tables:

- eval pass rate by criterion over time
- eval pass rate by category over time
- `P0`, `P1`, `P2` count by week
- freshness rewrite count by day
- wrong-route incidents by destination
- replies over 220 characters
- replies with more than one link
- top failed queries from Smart10 and 100-query regression
- median and P95 latency by intent

Release gate tiles:

- groundedness target `>= 90%`
- hallucination-safe target `>= 95%`
- routing target `>= 90%`
- DM style target `>= 85%`
- actionability target `>= 85%`
- current experiences category target `>= 90%`
- creator and partnership category target `>= 90%`

### 4. Ops and handover

Audience: ops team

Primary questions:

- Where is the bot failing operationally?
- Which conversations need human intervention?
- Is the inbox under control?

Charts and tables:

- suppressed conversations by day
- current `bot` / `human` / `paused` contact counts
- handover rate by intent
- unresolved open sessions older than 30 minutes
- ManyChat delivery failure rate
- plain-text fallback rate after button payload rejection
- Convex write failure count
- Outline health and cache age
- contacts with repeated follow-ups and no resolution

## Recommended KPI formulas

Use these definitions so the dashboard does not drift.

| KPI | Formula |
| --- | --- |
| Bot-handled rate | `bot_replied_conversations / total_conversations` |
| Suppression rate | `suppressed_conversations / total_conversations` |
| Correct first-route rate | `conversations_with_successful_primary_route / routeable_conversations` |
| Qualified lead rate | `qualified_leads / total_conversations` |
| CTA CTR | `contacts_with_click / contacts_sent_that_cta` |
| Typeform completion rate | `contacts_with_typeform_submit / contacts_sent_typeform` |
| Assisted booking rate | `contacts_with_booking_event / contacts_routed_to_hospitality_or_experiences` |
| Human salvage rate | `human_handovers_that_end_in_qualified_outcome / total_human_handovers` |
| Human-hours saved | `bot_resolved_conversations * assumed_manual_minutes_per_conversation / 60` |
| Brand safety incident rate | `(P0_prod_incidents + P0_eval_failures) / total_conversations_or_eval_rows` |

## Event model to add

Add a compact analytics object to the Convex event payload or write it to a dedicated metrics table. These fields are the minimum needed:

| Field | Why it matters |
| --- | --- |
| `intent` | Tie performance to creator, partnership, stay, experience, product, or collective demand |
| `route_target` | Know where the bot sent the user |
| `route_url` | Tie downstream clicks and conversions back to a specific destination |
| `freshness_rewrite_applied` | Track how often the safety rewrite saves a bad live-experiences answer |
| `reply_chars` | Monitor DM brevity in production |
| `link_count` | Monitor link discipline in production |
| `queued_to_manychat` | Separate synchronous replies from background delivery |
| `manychat_delivery_status` | Measure actual delivery reliability |
| `manychat_fallback_used` | Catch button payload issues |
| `session_continued` | See whether follow-up handling is helping or confusing users |
| `session_status_after_reply` | Track `open`, `awaiting_confirmation`, `solved`, `auto_closed` |
| `primary_cta_type` | Distinguish `typeform`, `contact`, `hospitality`, `experiences`, `bewild`, `collective_page` |
| `lead_stage` | Move from activity metrics to funnel metrics |
| `handover_reason` | Explain why ops had to step in |

## External systems to connect

To measure actual brand benefit, the dashboard has to combine the service with downstream data:

- Convex for conversation events and session state
- ManyChat for send outcomes
- web analytics for UTM clicks
- Typeform for starts and submissions
- contact inbox or CRM for qualified creator and partnership leads
- hospitality and experiences booking systems for assisted conversions
- Langfuse or LangSmith for trace-level debugging and human feedback

If there is no CRM yet, a simple interim table with `lead_stage`, `owner`, `outcome`, and `estimated_value` is enough to start.

## Suggested dashboard layout

Use a single Beforest Ops Analytics page with four tabs:

1. `Overview`
2. `Demand`
3. `Quality`
4. `Ops`

Recommended top-row cards:

- conversations this week
- unique contacts this week
- bot-handled rate
- qualified leads
- assisted conversions
- P0 incidents
- median latency
- human-hours saved

Recommended filter bar:

- date range
- intent
- route target
- automation state
- source channel

## Build order

### Phase 1: dashboard from current data

- Read Convex recent conversations and event history.
- Add an aggregate view for volume, handover mix, session mix, and suppression.
- Add eval scorecards using the existing smoke and Smart10 CSVs.
- Add technical baseline charts from the 100-query CSV.

Outcome:

- good operations visibility
- limited brand-impact visibility

### Phase 2: add production analytics fields

- Save `intent`, `route_target`, `route_url`, `reply_chars`, `link_count`, `freshness_rewrite_applied`, `queued_to_manychat`, and ManyChat delivery results on every reply.
- Store every eval run in a structured table instead of only CSV files.

Outcome:

- trustworthy quality dashboard
- first real routing-performance dashboard

### Phase 3: connect downstream conversion systems

- ingest UTM click data
- ingest Typeform starts and submissions
- ingest contact-form submissions
- ingest booking starts and confirmed bookings
- add lead stage and estimated value

Outcome:

- actual brand and revenue impact measurement

## Immediate recommendations

- Re-run the Smart10 and full 100-query evaluations after the 2026-04-02 hardening changes.
- Manually score the existing 100-query run so it stops being only a latency artifact.
- Add structured analytics fields to the Convex event payload before building charts.
- Treat `current experiences` and `creator/partnership` as protected intents with permanent release-gate monitoring.
- Build the first dashboard from Convex plus eval CSVs, but do not present it as ROI reporting until downstream conversion events are wired in.

## Bottom line

The agent is already strong enough to justify an operations and quality dashboard today. It is not yet instrumented well enough to prove full brand benefit or commercial impact. The missing step is not more prompting. It is outcome instrumentation:

- what intent came in
- where the bot routed the user
- whether the user clicked
- whether the user converted
- whether human intervention was still needed

Once those are tracked, the dashboard can move from "the bot answered messages" to "the bot generated qualified demand and reduced manual workload without hurting trust."
