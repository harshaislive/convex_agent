# Beforest DM Instagram Live Eval Runbook

## Scope
Evaluate `POST /beforest/reply` on the live endpoint (`https://agentig.devsharsha.live`) for Instagram DM readiness across partnerships, collectives, experiences, hospitality, and Bewild produce routing.

## Phase 1: 5-Query Smoke Gate (Completed on 2026-04-01)
Use this gate before any large run. Query set:
1. Creator collaboration ask
2. Current live experiences ask
3. Sustainability partnership ask
4. Family stay availability ask
5. Bewild produce browsing ask

Smoke outcome:
- API/auth: Pass (`200` on all 5)
- DM brevity: Pass (all replies under 220 chars)
- Routing: Pass in 4/5
- Critical failure: 1/5 on "current experiences" freshness

Critical note:
- The bot replied with experiences marked "currently live" but cited dates `Dec 14, 2025`, `Jan 4, 2026`, and `Jan 26, 2026`. On `April 1, 2026`, these are in the past, so this is a launch-blocking freshness/groundedness failure for this intent.

## Phase 2: 100-Query Evaluation Pack
Use the existing 100-query set in `docs/evals/Beforest_DM_Evaluation_Plan.md`. Keep category mix as written:
- Beforest overview: 15
- Collectives: 20
- Visiting/joining/staying: 15
- Influencers/creators: 10
- Partnerships: 10
- Event hosts/retreat facilitators: 10
- Current experiences: 10
- Bewild products: 5
- Routing/link requests: 3
- Edge cases: 2

## Scoring Framework (Per Query)
Required fields (pass/fail): `grounded_pass`, `routing_pass`, `hallucination_pass`, `dm_style_pass`, `actionability_pass`, `tone_pass`, `link_discipline_pass`.

Severity model:
- `P0` (blocker): hallucinated facts, stale "current/live" claims, wrong destination domain, false promise of approval.
- `P1` (major): unclear next step for commercial leads, overlong/rambling DM, weak route clarity.
- `P2` (minor): tone mismatch, unnecessary extra link, wording polish issues.

## Release Gates
Do not deploy until all are met:
- `P0 = 0` in the 100-query run
- Groundedness `>= 90/100`
- Hallucination-safe `>= 95/100`
- Routing `>= 90/100`
- DM style `>= 85/100`
- Actionability `>= 85/100`
- Category floor for "Current experiences" and "Partnership/Creator" buckets: `>= 90%` each

## Execution Workflow
1. Run smoke gate on live API.
2. If smoke has any `P0`, fix prompt/tooling first.
3. Run full 100-query batch.
4. Dual-review scoring (2 reviewers) on all `P0/P1` candidates.
5. Patch and rerun only failed rows.
6. Re-run complete 100-query regression before production rollout.

## Reporting
Store outputs with:
- `docs/evals/beforest_dm_eval_template.json` format
- One CSV row per query with short notes on every fail
- A top summary: pass rates by criterion, fail count by severity, and category-wise performance
