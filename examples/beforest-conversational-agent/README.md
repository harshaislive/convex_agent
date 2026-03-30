# Beforest Conversational Agent

A starter conversational agent for `Beforest.co` built on LangChain's `deepagents` framework.

This example is designed for a practical Beforest use case instead of a generic demo:

- answer common visitor questions about Beforest and its collectives
- stay grounded in a Convex-backed knowledge base with local fallback
- guide people to the right Beforest destination link when they want to act

The bundled knowledge files are based on a snapshot of `https://beforest.co` reviewed on March 27, 2026. Subscription status, availability, pricing, and operating details can change, so the agent is instructed to guide people to the appropriate Beforest destination instead of guessing.

## Quick Start

```bash
cd examples/beforest-conversational-agent
cp .env.example .env
```

Add your API settings to `.env`.

For your Azure AI Foundry deployment, use:

```bash
BEFOREST_MODEL_PROVIDER=azure_ai
AZURE_AI_API_KEY=...
AZURE_AI_BASE_URL=https://azureuserbeforest.services.ai.azure.com/openai/v1/
AZURE_AI_MODEL=your_deployed_model_name
```

`AZURE_AI_MODEL` must be the deployed model name exposed by your Azure resource. It is not inferable from the preview chat completions URL alone, so you still need to supply that value explicitly.

If you want to use standard OpenAI instead, switch back to:

```bash
BEFOREST_MODEL_PROVIDER=openai
OPENAI_API_KEY=...
BEFOREST_MODEL=openai:gpt-4.1-mini
```

Install and run:

```bash
uv run python agent.py --interactive
```

To run it as an HTTP service:

```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000
```

Then send requests like:

```bash
curl -X POST http://localhost:8000/reply \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How can one be part of this community?",
    "user_id": "123456789",
    "subscriber_data": {"contact_id": 987654321, "first_name": "Harsha"},
    "push_to_manychat": true
  }'
```

To test the Beforest page browser directly:

```bash
uv run python check_browser.py https://beforest.co "collectives"
uv run python check_browser.py https://experiences.beforest.co "retreat"
```

One-shot usage:

```bash
uv run python agent.py "What is a Beforest collective?"
uv run python agent.py "I live in Hyderabad and want to understand how membership works."
```

For Instagram-style session continuity, pass a stable `user_id`. The example will use `ig:<user_id>` as the thread id automatically:

```bash
uv run python agent.py --user-id 123456789 "How can one be part of this community?"
uv run python agent.py --interactive --user-id 123456789
```

You can also pass subscriber metadata as JSON:

```bash
uv run python agent.py --user-id 123456789 --subscriber-data "{\"first_name\":\"Harsha\",\"city\":\"Bengaluru\"}" "What is 10%?"
```

To push the final reply back to a ManyChat contact, pass the ManyChat subscriber/contact id and set `MANYCHAT_API_TOKEN`:

```bash
uv run python agent.py --user-id 123456789 --manychat-subscriber-id 987654321 "How do I join?"
```

If your `subscriber_data` already contains `subscriber_id`, `contact_id`, `manychat_subscriber_id`, or `manychat_contact_id`, the example will use that automatically and you do not need `--manychat-subscriber-id`.

To persist inbound/outbound DM events to Convex, set:

```bash
CONVEX_HTTP_ACTION_URL=https://knowing-raccoon-518.convex.site/instagram/store-dm-event
AGENT_SHARED_SECRET=replace_with_a_shared_secret
```

This service will POST directly to the Convex HTTP action we just deployed and store records in `instagramConversations`.

The same Convex deployment now also exposes protected knowledge routes:

- `GET /knowledge/search`
- `GET /knowledge/entries`
- `POST /knowledge/upsert-entry`

The Python retrieval tool uses these automatically when `CONVEX_HTTP_ACTION_URL` and `AGENT_SHARED_SECRET` are set.

To seed the existing markdown knowledge into Convex:

```bash
uv run python sync_knowledge_to_convex.py
```

Run that from `examples/beforest-conversational-agent/` after deploying the updated Convex schema and functions.

URL import in the knowledge center now uses a two-step fetch path:

1. fast HTTP fetch for simple pages
2. Playwright Chromium fallback for JS-heavy or bot-protected pages

If you deploy with Docker, rebuild the image so Playwright and Chromium are installed.

## What This Example Includes

```text
beforest-conversational-agent/
├── agent.py
├── tools.py
├── AGENTS.md
├── knowledge/
│   ├── overview.md
│   ├── collectives.md
│   └── contact_and_next_steps.md
├── skills/
│   ├── company-qa/
│   │   └── SKILL.md
└── data/
    └── optional local files created only if you extend the example
```

## How It Works

The example uses three layers:

1. `AGENTS.md` defines the agent persona, guardrails, and routing rules.
2. `knowledge/` stores local fallback files and seed content.
3. `tools.py` adds:
   - `search_beforest_knowledge` for Convex-first retrieval with metadata-aware ranking
   - `search_beforest_experiences` for live lookups against `experiences.beforest.co`
   - `browse_beforest_page` for fetching a specific `beforest.co` or Beforest subdomain page on demand

`agent.py` exposes both:

- `create_beforest_agent()` for reuse in an app or API
- an interactive CLI for local testing
- optional push-back to ManyChat via `sendContent` after the final reply is generated
- a FastAPI app in `server.py` with `POST /reply` and `GET /health`

## Production Notes

This example uses `FilesystemBackend` for local development. For a real web app, avoid giving a public-facing agent unrestricted filesystem access. Swap to a safer backend such as `StateBackend`, `StoreBackend`, or a sandboxed backend and keep only the tools you actually need.

## Suggested Next Steps For Beforest

- replace the static knowledge files with content pulled from the website CMS
- add multilingual support if your visitors ask in multiple Indian languages
- mount the agent behind a website chat widget or WhatsApp handoff flow


