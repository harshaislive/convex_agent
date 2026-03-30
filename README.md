# Beforest DM Agent

This repo contains the Beforest Instagram DM agent, the FastAPI service, and the Convex functions used for DM event persistence and knowledge retrieval.

## Coolify

Use the included `Dockerfile`.

Port:
- `8000`

Health check:
- `/health`

Required environment variables:
- `BEFOREST_MODEL_PROVIDER`
- `AZURE_AI_API_KEY`
- `AZURE_AI_BASE_URL`
- `AZURE_AI_MODEL`
- `MANYCHAT_API_TOKEN`
- `MANYCHAT_API_BASE_URL`
- `MANYCHAT_CHANNEL`
- `CONVEX_HTTP_ACTION_URL`
- `AGENT_SHARED_SECRET`
- `BEFOREST_HTTP_TIMEOUT_SECONDS`
- `KNOWLEDGE_CENTER_PASSWORD`

## Knowledge Center

The app now includes a private knowledge workspace at `/knowledge-center`.

Use it to:
- paste team-maintained Markdown
- ingest page URLs into Markdown snapshots
- optionally fetch protected pages with a cookie header or authorization header

Notes:
- uploaded knowledge is stored under `examples/beforest-conversational-agent/knowledge_center/`
- ingested docs are included in `search_beforest_knowledge`, so the DM agent can use them in replies
- set `KNOWLEDGE_CENTER_PASSWORD` before using the UI

## Convex Knowledge Layer

The repo now includes Convex-backed knowledge storage and search:

- `knowledge_entries` stores structured agent knowledge
- `GET /knowledge/search` is the agent retrieval endpoint
- `POST /knowledge/upsert-entry` is the write endpoint for sync/admin flows

The example agent will use Convex knowledge first when `CONVEX_HTTP_ACTION_URL` and `AGENT_SHARED_SECRET` are configured, and fall back to local markdown files otherwise.
