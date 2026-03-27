# Beforest DM Agent

This repo contains the Beforest Instagram DM agent, the FastAPI service, and the Convex functions used for DM event persistence.

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
