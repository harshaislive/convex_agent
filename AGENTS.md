# Repository Guidelines

## Project Structure & Module Organization
Core application code lives in `src/`. Use `src/agents/` for agent implementations and agent-specific helpers, `src/service/` for the FastAPI API layer, `src/client/` for the reusable client, `src/core/` for settings and LLM wiring, `src/schema/` for shared models, and `src/voice/` or `src/memory/` for subsystem code. Entry scripts such as `src/run_service.py`, `src/run_client.py`, and `src/streamlit_app.py` stay at the top of `src/`.

Tests mirror the source tree under `tests/` and include focused suites like `tests/service/`, `tests/agents/`, and `tests/integration/`. Reference material lives in `docs/`, container assets in `docker/`, and local-only secrets in `privatecredentials/`.

## Build, Test, and Development Commands
- `uv sync --frozen`: install the locked dependencies into `.venv`.
- `uv run python src/run_service.py`: start the FastAPI service locally.
- `uv run streamlit run src/streamlit_app.py`: run the Streamlit chat UI.
- `uv run pytest`: run the default test suite.
- `uv run pytest tests/integration -v --run-docker`: run Docker-backed end-to-end tests.
- `uv run ruff check .`: lint and sort imports.
- `uv run mypy src/`: run static type checks.
- `docker compose watch`: run the full local stack with live reload.

## Coding Style & Naming Conventions
Target Python 3.11+ with 4-space indentation and a 100-character line limit. Ruff is the primary style gate; keep imports normalized and avoid unused symbols. Use `snake_case` for modules, files, functions, and variables, `PascalCase` for classes, and descriptive agent filenames such as `beforest_agent.py` or `github_mcp_agent.py`.

## Testing Guidelines
Use `pytest` with `pytest-asyncio` for async coverage. Place tests beside the matching domain and name files `test_<feature>.py`. New behavior should include success-path and failure-path assertions; service changes should usually add request/response coverage.

## Commit & Pull Request Guidelines
Recent history uses short imperative subjects such as `Clamp Beforest DM replies` and `Fix ManyChat URL cleanup helper`. Keep commit titles specific and present tense. PRs should summarize behavior changes, note any `.env` or credential impacts, link related issues, and include screenshots for Streamlit or UX changes.

## Security & Configuration Tips
Copy `.env.example` to `.env` for local setup and never commit real secrets. Treat `privatecredentials/` as local-only storage for certificates and provider credential files.
