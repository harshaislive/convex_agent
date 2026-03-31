# Repository Guidelines

## Project Structure & Module Organization
`src/` contains the application code. Key areas are `src/agents/` for agent definitions, `src/service/` for the FastAPI service, `src/client/` for the reusable client, `src/core/` for settings and shared runtime logic, and `src/schema/` for request/response models. Entry points such as `src/run_service.py` and `src/streamlit_app.py` are kept at the top of `src/`.

`tests/` mirrors the source layout with unit and integration coverage, including `tests/integration/` for docker-backed end-to-end checks. Supporting material lives in `docs/`, `docker/`, `media/`, and `data/`. Keep secrets out of Git; use `.env` and `privatecredentials/`.

## Build, Test, and Development Commands
Use `uv` for dependency management and task execution:

- `uv sync --frozen`: install locked dependencies into `.venv`.
- `uv run python src/run_service.py`: start the FastAPI service locally.
- `uv run streamlit run src/streamlit_app.py`: launch the Streamlit UI.
- `uv run pytest`: run the test suite.
- `uv run pytest tests/integration -v --run-docker`: run docker-dependent integration tests.
- `uv run ruff check . && uv run ruff format --check .`: lint and verify formatting.
- `uv run mypy src/`: run static type checks.
- `docker compose watch`: run the full local stack with live reload.

## Coding Style & Naming Conventions
Target Python 3.11+ with 4-space indentation and a maximum line length of 100. Ruff handles import sorting and formatting; run pre-commit before pushing with `uv run pre-commit run --all-files`. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and clear agent names matching their route or domain, such as `beforest_agent.py`.

## Testing Guidelines
Write tests with `pytest`; async tests are supported via `pytest-asyncio`. Place tests alongside the matching domain under `tests/` and name files `test_<feature>.py`. CI runs `pytest --cov=src/ --cov-report=xml`, so new work should include coverage for success paths and edge cases.

## Commit & Pull Request Guidelines
Recent commits use short, imperative summaries such as `Add stronger live search for Beforest sites`. Follow that pattern: one-line subject, present tense, specific scope. PRs should explain behavior changes, call out config or env updates, link related issues, and include screenshots when UI or Streamlit behavior changes. Confirm Ruff, Mypy, and Pytest pass before opening the PR.

## Security & Configuration Tips
Copy `.env.example` when setting up locally and never commit real credentials. Treat `privatecredentials/` as local-only storage for file-based secrets and certificates.
