import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from dotenv import load_dotenv

EXAMPLE_DIR = Path(__file__).parent.resolve()
REPO_ROOT = EXAMPLE_DIR.parent.parent
KNOWLEDGE_DIR = EXAMPLE_DIR / "knowledge"
NARRATIVE_PATH = EXAMPLE_DIR / "BEFOREST LIFESTYLE SOLUTIONS- NARRATIVE 2025.docx.md"

load_dotenv(EXAMPLE_DIR / ".env")
load_dotenv(EXAMPLE_DIR / ".env.local", override=False)
load_dotenv(REPO_ROOT / ".env", override=False)
load_dotenv(REPO_ROOT / ".env.local", override=False)


def _convex_upsert_url() -> str:
    base_url = os.getenv("CONVEX_HTTP_ACTION_URL", "").strip()
    if base_url:
        if not base_url.endswith("/instagram/store-dm-event"):
            raise ValueError(
                "CONVEX_HTTP_ACTION_URL must end with /instagram/store-dm-event so the Convex base can be derived."
            )
        return base_url.replace("/instagram/store-dm-event", "/knowledge/upsert-entry")

    site_url = os.getenv("CONVEX_SITE_URL", "").strip()
    if site_url:
        return site_url.rstrip("/") + "/knowledge/upsert-entry"

    raise ValueError("CONVEX_HTTP_ACTION_URL or CONVEX_SITE_URL is required.")


def _shared_secret() -> str:
    secret = os.getenv("AGENT_SHARED_SECRET", "").strip()
    if not secret:
        raise ValueError("AGENT_SHARED_SECRET is required.")
    return secret


def _upsert_entry(url: str, secret: str, payload: dict[str, object]) -> None:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-agent-secret": secret,
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=20) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Convex upsert failed with {exc.code}: {detail}") from exc

    parsed = json.loads(raw)
    if parsed.get("ok") is not True:
        raise RuntimeError(f"Convex upsert returned ok=false: {raw}")


def _payload_for_markdown(path: Path) -> dict[str, object]:
    title = path.stem.replace("_", " ").replace("-", " ").title()
    return {
        "slug": f"seed-{path.stem.lower().replace('_', '-').replace(' ', '-')}",
        "title": title,
        "type": "fact",
        "summary": f"Seeded from {path.name}",
        "body": path.read_text(encoding="utf-8"),
        "tags": ["seed", "built-in"],
        "intentTags": [],
        "audienceTags": [],
        "priority": 0.8,
        "status": "approved",
        "sourceType": "markdown",
    }


def _payload_for_narrative(path: Path) -> dict[str, object]:
    return {
        "slug": "seed-beforest-narrative-2025",
        "title": "Beforest Narrative 2025",
        "type": "playbook",
        "summary": "Seeded long-form brand narrative document.",
        "body": path.read_text(encoding="utf-8"),
        "tags": ["seed", "narrative", "brand"],
        "intentTags": ["brand", "overview"],
        "audienceTags": ["prospect"],
        "priority": 0.6,
        "status": "approved",
        "sourceType": "markdown",
    }


def main() -> None:
    url = _convex_upsert_url()
    secret = _shared_secret()

    for path in sorted(KNOWLEDGE_DIR.glob("*.md")):
        _upsert_entry(url, secret, _payload_for_markdown(path))
        print(f"Synced {path.name}")

    if NARRATIVE_PATH.exists():
        _upsert_entry(url, secret, _payload_for_narrative(NARRATIVE_PATH))
        print(f"Synced {NARRATIVE_PATH.name}")


if __name__ == "__main__":
    main()
