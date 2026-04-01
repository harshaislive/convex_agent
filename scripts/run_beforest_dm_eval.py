#!/usr/bin/env python3
"""Run Beforest DM evaluation queries against /beforest/reply and export CSV."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib import error, request

QUERY_LINE_RE = re.compile(r"^(\d+)\.\s+(.*\S)\s*$")
CATEGORY_LINE_RE = re.compile(r"^###\s+(.*?)(?:\s+\(\d+\))?\s*$")
LINK_RE = re.compile(r"https?://[^\s)]+")
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


@dataclass
class EvalQuery:
    query_id: int
    category: str
    text: str


def load_env_file(env_file: Path) -> None:
    if not env_file.exists():
        return
    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if value and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def parse_queries(markdown_path: Path) -> list[EvalQuery]:
    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    queries: list[EvalQuery] = []
    in_queries_section = False
    current_category = ""

    for line in lines:
        if line.strip() == "## 100 Queries":
            in_queries_section = True
            continue
        if not in_queries_section:
            continue
        if line.startswith("## "):
            break

        category_match = CATEGORY_LINE_RE.match(line.strip())
        if category_match:
            current_category = category_match.group(1).strip()
            continue

        query_match = QUERY_LINE_RE.match(line.strip())
        if query_match and current_category:
            queries.append(
                EvalQuery(
                    query_id=int(query_match.group(1)),
                    category=current_category,
                    text=query_match.group(2).strip(),
                )
            )

    queries.sort(key=lambda item: item.query_id)
    return queries


def build_request_payload(query: EvalQuery, run_id: str) -> dict[str, Any]:
    return {
        "message": query.text,
        "thread_id": f"{run_id}-{query.query_id:03d}",
        "user_id": f"eval-user-{query.query_id:03d}",
        "push_to_manychat": False,
    }


def parse_retry_after(headers: Any) -> float | None:
    retry_after = headers.get("Retry-After")
    if not retry_after:
        return None
    retry_after = retry_after.strip()
    try:
        return max(0.0, float(retry_after))
    except ValueError:
        return None


def call_beforest_reply(
    *,
    endpoint: str,
    auth_secret: str,
    payload: dict[str, Any],
    timeout_seconds: float,
    min_interval_seconds: float,
    max_retries: int,
    backoff_base_seconds: float,
    backoff_max_seconds: float,
    state: dict[str, float],
) -> tuple[int, dict[str, Any], str, int, float]:
    attempt = 0
    while True:
        now = time.monotonic()
        wait_seconds = max(0.0, state["next_allowed_at"] - now)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
        state["next_allowed_at"] = time.monotonic() + min_interval_seconds

        started = time.perf_counter()
        body = json.dumps(payload).encode("utf-8")
        http_request = request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {auth_secret}",
                "Content-Type": "application/json",
            },
        )
        try:
            with request.urlopen(http_request, timeout=timeout_seconds) as response:  # noqa: S310
                raw = response.read().decode("utf-8", errors="replace")
                elapsed_ms = (time.perf_counter() - started) * 1000
                parsed = json.loads(raw) if raw.strip() else {}
                if not isinstance(parsed, dict):
                    parsed = {"_raw": raw}
                return response.status, parsed, "", attempt, elapsed_ms
        except error.HTTPError as exc:
            raw_error = exc.read().decode("utf-8", errors="replace")
            elapsed_ms = (time.perf_counter() - started) * 1000
            retry_after_seconds = parse_retry_after(exc.headers)
            if exc.code in RETRYABLE_STATUS_CODES and attempt < max_retries:
                if retry_after_seconds is None:
                    retry_after_seconds = min(
                        backoff_max_seconds,
                        backoff_base_seconds * (2**attempt) + random.uniform(0.0, 0.5),
                    )
                time.sleep(max(0.0, retry_after_seconds))
                attempt += 1
                continue

            parsed_error: dict[str, Any]
            try:
                loaded = json.loads(raw_error) if raw_error.strip() else {}
                parsed_error = loaded if isinstance(loaded, dict) else {"_raw": raw_error}
            except json.JSONDecodeError:
                parsed_error = {"_raw": raw_error}
            return exc.code, parsed_error, f"HTTPError: {exc.code}", attempt, elapsed_ms
        except (error.URLError, TimeoutError) as exc:
            if attempt < max_retries:
                sleep_for = min(
                    backoff_max_seconds,
                    backoff_base_seconds * (2**attempt) + random.uniform(0.0, 0.5),
                )
                time.sleep(max(0.0, sleep_for))
                attempt += 1
                continue
            return 0, {}, f"{type(exc).__name__}: {exc}", attempt, 0.0


def make_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--endpoint",
        default="https://agentig.devsharsha.live/beforest/reply",
        help="Live /beforest/reply endpoint.",
    )
    parser.add_argument(
        "--queries-md",
        type=Path,
        default=Path("docs/evals/Beforest_DM_Evaluation_Plan.md"),
        help="Markdown file containing the 100 numbered queries.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Env file path used to load AUTH_SECRET.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(f"docs/evals/beforest_dm_100_live_{datetime.now(UTC).date().isoformat()}.csv"),
        help="Output CSV path.",
    )
    parser.add_argument("--auth-env-key", default="AUTH_SECRET", help="Env var name for bearer secret.")
    parser.add_argument(
        "--min-interval-seconds",
        type=float,
        default=1.8,
        help="Minimum spacing between outbound requests to avoid rate limiting.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=45.0,
        help="Per-request timeout.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Retries for 429/5xx/network errors.",
    )
    parser.add_argument(
        "--backoff-base-seconds",
        type=float,
        default=2.0,
        help="Base duration for exponential backoff.",
    )
    parser.add_argument(
        "--backoff-max-seconds",
        type=float,
        default=20.0,
        help="Upper bound for backoff sleep.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=100,
        help="Expected number of parsed queries.",
    )
    return parser


def main() -> int:
    args = make_arg_parser().parse_args()

    load_env_file(args.env_file)
    auth_secret = os.getenv(args.auth_env_key, "").strip()
    if not auth_secret:
        print(
            f"Missing bearer token in environment variable {args.auth_env_key!r}.",
            file=sys.stderr,
        )
        return 2

    queries = parse_queries(args.queries_md)
    if len(queries) != args.expected_count:
        print(
            f"Expected {args.expected_count} queries but parsed {len(queries)} from {args.queries_md}.",
            file=sys.stderr,
        )
        return 2

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    run_id = f"eval-{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}"
    state = {"next_allowed_at": 0.0}

    fieldnames = [
        "run_id",
        "evaluated_at_utc",
        "id",
        "category",
        "query",
        "response",
        "http_status",
        "retries",
        "latency_ms",
        "reply_chars",
        "link_count",
        "thread_id",
        "error",
        "grounded_pass",
        "routing_pass",
        "hallucination_pass",
        "dm_style_pass",
        "actionability_pass",
        "tone_pass",
        "link_discipline_pass",
        "notes",
    ]

    status_buckets: dict[int, int] = {}
    started = time.perf_counter()
    with args.output_csv.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for index, query in enumerate(queries, start=1):
            payload = build_request_payload(query, run_id=run_id)
            status, parsed, error_text, retries, latency_ms = call_beforest_reply(
                endpoint=args.endpoint,
                auth_secret=auth_secret,
                payload=payload,
                timeout_seconds=args.timeout_seconds,
                min_interval_seconds=args.min_interval_seconds,
                max_retries=args.max_retries,
                backoff_base_seconds=args.backoff_base_seconds,
                backoff_max_seconds=args.backoff_max_seconds,
                state=state,
            )

            reply_text = str(parsed.get("reply", "") or "")
            thread_id = str(parsed.get("thread_id", payload["thread_id"]))
            status_buckets[status] = status_buckets.get(status, 0) + 1

            writer.writerow(
                {
                    "run_id": run_id,
                    "evaluated_at_utc": datetime.now(UTC).isoformat(),
                    "id": query.query_id,
                    "category": query.category,
                    "query": query.text,
                    "response": reply_text,
                    "http_status": status,
                    "retries": retries,
                    "latency_ms": f"{latency_ms:.1f}",
                    "reply_chars": len(reply_text),
                    "link_count": len(LINK_RE.findall(reply_text)),
                    "thread_id": thread_id,
                    "error": error_text,
                    "grounded_pass": "",
                    "routing_pass": "",
                    "hallucination_pass": "",
                    "dm_style_pass": "",
                    "actionability_pass": "",
                    "tone_pass": "",
                    "link_discipline_pass": "",
                    "notes": "",
                }
            )
            csv_file.flush()

            print(
                f"[{index:03d}/{len(queries)}] id={query.query_id} status={status} "
                f"retries={retries} latency_ms={latency_ms:.1f}",
                flush=True,
            )

    elapsed_seconds = time.perf_counter() - started
    status_summary = ", ".join(f"{code}:{count}" for code, count in sorted(status_buckets.items()))
    print(f"Completed run_id={run_id} in {elapsed_seconds:.1f}s")
    print(f"Status summary: {status_summary}")
    print(f"CSV written to: {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
