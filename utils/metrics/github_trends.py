#!/usr/bin/env python3
"""Generate GitHub repository trend charts for README embedding.

This script fetches the list of stargazers and forks from the GitHub API,
aggregates the timestamps to daily counts, and saves a simple SVG/PNG plot
containing both curves.  It is designed to be executed from a scheduled
GitHub Action so that the chart in ``docs/assets/github-trends.svg`` keeps
updating without manual work.

Example::

    python utils/metrics/github_trends.py \
        --owner zimingttkx --repo AI-Practices \
        --output docs/assets/github-trends.svg \
        --data-output docs/assets/github-trends.json

`GITHUB_TOKEN` is optional but strongly recommended to avoid rate limits.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import requests


API_BASE = "https://api.github.com"
STAR_ACCEPT = "application/vnd.github.v3.star+json"


class GitHubAPIError(RuntimeError):
    """Raised when the GitHub API replies with an error payload."""


def _iso_to_date(timestamp: str) -> dt.date:
    """Convert an ISO-8601 timestamp from GitHub into a :class:`datetime.date`."""

    if timestamp.endswith("Z"):
        timestamp = timestamp[:-1] + "+00:00"
    return dt.datetime.fromisoformat(timestamp).date()


def _fetch_paginated(
    session: requests.Session,
    url: str,
    *,
    accept: str | None = None,
) -> List[Dict]:
    """Fetch every page for a GitHub REST collection endpoint."""

    page = 1
    payload: List[Dict] = []
    headers = {"Accept": accept} if accept else None

    while True:
        response = session.get(
            url,
            params={"per_page": 100, "page": page},
            headers=headers,
            timeout=30,
        )
        if response.status_code == 403:
            raise GitHubAPIError(
                "GitHub API rate limit hit. Provide a token via GITHUB_TOKEN."
            )
        if response.status_code >= 400:
            raise GitHubAPIError(f"GitHub API request failed: {response.text}")

        data = response.json()
        if not data:
            break

        payload.extend(data)
        if "next" not in response.links:
            break
        page += 1

    return payload


def _build_series(items: Iterable[Dict], *, timestamp_key: str) -> List[Tuple[dt.date, int]]:
    """Convert event timestamps into a cumulative daily series."""

    buckets: Counter[dt.date] = Counter()
    for entry in items:
        stamp = entry.get(timestamp_key)
        if not stamp:
            continue
        buckets[_iso_to_date(stamp)] += 1

    if not buckets:
        return []

    total = 0
    series: List[Tuple[dt.date, int]] = []
    for day in sorted(buckets):
        total += buckets[day]
        series.append((day, total))
    return series


def _extend_to(series: List[Tuple[dt.date, int]], *, target: dt.date, value: int) -> None:
    """Ensure the series reaches ``target`` with ``value``."""

    if not series:
        series.append((target, value))
        return

    last_day, last_value = series[-1]
    if target < last_day:
        return
    if value != last_value or target != last_day:
        series.append((target, value))


def _plot_series(
    stars: Sequence[Tuple[dt.date, int]],
    forks: Sequence[Tuple[dt.date, int]],
    *,
    title: str,
    output: Path,
) -> None:
    """Render both time series into ``output``."""

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    if stars:
        ax.plot([p[0] for p in stars], [p[1] for p in stars], label="Stars", color="#f6c343")
    if forks:
        ax.plot([p[0] for p in forks], [p[1] for p in forks], label="Forks", color="#7c3aed")

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def _dump_json(
    *,
    output: Path | None,
    stars: Sequence[Tuple[dt.date, int]],
    forks: Sequence[Tuple[dt.date, int]],
) -> None:
    if not output:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "stars": [(day.isoformat(), value) for day, value in stars],
        "forks": [(day.isoformat(), value) for day, value in forks],
    }
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True, help="GitHub owner (user or org)")
    parser.add_argument("--repo", required=True, help="Repository name")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path where the SVG/PNG chart will be written",
    )
    parser.add_argument(
        "--data-output",
        type=Path,
        help="Optional path to dump the aggregated data as JSON",
    )
    args = parser.parse_args(argv)

    session = requests.Session()
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})

    repo_url = f"{API_BASE}/repos/{args.owner}/{args.repo}"
    repo_resp = session.get(repo_url, timeout=30)
    if repo_resp.status_code >= 400:
        raise GitHubAPIError(f"Failed to read repo metadata: {repo_resp.text}")
    repo_data = repo_resp.json()

    stars = _fetch_paginated(
        session, f"{repo_url}/stargazers", accept=STAR_ACCEPT
    )
    forks = _fetch_paginated(session, f"{repo_url}/forks")

    star_series = _build_series(stars, timestamp_key="starred_at")
    fork_series = _build_series(forks, timestamp_key="created_at")

    today = dt.date.today()
    _extend_to(star_series, target=today, value=repo_data.get("stargazers_count", 0))
    _extend_to(fork_series, target=today, value=repo_data.get("forks_count", 0))

    title = f"{args.owner}/{args.repo} · Stars & Forks"
    _plot_series(star_series, fork_series, title=title, output=args.output)
    _dump_json(output=args.data_output, stars=star_series, forks=fork_series)

    print(
        f"Generated chart with {len(star_series)} star points and {len(fork_series)} fork points"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
