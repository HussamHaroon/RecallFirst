"""SWE-bench Lite data utilities: gold-file extraction, repo snapshots, null issues."""
from __future__ import annotations

import io
import os
import re
import tarfile
import urllib.request

DIFF_FILE_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)


def gold_files(patch: str) -> list[str]:
    """Files modified by the gold patch (b-side paths)."""
    return sorted({m.group(2) for m in DIFF_FILE_RE.finditer(patch or "")})


def fetch_repo_snapshot(repo: str, commit: str, dest: str) -> str:
    """Download the repo tarball at `commit` from GitHub codeload and extract.

    dest gets a single subdir named after the repo. Returns the repo dir path.
    Skips tests/ and docs/ trees at extraction time to keep repos lean.
    """
    owner, name = repo.split("/")
    url = f"https://codeload.github.com/{owner}/{name}/tar.gz/{commit}"
    root = os.path.join(dest, f"{name}-{commit}")
    if os.path.isdir(root):
        return root
    os.makedirs(dest, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        members = [m for m in tf.getmembers()
                   if "/tests/" not in m.name and "/docs/" not in m.name
                   and "/.github/" not in m.name]
        tf.extractall(dest, members=members)
    # codeload top dir is usually {repo}-{commit} (or {repo}-{tag}); normalize
    extracted = [d for d in os.listdir(dest)
                 if os.path.isdir(os.path.join(dest, d)) and d.startswith(name)]
    if extracted and not os.path.isdir(root):
        os.rename(os.path.join(dest, extracted[0]), root)
    return root


def pick_null_issue(dataset_rows: list[dict], issue_text: str, exclude_repo: str,
                    token_counter=len) -> dict:
    """Length-matched null control: another instance's issue from a DIFFERENT
    repo whose token count is within +-30% of the real issue."""
    target = token_counter(issue_text)
    lo, hi = 0.7 * target, 1.3 * target
    best, best_gap = None, float("inf")
    for row in dataset_rows:
        if row["repo"] == exclude_repo:
            continue
        n = token_counter(row["problem_statement"])
        if lo <= n <= hi:
            gap = abs(n - target)
            if gap < best_gap:
                best, best_gap = row, gap
    if best is None:  # relax: nearest length regardless of band
        for row in dataset_rows:
            if row["repo"] == exclude_repo:
                continue
            gap = abs(token_counter(row["problem_statement"]) - target)
            if gap < best_gap:
                best, best_gap = row, gap
    return best
