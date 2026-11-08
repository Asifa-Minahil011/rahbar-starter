#!/usr/bin/env python3
"""After scripts/finish_all.py: fill the docs, commit in logical pieces, and push when the network is back.

    nohup .venv/bin/python scripts/after_finish.py > /tmp/after.log 2>&1 &

Works offline: commits are local; only the push needs the network, and it is retried every 3 minutes
for up to 12 hours. It will NOT push if the tests fail or a merge-conflict marker is present.
It never force-pushes. Everything it writes into docs/ is a draft built from your numbers: review it later.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def git(repo: Path, *args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=check)


def wait_for_finish(path: Path, hours: float = 8) -> str:
    deadline = time.time() + hours * 3600
    while time.time() < deadline:
        text = path.read_text(errors="replace") if path.exists() else ""
        if "ALL DONE" in text:
            return "done"
        if "STOPPED by an error" in text:
            return "stopped"
        time.sleep(60)
    return "timeout"


def commit_group(repo: Path, message: str, paths: list[str]) -> None:
    existing = [p for p in paths if (repo / p).exists()]
    if not existing:
        return
    git(repo, "add", "--", *existing)
    if git(repo, "diff", "--cached", "--quiet").returncode == 0:
        return
    r = git(repo, "commit", "-m", message)
    log(f"commit: {message}" if r.returncode == 0 else f"commit failed: {r.stderr.strip()[:200]}")


def has_conflict_markers(repo: Path) -> bool:
    r = git(repo, "grep", "-nE", "^(<<<<<<< |>>>>>>> )", "--", "api", "indexer", "eval", "scripts", "docs", "k8s")
    return r.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--finish-log", type=Path, default=Path("/tmp/finish.log"))
    ap.add_argument("--summary", type=Path, default=Path("/tmp/finish/SUMMARY.md"))
    ap.add_argument("--floor", type=Path, default=Path("/tmp/finish/floor.txt"))
    ap.add_argument("--remote", default="origin")
    ap.add_argument("--branch", default="dev")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--no-tests", action="store_true")
    args = ap.parse_args()
    repo = args.repo

    if not args.no_wait:
        log(f"waiting for {args.finish_log} to say ALL DONE ...")
        state = wait_for_finish(args.finish_log)
        log(f"measuring finished: {state} (continuing in every case: partial results are still worth committing)")

    py = repo / ".venv" / "bin" / "python"
    py_cmd = str(py) if py.exists() else sys.executable
    fill = subprocess.run([py_cmd, "scripts/fill_results.py"], cwd=repo, text=True, capture_output=True)
    log("fill_results:\n" + fill.stdout.strip())

    evidence = repo / "docs" / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    if args.summary.exists():
        shutil.copy(args.summary, evidence / "measurement-run-summary.md")
    values = []
    if args.floor.exists():
        values.append(f"RELEVANCE_FLOOR={args.floor.read_text().strip()}")
    cache_value = repo / "eval" / "cache-threshold.txt"
    if cache_value.exists() and cache_value.read_text().strip():
        values.append(f"CACHE_THRESHOLD={cache_value.read_text().strip()}")
    if values:
        (repo / "eval" / "config-values.txt").write_text("\n".join(values) + "\n")
        log("values for the ConfigMap: " + ", ".join(values))

    commit_group(repo, "eval: chunk tuning results (three configurations)", ["eval/tuning.json"])
    commit_group(repo, "eval: relevance floor, full eval and baseline", [
        "eval/results.json", "eval/baseline.json", "eval/config-values.txt", "eval/cache-threshold.txt",
        "docs/evidence/measurement-run-summary.md"])
    commit_group(repo, "eval: failure experiment measurements (B, C, E, F)",
                 ["eval/failure-CE.json", "eval/failure-F.json", "eval/tuning-b.json"])
    commit_group(repo, "docs: measured results in PIPELINE, FAILURE-ANALYSIS and engineering notes Q5",
                 ["docs/PIPELINE.md", "docs/FAILURE-ANALYSIS.md", "docs/ENGINEERING-NOTES.md"])

    if has_conflict_markers(repo):
        log("NOT pushing: a merge-conflict marker is in the tree. Fix it by hand.")
        return 1
    if not args.no_tests:
        make = subprocess.run(["make", "test"], cwd=repo, text=True, capture_output=True)
        if make.returncode != 0:
            log("NOT pushing: tests fail.\n" + (make.stdout + make.stderr)[-800:])
            return 1
        log("tests pass")

    for attempt in range(1, 241):          # 12 hours at 3-minute intervals
        pull = git(repo, "pull", "--no-rebase", "--no-edit", args.remote, args.branch)
        if pull.returncode != 0 and "CONFLICT" in (pull.stdout + pull.stderr):
            git(repo, "merge", "--abort")
            log("NOT pushing: merging the remote branch conflicts. Commits are safe locally; resolve by hand.")
            return 1
        push = git(repo, "push", args.remote, args.branch)
        if push.returncode == 0:
            log(f"pushed to {args.remote}/{args.branch}")
            return 0
        log(f"push attempt {attempt} failed (offline?): {push.stderr.strip().splitlines()[-1][:160] if push.stderr.strip() else ''}")
        time.sleep(180)
    log("gave up pushing after 12 hours; commits are safe locally")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
