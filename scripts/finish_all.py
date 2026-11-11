#!/usr/bin/env python3
"""Run every remaining measurement step, unattended, and log each one.

    nohup systemd-inhibit --what=sleep:idle .venv/bin/python scripts/finish_all.py > /tmp/finish.log 2>&1 &

Needs Chroma (port 8001) and Ollama running, and the laptop plugged in and awake. Steps, in order:
  wait   wait for scripts/tune_chunking.py (log: /tmp/tune.log) to finish
  floor  start the API, measure the best similarity of every golden question, pick RELEVANCE_FLOOR
  eval   full eval with that floor, writes eval/results.json and eval/baseline.json
  cache  semantic-cache threshold table
  ce     failure experiments C (TOP_K=1) and E (superseded filter off), retrieval-only
  b      failure experiment B: 100-word chunks, no overlap (scripts/tune_chunking.py --configs 100:0)
  f      failure experiment F: temperature 1.2, full eval (slowest)
A failing step is logged and the rest still run. Output: /tmp/finish/*.log and /tmp/finish/SUMMARY.md.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/tmp/finish")
STEPS = ["wait", "floor", "eval", "cache", "ce", "b", "f"]
PORT = 8010


def note(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(OUT / "SUMMARY.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], log: str, env: dict[str, str] | None = None) -> int:
    with open(OUT / f"{log}.log", "w", encoding="utf-8") as f:
        code = subprocess.run(cmd, cwd=ROOT, env={**os.environ, **(env or {})},
                              stdout=f, stderr=subprocess.STDOUT).returncode
    return code


def tail(log: str, n: int = 14) -> str:
    return "\n".join((OUT / f"{log}.log").read_text(errors="replace").splitlines()[-n:])


def load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = module
    spec.loader.exec_module(module)                 # type: ignore[union-attr]
    return module


def start_api(env_over: dict[str, str]) -> subprocess.Popen[bytes]:
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(PORT)],
                            cwd=ROOT / "api", env={**os.environ, **env_over},
                            stdout=open(OUT / "api.log", "a"), stderr=subprocess.STDOUT)
    for _ in range(90):
        if proc.poll() is not None:
            raise RuntimeError("API exited during start-up: " + tail("api"))
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/healthz", timeout=2).status_code == 200:
                return proc
        except httpx.HTTPError:
            pass
        time.sleep(1)
    proc.terminate()
    raise RuntimeError("API did not become healthy")


def stop_api(proc: subprocess.Popen[bytes]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        proc.kill()


def step_wait(args: argparse.Namespace) -> None:
    log = Path(args.tune_log)
    note(f"waiting for tuning to finish ({log}) ...")
    deadline = time.time() + 4 * 3600
    while time.time() < deadline:
        text = log.read_text(errors="replace") if log.exists() else ""
        if "written to" in text:
            note("tuning finished:\n" + "\n".join(row for row in text.splitlines() if row.startswith("|")))
            return
        if log.exists() and time.time() - log.stat().st_mtime > 20 * 60:
            note("tuning log has not changed for 20 minutes: assuming it stopped, continuing anyway")
            return
        time.sleep(30)
    note("gave up waiting for tuning after 4 hours, continuing")


def step_floor(args: argparse.Namespace) -> float:
    pf = load("pick_floor")
    questions = yaml.safe_load(Path(args.golden).read_text(encoding="utf-8"))["questions"]
    api = start_api({})
    try:
        scored = pf.best_scores(f"http://127.0.0.1:{PORT}", questions)
    finally:
        stop_api(api)
    answerable = [s for s, q in scored if q.get("relevant") and not q.get("must_abstain")]
    abstain = [s for s, q in scored if q.get("must_abstain")]
    floor, why = pf.suggest(answerable, abstain)
    if floor is None:
        floor = round(max(abstain) + 0.01, 2)
        why += f" Using {floor} (just above the highest abstain score)."
    (OUT / "floor.txt").write_text(f"{floor}\n")
    note(f"RELEVANCE_FLOOR = {floor}. {why}")
    for s, q in sorted(scored, key=lambda t: t[0]):
        note(f"   {s:.3f} {q['id']} {q['type']:<9} {q['question'][:60]}")
    return float(floor)


def step_eval(args: argparse.Namespace, floor: float) -> None:
    api = start_api({"RELEVANCE_FLOOR": str(floor)})
    try:
        code = run([sys.executable, "eval/run_eval.py", "--api", f"http://127.0.0.1:{PORT}",
                    "--golden", args.golden, "--update-baseline"], "eval")
    finally:
        stop_api(api)
    note(f"full eval exit code {code} (1 means a gate failed; the files are still written)\n{tail('eval')}")
    run([sys.executable, "scripts/results_table.py"], "results_table")
    note("recall by question type:\n" + (OUT / "results_table.log").read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--golden", default=str(ROOT / "eval" / "golden.yaml"))
    ap.add_argument("--tune-log", default="/tmp/tune.log")
    ap.add_argument("--skip", default="", help=f"comma-separated steps to skip from {STEPS}")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    OUT.mkdir(exist_ok=True)
    os.environ.setdefault("RETRIEVER", "chroma")
    os.environ.setdefault("LLM_PROVIDER", "ollama-host")
    os.environ.setdefault("CHROMA_HOST", "localhost")
    os.environ.setdefault("CHROMA_PORT", "8001")
    os.environ.setdefault("OLLAMA_URL", "http://localhost:11434")
    os.environ.setdefault("EMBED_MODEL", "nomic-embed-text")
    note("start")
    floor = float(os.environ.get("RELEVANCE_FLOOR", "0.5"))
    failed: list[str] = []

    def guard(name: str, fn: Any) -> Any:
        """One step failing must not stop the others: note it and carry on."""
        try:
            return fn()
        except Exception as exc:
            failed.append(name)
            note(f"STEP '{name}' FAILED: {exc!r} (continuing with the next step)")
            return None

    if "wait" not in skip:
        guard("wait", lambda: step_wait(args))
    if "floor" not in skip:
        picked = guard("floor", lambda: step_floor(args))
        floor = picked if picked is not None else floor
        if picked is None:
            note(f"floor step failed: using RELEVANCE_FLOOR={floor} for the rest")
    os.environ["RELEVANCE_FLOOR"] = str(floor)
    if "eval" not in skip:
        guard("eval", lambda: step_eval(args, floor))
    if "cache" not in skip:
        def cache_step() -> None:
            code = run([sys.executable, "scripts/cache_threshold.py"], "cache")
            note(f"cache threshold (exit {code}):\n{tail('cache', 30)}")
        guard("cache", cache_step)
    if "ce" not in skip:
        def ce_step() -> None:
            code = run([sys.executable, "scripts/failure_experiments.py", "--stages", "C,E",
                        "--golden", args.golden, "--out", "eval/failure-CE.json"], "failure_ce")
            note(f"failure experiments C and E (exit {code}):\n{tail('failure_ce', 30)}")
        guard("ce", ce_step)
    if "b" not in skip:
        def b_step() -> None:
            code = run([sys.executable, "scripts/tune_chunking.py", "--configs", "100:0", "--golden", args.golden,
                        "--out", "eval/tuning-b.json"], "stage_b")
            note(f"stage B, 100-word chunks without overlap (exit {code}):\n{tail('stage_b', 8)}")
        guard("b", b_step)
    if "f" not in skip:
        def f_step() -> None:
            code = run([sys.executable, "scripts/failure_experiments.py", "--stages", "F", "--golden", args.golden,
                        "--baseline-full", "eval/results.json", "--out", "eval/failure-F.json"], "failure_f")
            note(f"failure experiment F (exit {code}):\n{tail('failure_f', 30)}")
        guard("f", f_step)
    if failed:
        note("steps that failed: " + ", ".join(failed))
    note("ALL DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
