## RAGOps (Assignment 3): one command per part

Run from the repo root with the venv (`make install` creates `.venv`). The cluster from the
Assignment 2 steps above must be up (`make cluster images apply`). The suite needs the API on
`localhost:8010`:

```bash
kubectl port-forward -n rahbar svc/api 8010:8000      # leave this terminal open
```

| Part | What | Command |
| --- | --- | --- |
| A1 | Verify documents before indexing (report + quarantine) | `cd indexer && python -m pipeline.verify --corpus ../corpus/raw --out /tmp/data` (add `--no-embed` if Ollama is not reachable). The indexer Job runs it automatically; report in `$DATA_DIR/reports`, rejected files in `$DATA_DIR/quarantine`. |
| A2 | Incremental ingestion, 0 chunks on an unchanged corpus | `make index` then `kubectl logs job/<name> -n rahbar`. Manage versions: `scripts/manage.sh` (see `docs/A2-incremental-store.md`). |
| B | Three-level suite (module, component, end to end) | `.venv/bin/python eval/ragops_suite.py --api http://localhost:8010 --tag baseline` (all levels, about 1 hour on CPU). Fast, no LLM: `make suite-fast TAG=baseline`. Subset: add `--ids q01,q02`. Stochasticity: `--repeat 2 --seed 42`. LLM judge: `--judge`. |
| B (golden set) | Check the golden set against the assignment rules | `make golden-check` |
| C | Release gate: candidate vs baseline | `.venv/bin/python eval/gate.py --candidate eval/results/suite-<stamp>-<tag>.json --baseline eval/results/baseline.json --config eval/gate.yaml` (exit 0 = release, 1 = blocked). Thresholds are in `eval/gate.yaml` only. |
| D1, D2 | Trace record and prompt rebuilt from a trace id | `kubectl logs deploy/api -n rahbar \| python3 scripts/trace_show.py <trace_id> --file -` or `python3 scripts/trace_show.py <trace_id> --file <log file>` |
| D3 | Monitoring numbers (outcome mix, abstention rate, similarity, latency) | `python3 scripts/trace_metrics.py --file <log file>` |
| E | Query coverage before/after new golden questions | `python3 scripts/coverage.py --help`; calibration: `--calibrate`; results in `eval/results/coverage-before.json`, `coverage-after.json` |
| E3 | Data drift after Batch B | run the suite before and after, then `python3 scripts/drift_report.py before.json after.json --new-sources <batch B files>` |
| F | Evaluation Job on Kubernetes | `scripts/eval_job.sh` is the Job's entry point (environment variables are listed at its top); the gate result is the Job's exit status. Edit the `gate-config` ConfigMap and run it again to see the outcome change. |

Result files, traces and reports are committed under `eval/results/` and `traces/`.
