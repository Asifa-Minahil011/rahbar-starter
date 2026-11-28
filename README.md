# RAHBAR: Grounded RAG on Kubernetes

[![ci](https://github.com/Asifa-Minahil011/rahbar-starter/actions/workflows/ci.yml/badge.svg)](https://github.com/Asifa-Minahil011/rahbar-starter/actions/workflows/ci.yml) [![cd](https://github.com/Asifa-Minahil011/rahbar-starter/actions/workflows/cd.yml/badge.svg)](https://github.com/Asifa-Minahil011/rahbar-starter/actions/workflows/cd.yml)

A question-answering service that answers only from a document corpus, cites its sources,
and abstains when the corpus does not cover the question. It runs on Kubernetes (minikube).

## Prerequisites
Docker, minikube, kubectl and make. On Windows, use WSL 2 (Ubuntu) with Docker Desktop's
WSL integration turned on.

## Run it from a clean clone
```bash
git clone <this repository> rahbar && cd rahbar

make cluster          # minikube with Calico (NetworkPolicy is enforced), metrics-server, ingress
make images           # build the api, ui and indexer images and load them into minikube
make apply            # namespace, config, chroma, redis, indexer Job, api, ui, ingress, HPA, policies
kubectl get pods -n rahbar -w     # wait: chroma, redis, 3 x api, ui Running; indexer Completed
```

Reach the Ingress. On Linux, `make ask` finds the address itself. On macOS or Windows (docker
driver), run this in a second terminal and leave it open:
```bash
minikube tunnel
```
Then ask a question:
```bash
make ask INGRESS_IP=127.0.0.1 Q="What is the attendance requirement?"
```

## Re-index without a redeploy
```bash
make index            # kubectl create job --from=cronjob/reindex ...
kubectl get jobs -n rahbar -w
```

## Secrets
Nothing secret is committed. If you use a hosted LLM (`LLM_PROVIDER=hosted`), create the Secret
from the command line, never from a file:
```bash
export LLM_API_KEY=...
make secret
```
`k8s/base/secret.placeholder.yaml` only documents the shape and is deliberately not applied.

## Corpus
The documents are not committed. See `docs/CORPUS.md` and `corpus/MANIFEST.md` (URLs and
SHA-256 checksums) and fetch them with `make fetch`.

## Documentation
Architecture decisions are in `docs/adr/`, the pipeline in `docs/PIPELINE.md`, failure analysis in
`docs/FAILURE-ANALYSIS.md`, engineering notes in `docs/ENGINEERING-NOTES.md`, and AI assistance
in `docs/AI-USAGE.md`.

<!-- rahbar-readme-extra -->
## Architecture

Two paths that never share a process: the indexer Job writes the vector store, the API only reads it.

```mermaid
flowchart LR
  U[User] --> I[Ingress]
  I -->|/| UI[ui: nginx]
  I -->|/api| API[api x3: FastAPI]
  API -->|cache check| R[(redis: semantic cache)]
  API -->|reads only| C[(chroma: StatefulSet + PVC)]
  API -->|generate| L[LLM provider]
  IDX[indexer Job / CronJob] -->|the only writer| C
```

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /healthz | liveness: the process is alive, touches no dependency |
| GET | /readyz | readiness: Chroma and Redis reachable, collection non-empty, embedding model matches |
| GET | /metrics | Prometheus metrics |
| POST | /api/ask | body `{"question": "..."}`; returns a cited, schema-validated answer, or abstains |

## Apply order

From a clean clone, in this order (`make cluster`, `make images` and `make apply` run these steps):

```bash
minikube start --driver=docker --cni=calico --memory=6144 --cpus=4   # Calico enforces NetworkPolicy
minikube addons enable metrics-server && minikube addons enable ingress
make images                                  # build the three images and load them into minikube
kubectl apply -k k8s/overlays/dev            # namespace, config, chroma, redis, indexer Job, api, ui, ingress, HPA, policies
kubectl get pods -n rahbar -w                # indexer Completed, everything else Running
minikube tunnel                              # second terminal on macOS/Windows
make ask INGRESS_IP=127.0.0.1 Q="..."        # ask through the Ingress
```

<!-- rahbar-deploy-modes -->
## What runs where

- **The committed manifests deploy the stub retriever and the dummy index** (`RETRIEVER=stub`,
  `EMBED_MODEL=dummy-768`, and the indexer runs with `--dummy`). That keeps CI and a clean-clone
  run independent of any model. In the cluster, `POST /api/ask` therefore returns three fixed
  stub passages, not grounded answers.
- **The real pipeline** (extraction with OCR, structure-aware chunking, `nomic-embed-text`
  embeddings, hybrid retrieval, grounded generation, the semantic cache) was run and measured
  on a developer machine with Ollama. The numbers are in `eval/results.json` and
  `eval/baseline.json`, and the tuning and failure experiments are in `docs/PIPELINE.md`,
  `docs/FAILURE-ANALYSIS.md` and `docs/evidence/measurement-run-summary.md`.
- **CI** runs lint, tests, manifest validation, an image scan and the eval gate. The gate applies
  `eval/run_eval.py`'s `gate()` to the committed results against the baseline. The live run,
  `make eval`, needs a running stack with Ollama.
- **Not done:** running the real pipeline inside the cluster. It needs the corpus mounted into
  the indexer Job and Ollama reachable from the pods.
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
