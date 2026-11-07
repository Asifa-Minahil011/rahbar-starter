# RAHBAR: Grounded RAG on Kubernetes

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
