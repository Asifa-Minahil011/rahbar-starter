# ADR 0001: The indexer is a Job, and the API only reads

## Status
Accepted.

## Context
RAHBAR runs three `api` replicas against one Chroma volume. The volume belongs to the Chroma
StatefulSet (`k8s/base/chroma.yaml`, `volumeClaimTemplates` at line 82, mounted at `/data` on
line 52); the API pods never mount it and reach Chroma over the network.

Indexing is expensive (extraction, OCR, embedding) and it writes. The obvious place to run it
is API startup, so that every pod builds its own index. We rejected that.

## Decision
Indexing runs in a one-shot Kubernetes Job, `indexer` (`k8s/base/indexer-job.yaml`), with
`restartPolicy: OnFailure` (line 22) and `backoffLimit: 6` (line 14). A CronJob, `reindex`
(`k8s/base/reindex-cronjob.yaml`, weekly), runs the same image with the same arguments. The API
only reads. A manual re-index is one command, with no redeploy:

    kubectl create job --from=cronjob/reindex manual-reindex-1 -n rahbar     (make index)

The API's readiness probe (`k8s/base/api.yaml`, lines 78 to 84, path `/readyz`) answers 503
until the collection exists, is non-empty, and was built with the model this API is configured
for (`api/app/routes/health.py`, `run_checks`, line 84). A pod without an index is therefore
withheld from the Service instead of serving empty retrievals.

## Why not index at startup
1. Three concurrent writers. `indexer/pipeline/store.py` builds a collection named
   `<collection>__building`, deletes any leftover one first (line 49), then deletes the live
   collection (line 77) and renames the new one into place. Three pods running that at the same
   time delete each other's half-built collection and the live one, so the index that survives
   may match no document set.
2. Every restart re-indexes. A rolling update, an OOMKill or a node drain would pay the full
   extraction and embedding cost again, while serving traffic.
3. Readiness would lie, or Kubernetes would kill the pod halfway through indexing.
4. Re-indexing would need a redeploy.

## Consequences
- Re-indexing is independent of the API's lifecycle and can fail, retry and be inspected on its
  own (`kubectl logs job/indexer`).
- The "API only reads" rule is a code convention, not a network rule: the NetworkPolicy
  `chroma-from-api-and-indexer` (`k8s/base/networkpolicy.yaml`, line 83) lets the API reach
  Chroma over HTTP, so nothing but the code stops it writing.
- First deploy lesson: the Job started before Chroma had finished pulling its image and used up
  its three retries, so we raised `backoffLimit` from 3 to 6.
- A new index gets a new `index_id` in the collection metadata, which the semantic cache uses to
  stop serving answers from the old index.

## Alternatives considered
- An init container in every API pod: still three writers.
- In-process leader election: more code and more failure modes than a Job gives for free.
