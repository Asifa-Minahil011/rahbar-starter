#!/usr/bin/env bash
# Run an indexer "manage" command on the cluster (archive / restore / rollback / counts / unpin).
# The vector store has one writer: an indexer pod (the NetworkPolicy only lets app=indexer reach
# Chroma), so this starts a one-off indexer pod instead of port-forwarding from the laptop.
#
#   scripts/manage.sh counts
#   scripts/manage.sh counts --doc-id academic-rules
#   scripts/manage.sh rollback academic-rules --to 2023-08
#   scripts/manage.sh archive student-handbook-jun-2026.pdf
#   scripts/manage.sh restore student-handbook-jun-2026.pdf
#   scripts/manage.sh unpin student-handbook-jun-2026.pdf
set -euo pipefail
NS="${NS:-rahbar}"
IMAGE="${IMAGE:-rahbar-indexer:dev}"
NAME="manage-$(date +%s)"
[ "$#" -ge 1 ] || { sed -n '2,12p' "$0"; exit 2; }

OVERRIDES=$(cat <<JSON
{"spec":{"securityContext":{"runAsNonRoot":true,"runAsUser":10001,"runAsGroup":10001},
 "containers":[{"name":"manage","image":"${IMAGE}","imagePullPolicy":"IfNotPresent",
 "envFrom":[{"configMapRef":{"name":"rahbar-config"}}],
 "command":["python","-m","pipeline.manage"$(printf ',"%s"' "$@")]}]}}
JSON
)
kubectl run "$NAME" -n "$NS" --image="$IMAGE" --restart=Never --labels=app=indexer \
  --overrides="$OVERRIDES" >/dev/null
trap 'kubectl delete pod "$NAME" -n "$NS" --ignore-not-found >/dev/null 2>&1' EXIT
kubectl wait --for=jsonpath='{.status.phase}'=Succeeded pod/"$NAME" -n "$NS" --timeout=120s \
  >/dev/null 2>&1 || true
kubectl logs "$NAME" -n "$NS"
phase=$(kubectl get pod "$NAME" -n "$NS" -o jsonpath='{.status.phase}')
[ "$phase" = "Succeeded" ]
