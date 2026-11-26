#!/bin/sh
# Entry point of the Kubernetes evaluation Job (Part F2): run the Part B suite, then apply
# the release gate. The Job's exit status is the gate's: 0 = pass, 1 = blocked, 2 = could not run.
#
# Environment (set in k8s/eval-job.yaml):
#   API_URL        the API to test, e.g. http://api.rahbar.svc.cluster.local:8000 (or the candidate Service)
#   RESULTS_DIR    a mounted volume for the result files        (default eval/results)
#   TAG            label in the result file name                (default job)
#   GATE_CONFIG    mounted ConfigMap with the thresholds        (default eval/gate.yaml)
#   GATE_BASELINE  the baseline suite result                    (default $RESULTS_DIR/baseline.json)
#   TRACE_FILE     the API's trace file on the shared volume    (for faithfulness)
#   EXTRA_ARGS     anything else for ragops_suite.py, e.g. "--judge --repeat 2"
set -u
RESULTS_DIR="${RESULTS_DIR:-eval/results}"
TAG="${TAG:-job}"
mkdir -p "$RESULTS_DIR"
# shellcheck disable=SC2086
python eval/ragops_suite.py --api "${API_URL:?set API_URL}" --out-dir "$RESULTS_DIR" --tag "$TAG" ${EXTRA_ARGS:-} || exit 2
CAND=$(ls -t "$RESULTS_DIR"/suite-*-"$TAG".json 2>/dev/null | head -1)
python eval/gate.py --candidate "$CAND" --baseline "${GATE_BASELINE:-$RESULTS_DIR/baseline.json}" --config "${GATE_CONFIG:-eval/gate.yaml}"
