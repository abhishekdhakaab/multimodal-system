#!/usr/bin/env bash
# Runs the full Phase 6 canary rollout/rollback demo end to end against the
# live k3d cluster. Assumes scripts/setup_k3s.sh and the manifests in
# k8s/manifests/ have already been applied (the cluster + deployments exist).
#
# Sequence: reset both deployments to v1, roll out v2 (same good model,
# should promote), then roll out v3-bad (deliberately untrained checkpoint,
# should roll back). Prints the final deployment state so you can see
# stable never actually ran the bad model.

set -e
cd "$(dirname "$0")/.."

echo "=== resetting stable + canary to v1 ==="
kubectl patch deployment fenris-inference-stable --type=strategic -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"inference","env":[{"name":"MODEL_PATH","value":"/app/model/checkpoints/model.pkl"},{"name":"MODEL_VERSION","value":"v1"},{"name":"NODE_TYPE","value":"arm"}]}]}}}}'
kubectl patch deployment fenris-inference-canary --type=strategic -p \
  '{"spec":{"template":{"spec":{"containers":[{"name":"inference","env":[{"name":"MODEL_PATH","value":"/app/model/checkpoints/model.pkl"},{"name":"MODEL_VERSION","value":"v1"},{"name":"NODE_TYPE","value":"arm"}]}]}}}}'
kubectl rollout status deployment/fenris-inference-stable --timeout=60s
kubectl rollout status deployment/fenris-inference-canary --timeout=60s

echo
echo "=== rollout v2 (same good model -- should PROMOTE) ==="
python3 -m k8s.canary_controller rollout --model-path /app/model/checkpoints/model.pkl --version v2

echo
echo "=== rollout v3-bad (untrained checkpoint -- should ROLLBACK) ==="
python3 -m k8s.canary_controller rollout --model-path /app/model/checkpoints/model_bad.pkl --version v3-bad

echo
echo "=== final state (stable should still be v2, never touched by v3-bad) ==="
echo -n "stable:  "; kubectl get deployment fenris-inference-stable -o jsonpath='{.spec.template.spec.containers[0].env}'; echo
echo -n "canary:  "; kubectl get deployment fenris-inference-canary -o jsonpath='{.spec.template.spec.containers[0].env}'; echo
