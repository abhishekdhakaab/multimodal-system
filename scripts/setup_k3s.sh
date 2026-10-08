#!/usr/bin/env bash
# Creates a local k3d (K3s-in-Docker) cluster with 3 nodes for Phase 6's
# fleet control plane demo: 1 server + 2 agents.
#
# Node labeling, stated honestly:
#   - agent-0 is labeled node-type=arm. This is NOT a simulation -- it's a
#     real container running on the M1's actual ARM64 silicon, same as
#     edge/arm_path/infer.py. Docker on Apple Silicon runs native ARM64
#     containers, so this node genuinely is ARM hardware.
#   - agent-1 is labeled node-type=gpu-simulated, gpu=false. There is no
#     real GPU attached. This stands in for "the fleet's GPU-accelerated
#     edge box" so the canary/rollout logic has two distinct node types to
#     schedule across -- it is explicitly named "simulated" in the label
#     itself so nobody mistakes it for a real GPU node later. Once Phase 3/4
#     has real Colab/rented-GPU results, this can be replaced with a real
#     remote node if desired, but isn't required for the fleet-control-plane
#     logic itself, which only needs node *labels* to schedule against.

set -e

CLUSTER_NAME="fenris"

if k3d cluster list | grep -q "^${CLUSTER_NAME} "; then
    echo "cluster '${CLUSTER_NAME}' already exists, deleting it first for a clean setup"
    k3d cluster delete "${CLUSTER_NAME}"
fi

k3d cluster create "${CLUSTER_NAME}" --agents 2

kubectl label node "k3d-${CLUSTER_NAME}-agent-0" node-type=arm --overwrite
kubectl label node "k3d-${CLUSTER_NAME}-agent-1" node-type=gpu-simulated gpu=false --overwrite

echo
echo "cluster ready. nodes:"
kubectl get nodes --show-labels
