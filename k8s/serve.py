"""
Minimal inference service for the Kubernetes fleet demo. Stdlib HTTP server
only (no Flask/FastAPI dependency) -- this just needs to serve two small
endpoints inside a container, not handle real production traffic.

On startup, loads a checkpoint and computes REAL accuracy on a small
held-out validation shard baked into the image -- not a hardcoded number.
This is what the canary controller polls to decide whether a rollout is
healthy.

Env vars:
    MODEL_PATH    path to the .pkl checkpoint to serve (default: the real trained model)
    MODEL_VERSION free-text label returned in /accuracy, for logging/display only
    NODE_TYPE     free-text label returned in /accuracy, for logging/display only
"""

import json
import os
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer

import jax.numpy as jnp
import numpy as np

from model.full_model import forward

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/checkpoints/model.pkl")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")
NODE_TYPE = os.environ.get("NODE_TYPE", "unknown")
VAL_SHARD_PATH = os.environ.get("VAL_SHARD_PATH", "/app/data_pipeline/shards/val/shard_000.npz")
PORT = int(os.environ.get("PORT", "8080"))


def compute_accuracy():
    with open(MODEL_PATH, "rb") as f:
        params = pickle.load(f)

    data = np.load(VAL_SHARD_PATH, allow_pickle=True)
    images = jnp.array(data["images"])
    points = jnp.array(data["pointclouds"])
    labels = jnp.array(data["labels"])

    logits = forward(params, images, points, backend="jax")
    preds = jnp.argmax(logits, axis=-1)
    acc = float(jnp.mean(preds == labels))
    return acc


print(f"[serve.py] loading {MODEL_PATH}, computing real accuracy on {VAL_SHARD_PATH}...")
ACCURACY = compute_accuracy()
print(f"[serve.py] model_version={MODEL_VERSION} node_type={NODE_TYPE} measured_accuracy={ACCURACY:.4f}")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        elif self.path == "/accuracy":
            self._send_json({
                "accuracy": ACCURACY,
                "model_version": MODEL_VERSION,
                "node_type": NODE_TYPE,
            })
        else:
            self._send_json({"error": "not found"}, status=404)

    def log_message(self, format, *args):
        pass  # keep container logs quiet; /health gets polled frequently


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[serve.py] listening on :{PORT}")
    server.serve_forever()
