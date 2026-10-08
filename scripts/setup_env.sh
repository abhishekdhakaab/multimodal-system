#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt -q

python -c "import jax; print('jax', jax.__version__); print(jax.devices())"
echo "Setup done. Activate with: source venv/bin/activate"
