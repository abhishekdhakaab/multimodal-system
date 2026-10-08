# Build context must be the project root (fenris/), not k8s/, since this
# needs files from model/ and data_pipeline/ too:
#   docker build -f k8s/Dockerfile -t fenris-inference:latest .

FROM python:3.11-slim

WORKDIR /app

COPY k8s/requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY model/__init__.py model/vision_encoder.py model/lidar_encoder.py model/fusion.py model/full_model.py model/
COPY data_pipeline/__init__.py data_pipeline/modelnet_loader.py data_pipeline/hard_case_miner.py data_pipeline/
COPY k8s/serve.py serve.py

COPY model/checkpoints/model.pkl model/checkpoints/model.pkl
COPY model/checkpoints/model_bad.pkl model/checkpoints/model_bad.pkl
COPY data_pipeline/shards/val/shard_000.npz data_pipeline/shards/val/shard_000.npz

ENV MODEL_PATH=/app/model/checkpoints/model.pkl
ENV VAL_SHARD_PATH=/app/data_pipeline/shards/val/shard_000.npz

EXPOSE 8080
CMD ["python", "serve.py"]
