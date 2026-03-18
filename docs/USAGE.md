# Nexus Node FL Engine — Usage Guide

This guide covers how to use the deployed FL Engine API. You need:

- **Server URL** — the deployed API endpoint
- **Org API Key** — for managing jobs and downloading models
- **Node API Key** — for running compute nodes

> Don't have these? Ask your admin, or [deploy your own instance](#deploy-your-own).

Throughout this guide we use three variables:

### `$SERVER_URL`
The deployed API endpoint. Ask your admin, or find it in Railway → API service → Settings → Networking → Public domain.
```
Example: https://nexus-node-fl-engine-production.up.railway.app
```

### `$ORG_KEY`
Organization API key (starts with `nxo_`). Used for creating jobs and downloading trained models. Ask your admin — it's printed in the deploy logs on first startup. If you're the admin, check Railway → API service → Logs and look for `SEED ORGANIZATION CREATED`.

### `$NODE_KEY`
Compute node API key (starts with `nxn_`). Used for running the node agent (heartbeat, training, submitting weights). You get one by registering a node:
```bash
curl -X POST $SERVER_URL/api/v1/auth/register-node \
  -H "Content-Type: application/json" \
  -d '{"name": "your-name"}'
```
The key is shown only once in the response — save it immediately.

---

## 1. Verify the Server

```bash
# Health check
curl $SERVER_URL/health
# → {"status":"ok"}

# DB connectivity
curl $SERVER_URL/ready
# → {"status":"ready","db":"ok"}
```

Interactive API docs are at `$SERVER_URL/docs`.

---

## 2. Register a Compute Node

```bash
curl -X POST $SERVER_URL/api/v1/auth/register-node \
  -H "Content-Type: application/json" \
  -d '{"name": "my-node", "region": "vietnam"}'
```

Response:
```json
{
  "id": "NODE_UUID",
  "name": "my-node",
  "api_key": "nxn_xxxxxxxxxxxx"
}
```

**Save the `id` and `api_key` — the key is only shown once.**

---

## 3. Create a Training Job

Requires an org API key (`nxo_`).

```bash
curl -X POST $SERVER_URL/api/v1/jobs \
  -H "X-API-Key: $ORG_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "plant-disease-v1",
    "model_type": "plantnet",
    "num_classes": 38,
    "total_rounds": 10,
    "min_nodes": 2,
    "node_ids": ["NODE1_UUID", "NODE2_UUID"]
  }'
```

| Field | Required | Default | Description |
|-------|----------|---------|-------------|
| `name` | yes | — | Job name |
| `model_type` | no | `plantnet` | Model architecture |
| `num_classes` | no | `38` | Output classes |
| `total_rounds` | no | `10` | FL rounds |
| `min_nodes` | no | `2` | Minimum nodes required |
| `node_ids` | no | — | UUIDs of nodes to assign |
| `config` | no | — | Extra config (JSON) |

---

## 4. Run a Compute Node

### Prerequisites

```bash
# Clone the repo
git clone <repo-url>
cd nexus-node-fl-engine

# Install client dependencies
pip install -e ".[client]"

# Download PlantVillage dataset (or your own ImageFolder dataset)
# See: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
```

### Start the node agent

```bash
python -m client.node \
  --server $SERVER_URL \
  --api-key $NODE_KEY \
  --data-dir ./plantvillage \
  --epochs 2 \
  --batch-size 32 \
  --max-samples 200
```

The node will:
1. Send heartbeats every 30s
2. Poll for assigned tasks
3. Download the global model
4. Train locally (no data leaves your machine)
5. Submit trained weights
6. Repeat for next round

### Node agent flags

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `http://localhost:8000` | Server URL |
| `--api-key` | required | Node API key (`nxn_...`) |
| `--data-dir` | required | Dataset path (ImageFolder layout) |
| `--epochs` | `2` | Training epochs per round |
| `--batch-size` | `32` | Training batch size |
| `--lr` | `0.001` | Learning rate |
| `--max-samples` | `200` | Max training samples (0 = all) |
| `--freeze-backbone` | auto on CPU | Freeze backbone for fast CPU training |
| `--poll-interval` | `5` | Seconds between task polls |

---

## 4b. Run a Mobile Node

Mobile nodes (phones/tablets) use a different flow optimized for battery and bandwidth.

### Mobile check-in

The app reports device conditions. Server only assigns work if conditions are met:

```bash
curl -X POST $SERVER_URL/api/v1/mobile/checkin \
  -H "X-API-Key: $NODE_KEY" \
  -H "Content-Type: application/json" \
  -d '{"battery_pct": 85, "is_charging": true, "is_wifi": true}'
```

Server rejects if: battery < 30%, not on WiFi, or not charging.

Response when a task is available:
```json
{
  "has_task": true,
  "job_id": "...",
  "job_name": "plant-disease-v1",
  "round_num": 0,
  "model_type": "plantnet",
  "num_classes": 38
}
```

### Download model in ONNX format

Mobile apps need ONNX (not PyTorch) to convert to TFLite (Android) or Core ML (iOS):

```bash
curl -H "X-API-Key: $NODE_KEY" \
  -o model.onnx \
  $SERVER_URL/api/v1/mobile/model/JOB_UUID/onnx
```

The ONNX file is cached server-side — conversion only happens once per round.

### Mobile training flow

1. **Checkin** → `POST /mobile/checkin` (get task + condition check)
2. **Download** → `GET /mobile/model/{job}/onnx` (ONNX binary)
3. **Convert on-device** → ONNX to TFLite (Android) or Core ML (iOS)
4. **Train locally** → using on-device ML framework
5. **Convert weights back** → to PyTorch state_dict format
6. **Submit** → `POST /nodes/task/{job}/submit` (same endpoint as desktop)

---

## 5. Monitor Progress

### Job status

```bash
curl -H "X-API-Key: $ORG_KEY" \
  $SERVER_URL/api/v1/jobs/JOB_UUID
```

Shows current round, status, and per-round submission counts.

### List all jobs

```bash
curl -H "X-API-Key: $ORG_KEY" \
  $SERVER_URL/api/v1/jobs
```

### Node stats

```bash
curl -H "X-API-Key: $NODE_KEY" \
  $SERVER_URL/api/v1/nodes/stats
```

Shows trust score, rounds completed, and node status.

### Cancel a job

```bash
curl -X DELETE -H "X-API-Key: $ORG_KEY" \
  $SERVER_URL/api/v1/jobs/JOB_UUID
```

---

## 6. Download Trained Models

### List checkpoints

```bash
curl -H "X-API-Key: $ORG_KEY" \
  $SERVER_URL/api/v1/inference/checkpoints/JOB_UUID
```

### Download latest checkpoint

```bash
# Download directly to file
curl -H "X-API-Key: $ORG_KEY" \
  -o checkpoint.pth \
  "$SERVER_URL/api/v1/inference/checkpoints/JOB_UUID/download"
```

If `S3_PUBLIC_URL` is configured on the server, the response is a JSON with a presigned URL instead:
```json
{
  "download_url": "https://...",
  "round_num": 9,
  "expires_in": 3600
}
```

### Download a specific round

```bash
curl -H "X-API-Key: $ORG_KEY" \
  -o checkpoint_r5.pth \
  "$SERVER_URL/api/v1/inference/checkpoints/JOB_UUID/download?round_num=5"
```

---

## 7. Authentication

All authenticated endpoints use the `X-API-Key` header.

| Key prefix | Who uses it | Endpoints |
|------------|-------------|-----------|
| `nxo_` | Organizations (admin, T3 app) | `/api/v1/jobs/*`, `/api/v1/inference/*` |
| `nxn_` | Compute nodes (desktop & mobile) | `/api/v1/nodes/*`, `/api/v1/mobile/*` |

Public endpoints (no key needed):
- `GET /health`
- `GET /ready`
- `GET /docs`
- `POST /api/v1/auth/register-node`
- `POST /api/v1/auth/register-org`

---

## 8. API Quick Reference

### Auth
```
POST /api/v1/auth/register-org     → Create org, get API key
POST /api/v1/auth/register-node    → Register node, get API key
```

### Jobs
```
POST   /api/v1/jobs                → Create training job
GET    /api/v1/jobs                → List jobs
GET    /api/v1/jobs/{id}           → Job detail + rounds
DELETE /api/v1/jobs/{id}           → Cancel job
```

### Nodes (desktop)
```
POST /api/v1/nodes/heartbeat          → Report health
GET  /api/v1/nodes/task               → Poll for training task
GET  /api/v1/nodes/task/{job}/model   → Download global model (PyTorch base64)
POST /api/v1/nodes/task/{job}/submit  → Submit trained weights
GET  /api/v1/nodes/stats              → Node stats & reputation
```

### Mobile
```
POST /api/v1/mobile/checkin            → Report conditions, get task if eligible
GET  /api/v1/mobile/model/{job}/onnx   → Download model (ONNX binary)
POST /api/v1/nodes/task/{job}/submit   → Submit weights (same as desktop)
```

### Inference
```
GET /api/v1/inference/checkpoints/{job}           → List checkpoints
GET /api/v1/inference/checkpoints/{job}/download   → Presigned download URL
```

---

## Deploy Your Own

### Railway (recommended)

1. Fork the repo
2. Create a Railway project with 3 services:
   - **PostgreSQL** — add from Railway's database menu
   - **MinIO** — Docker image `minio/minio`, start command: `minio server /data --console-address :9001`
   - **API** — connect your GitHub repo

3. Set environment variables on the API service:

| Variable | Value |
|----------|-------|
| `DATABASE_URL` | Reference Railway Postgres URL (auto-converted to asyncpg) |
| `S3_ENDPOINT_URL` | `http://<minio-service>.railway.internal:9000` |
| `S3_ACCESS_KEY` | `minioadmin` |
| `S3_SECRET_KEY` | Generate with `openssl rand -base64 24` |
| `S3_BUCKET` | `nexus` |
| `PORT` | `8000` |
| `LOG_LEVEL` | `info` |
| `SEED_ORG_NAME` | `your-org-name` |

4. Deploy — check logs for the seed org API key
5. Generate a public domain under API service → Settings → Networking

### Docker Compose (local / VPS)

```bash
git clone <repo-url>
cd nexus-node-fl-engine
cp .env.example .env
# Edit .env with your passwords
docker compose up -d
```

Services:
- API: `http://localhost:8000`
- Postgres: `localhost:5432`
- MinIO console: `http://localhost:9001`

---

## Troubleshooting

### Node can't connect to server
- Check the server URL (include `https://`)
- Verify the node API key is correct
- Run `curl $SERVER_URL/health` to confirm the server is up

### Job stuck at round N
- Check that all assigned nodes are running and submitting
- FedAvg only triggers when ALL assigned nodes submit for the current round
- Check node heartbeats: a node that stopped sending heartbeats may have crashed

### "Invalid API key" error
- Org endpoints need `nxo_` keys, node endpoints need `nxn_` keys
- Keys are shown only once at registration — if lost, register a new org/node

### Model download returns empty
- The initial checkpoint (round -1) is the pretrained model before any FL training
- Actual trained checkpoints appear after round 0 completes
