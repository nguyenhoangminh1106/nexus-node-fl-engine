# Nexus Node FL Engine

Federated Learning orchestration service for distributed AI training. Built as the compute backbone for the Nexus Node platform.

## What This Does

A REST API server that coordinates federated learning across distributed compute nodes:

- **Organizations** create training jobs via API (or through the T3 web app)
- **Compute nodes** register, poll for tasks, train locally, and submit weights
- **Server** aggregates weights using Weighted FedAvg after each round
- **Checkpoints** are saved to S3-compatible storage after every round

The core FL algorithm (async weighted FedAvg with MobileNetV2 backbone freezing for CPU nodes) is preserved from the original research implementation.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│  FastAPI Server                                      │
│                                                      │
│  /api/v1/auth/*      → register orgs & nodes        │
│  /api/v1/jobs/*      → create & manage FL jobs       │
│  /api/v1/nodes/*     → heartbeat, poll, submit       │
│  /api/v1/inference/* → download trained models       │
│  /health, /ready     → health checks                 │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │Orchestrator│ │ FedAvg   │  │ Model Registry   │  │
│  │(rounds,   │ │(weighted │  │(plantnet, ...    │  │
│  │ routing)  │ │ avg)     │  │ extensible)      │  │
│  └──────────┘  └──────────┘  └──────────────────┘  │
│         │              │                             │
│  ┌──────┴──────────────┴─────┐                      │
│  │  PostgreSQL    S3/MinIO   │                      │
│  │  (jobs, nodes, (weights,  │                      │
│  │   rounds)      checkpoints)│                     │
│  └───────────────────────────┘                      │
└─────────────────────────────────────────────────────┘
        ▲              ▲              ▲
        │              │              │
   ┌────┴───┐    ┌────┴───┐    ┌────┴───┐
   │ Node 1 │    │ Node 2 │    │ Node N │
   │(client)│    │(client)│    │(client)│
   └────────┘    └────────┘    └────────┘
```

## Project Structure

```
nexus-node-fl-engine/
├── nexus/                      # Server package
│   ├── main.py                 # FastAPI entrypoint
│   ├── config.py               # Pydantic settings (env-based)
│   ├── api/                    # REST endpoints
│   │   ├── auth.py             # Register orgs & nodes
│   │   ├── jobs.py             # CRUD training jobs
│   │   ├── nodes.py            # Heartbeat, task poll, weight submit
│   │   ├── inference.py        # Checkpoint download
│   │   ├── mobile.py           # Mobile mining (checkin, ONNX download)
│   │   ├── health.py           # /health, /ready
│   │   └── deps.py             # Auth dependencies
│   ├── core/                   # Business logic
│   │   ├── fedavg.py           # Weighted FedAvg algorithm
│   │   ├── orchestrator.py     # Round lifecycle & aggregation
│   │   ├── model_registry.py   # Pluggable model definitions
│   │   ├── serialization.py    # State dict encode/decode
│   │   └── conversion.py       # PyTorch → ONNX conversion
│   ├── models/                 # ML model definitions
│   │   ├── base.py             # Abstract model interface
│   │   └── plantnet.py         # MobileNetV2 (original model)
│   ├── db/                     # Database layer
│   │   ├── models.py           # SQLAlchemy ORM models
│   │   ├── session.py          # Async engine & session
│   │   └── seed.py             # Initial org creation
│   └── storage/
│       └── s3.py               # S3/MinIO operations
├── client/                     # Node agent (runs on compute nodes)
│   ├── node.py                 # Long-running training agent
│   └── config.py               # Node configuration
├── tests/                      # Unit tests
├── alembic/                    # DB migrations
├── docker-compose.yml          # Full stack (API + Postgres + MinIO)
├── Dockerfile
└── pyproject.toml
```

## Quick Start

### 1. Start the stack

```bash
cp .env.example .env
docker compose up -d
```

This starts:
- **API server** on `http://localhost:8000`
- **PostgreSQL** on `localhost:5432`
- **MinIO** on `http://localhost:9000` (console: `http://localhost:9001`)

The API server auto-creates tables and prints a **seed organization API key** on first startup. Check the logs:

```bash
docker compose logs api
```

Look for:
```
  SEED ORGANIZATION CREATED
  Name    : nexus-admin
  API Key : nxo_xxxxxxxxxxxxx
  (save this — it won't be shown again)
```

### 2. Explore the API

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### 3. Register a compute node

```bash
# Desktop node
curl -X POST http://localhost:8000/api/v1/auth/register-node \
  -H "Content-Type: application/json" \
  -d '{"name": "my-gpu-node", "device_type": "desktop", "region": "vietnam"}'

# Mobile node
curl -X POST http://localhost:8000/api/v1/auth/register-node \
  -H "Content-Type: application/json" \
  -d '{"name": "phone-1", "device_type": "mobile", "region": "vietnam"}'
```

Save the returned `api_key` (starts with `nxn_`).

### 4. Create a training job

```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "X-API-Key: nxo_YOUR_ORG_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "plant-disease-v1",
    "model_type": "plantnet",
    "num_classes": 38,
    "total_rounds": 10,
    "min_nodes": 2,
    "node_ids": ["NODE_UUID_1", "NODE_UUID_2"]
  }'
```

### 5. Start a compute node

```bash
python -m client.node \
  --server http://localhost:8000 \
  --api-key nxn_YOUR_NODE_KEY \
  --data-dir ./plantvillage \
  --epochs 2 \
  --batch-size 32
```

The node will:
1. Send heartbeats every 30s
2. Poll for assigned tasks
3. Download the global model
4. Train locally
5. Submit weights
6. Wait for the next round

## Local Development (without Docker)

```bash
# Create venv
python -m venv .venv && source .venv/bin/activate

# Install with dev deps
pip install -e ".[dev]"

# Start Postgres and MinIO (or use Docker for just these)
docker compose up -d db minio

# Copy and edit env
cp .env.example .env

# Run the server
uvicorn nexus.main:app --reload --port 8000

# Run tests
pytest tests/ -v
```

## API Reference

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register-org` | Create org, get API key |
| POST | `/api/v1/auth/register-node` | Register node, get API key |

### Jobs (org API key required via `X-API-Key` header)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/jobs` | Create training job |
| GET | `/api/v1/jobs` | List org's jobs |
| GET | `/api/v1/jobs/{id}` | Job detail + rounds |
| DELETE | `/api/v1/jobs/{id}` | Cancel job |

### Nodes — Desktop (node API key required via `X-API-Key` header)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/nodes/heartbeat` | Report health |
| GET | `/api/v1/nodes/task` | Poll for training task |
| GET | `/api/v1/nodes/task/{job_id}/model` | Download global model (PyTorch base64) |
| POST | `/api/v1/nodes/task/{job_id}/submit` | Submit trained weights |
| GET | `/api/v1/nodes/stats` | Node reputation & stats |

### Mobile (node API key required via `X-API-Key` header)
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/mobile/checkin` | Report device conditions, get task if eligible |
| GET | `/api/v1/mobile/model/{job_id}/onnx` | Download model in ONNX format |

Mobile nodes submit weights via the same `POST /api/v1/nodes/task/{job_id}/submit` endpoint as desktop.

### Inference (org API key required)
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/inference/checkpoints/{job_id}` | List checkpoints |
| GET | `/api/v1/inference/checkpoints/{job_id}/download` | Presigned download URL |

### Health
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness |
| GET | `/ready` | Readiness (DB check) |

## FL Training Flow

```
Round N:
  Desktop nodes:
    1. Poll GET /nodes/task → receive task
    2. Download model GET /nodes/task/{job}/model → PyTorch base64
    3. Train locally with PyTorch
    4. Submit weights POST /nodes/task/{job}/submit

  Mobile nodes:
    1. Checkin POST /mobile/checkin → get task (checks battery/wifi/charging)
    2. Download model GET /mobile/model/{job}/onnx → ONNX binary
    3. Convert ONNX → TFLite (Android) or Core ML (iOS) on-device
    4. Train locally with on-device ML framework
    5. Convert weights back to PyTorch format
    6. Submit weights POST /nodes/task/{job}/submit (same as desktop)

  Server (automatic):
    - When all assigned nodes submit:
      a. Load all weights from S3
      b. Run weighted FedAvg: global[k] = Σ(nᵢ/N) * wᵢ[k]
      c. Save aggregated checkpoint to S3
      d. Advance to Round N+1
    - Job completes when all rounds are done
```

## Adding a New Model

1. Create `nexus/models/mymodel.py` implementing `BaseModelDef`:

```python
from nexus.models.base import BaseModelDef

class MyModelDef(BaseModelDef):
    def build(self, num_classes, pretrained=True):
        # Return an nn.Module
        ...
    def freeze_for_client(self, model):
        # Freeze heavy layers for CPU nodes
        ...
    def unfreeze_all(self, model):
        ...
    def train_transform(self):
        # Return torchvision.transforms.Compose
        ...
    def val_transform(self):
        ...
```

2. Register it in `nexus/core/model_registry.py`:

```python
from nexus.models.mymodel import MyModelDef
register("mymodel", MyModelDef())
```

3. Create jobs with `"model_type": "mymodel"`

## Database Schema

| Table | Purpose |
|-------|---------|
| `organizations` | B2B tenants with hashed API keys |
| `jobs` | Training job config (model, rounds, status) |
| `rounds` | Per-round state tracking |
| `nodes` | Compute nodes with device type (desktop/mobile), hardware info & trust scores |
| `node_assignments` | Which nodes are assigned to which jobs |
| `submissions` | Weight submissions per round per node |
| `checkpoints` | Saved model checkpoints (S3 paths) |

For production, use Alembic migrations:

```bash
alembic revision --autogenerate -m "description"
alembic upgrade head
```

In development, tables are auto-created on startup via `Base.metadata.create_all`.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://nexus:nexus@localhost:5432/nexus` | Postgres connection |
| `S3_ENDPOINT_URL` | `http://localhost:9000` | S3/MinIO internal endpoint |
| `S3_PUBLIC_URL` | (empty) | Public S3 URL for presigned downloads. If empty, files are proxied through the API |
| `S3_ACCESS_KEY` | `minioadmin` | S3 access key |
| `S3_SECRET_KEY` | `minioadmin` | S3 secret key |
| `S3_BUCKET` | `nexus` | Bucket name |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |
| `LOG_LEVEL` | `info` | Logging level |
| `SEED_ORG_NAME` | `nexus-admin` | Initial org name |

## Node Agent Flags

```
python -m client.node --help
```

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `http://localhost:8000` | Server URL |
| `--api-key` | required | Node API key from registration |
| `--data-dir` | required | Local dataset directory (ImageFolder layout) |
| `--epochs` | 2 | Training epochs per round |
| `--batch-size` | 32 | Training batch size |
| `--lr` | 0.001 | SGD learning rate |
| `--max-samples` | 200 | Max training samples (0 = all) |
| `--freeze-backbone` | auto (on CPU) | Freeze backbone for fast CPU training |
| `--poll-interval` | 5 | Seconds between task polls |

## Original Research

The core FL algorithm is based on async weighted FedAvg for plant disease detection using the PlantVillage dataset (38 classes). The original implementation is preserved in `nexus/models/plantnet.py` and `nexus/core/fedavg.py`.

Key design decisions from the original:
- **Server as participant**: server trains on its own data partition, not just aggregates
- **Async training**: no per-batch synchronization, only per-round
- **Backbone freezing**: MobileNetV2 features frozen on CPU nodes (~10x speedup, ~0.3M vs ~3.4M trainable params)
- **Weighted FedAvg**: nodes with more samples have proportionally more influence

The original standalone scripts (`model.py`, `server.py`, `client.py`, `infer.py`, `evaluate.py`) are preserved in the repo root for reference but are superseded by the `nexus/` and `client/` packages.
