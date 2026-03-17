"""Nexus Node Agent — long-running compute node that trains FL models.

Replaces the original client.py with a proper agent that:
  - Registers with the server and sends periodic heartbeats
  - Polls for assigned training tasks
  - Downloads global model, trains locally, submits weights
  - Reports hardware info (CPU/GPU/RAM)
  - Respects resource limits

Usage:
    python -m client.node \\
        --server http://localhost:8000 \\
        --api-key nxn_xxxxx \\
        --data-dir ./plantvillage
"""

import argparse
import platform
import sys
import threading
import time

import httpx
import structlog
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from client.config import NodeConfig

# Lazy import: model registry lives in the nexus package. For the standalone
# node client we import the model classes directly to avoid requiring the
# full server dependencies.
sys.path.insert(0, ".")
from nexus.core import model_registry
from nexus.core.serialization import deserialize_state_dict, serialize_state_dict

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)
logger = structlog.get_logger()


def _get_hardware_info() -> dict:
    """Collect hardware info to report to the server."""
    info = {
        "platform": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_mb"] = torch.cuda.get_device_properties(0).total_mem // (1024 * 1024)
    try:
        import psutil

        info["cpu_count"] = psutil.cpu_count()
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except ImportError:
        pass
    return info


class NexusNode:
    """Long-running FL compute node agent."""

    def __init__(self, config: NodeConfig, data_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client = httpx.Client(
            base_url=config.server_url,
            headers={"X-API-Key": config.api_key},
            timeout=300.0,
        )
        self._stop_event = threading.Event()

        # Auto-freeze on CPU
        if not config.freeze_backbone and self.device.type == "cpu":
            self.config.freeze_backbone = True
            logger.info("auto_freeze_backbone", reason="no GPU detected")

    def _heartbeat_loop(self):
        """Background thread: send heartbeats every N seconds."""
        while not self._stop_event.is_set():
            try:
                resp = self.client.post(
                    "/api/v1/nodes/heartbeat",
                    json={
                        "hardware_info": _get_hardware_info(),
                        "status": "online",
                    },
                )
                if resp.status_code != 200:
                    logger.warning("heartbeat_failed", status=resp.status_code)
            except Exception as e:
                logger.warning("heartbeat_error", error=str(e))
            self._stop_event.wait(self.config.heartbeat_interval)

    def _poll_for_task(self) -> dict | None:
        """Poll the server for an assigned task."""
        try:
            resp = self.client.get("/api/v1/nodes/task")
            if resp.status_code == 200:
                data = resp.json()
                return data if data else None
        except Exception as e:
            logger.warning("poll_error", error=str(e))
        return None

    def _download_model(self, job_id: str) -> tuple[dict, int]:
        """Download the current global model for a job."""
        resp = self.client.get(f"/api/v1/nodes/task/{job_id}/model")
        resp.raise_for_status()
        data = resp.json()
        sd = deserialize_state_dict(data["model"])
        return sd, data["round"]

    def _submit_weights(self, job_id: str, round_num: int, state_dict: dict, n_samples: int):
        """Submit trained weights back to the server."""
        payload = {
            "round_num": round_num,
            "n_samples": n_samples,
            "weights": serialize_state_dict(state_dict),
        }
        resp = self.client.post(f"/api/v1/nodes/task/{job_id}/submit", json=payload)
        resp.raise_for_status()
        return resp.json()

    def _train_round(self, task: dict) -> None:
        """Execute one training round for a task."""
        job_id = task["job_id"]
        round_num = task["round_num"]
        model_type = task["model_type"]
        num_classes = task["num_classes"]

        logger.info("training_start", job=job_id, round=round_num, model=model_type)

        # Build model
        model_def = model_registry.get(model_type)
        model = model_def.build(num_classes=num_classes, pretrained=False).to(self.device)

        # Download and load global weights
        global_sd, _ = self._download_model(job_id)
        model.load_state_dict(global_sd)

        # Configure freezing
        if self.config.freeze_backbone:
            model_def.freeze_for_client(model)

        # Prepare data
        transform = model_def.train_transform()
        full_dataset = datasets.ImageFolder(root=self.data_dir, transform=transform)
        n_total = len(full_dataset)
        max_s = self.config.max_samples
        end = min(n_total, max_s) if max_s > 0 else n_total
        subset = Subset(full_dataset, list(range(end)))
        loader = DataLoader(
            subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        n_samples = len(subset)

        # Train
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.config.learning_rate,
            momentum=0.9,
            weight_decay=1e-4,
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(self.config.epochs_per_round):
            model.train()
            total_loss = correct = total = 0
            pbar = tqdm(
                loader,
                desc=f"R{round_num} E{epoch+1}/{self.config.epochs_per_round}",
                unit="batch",
            )
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                correct += output.argmax(1).eq(target).sum().item()
                total += target.size(0)
                pbar.set_postfix(
                    loss=f"{loss.item():.3f}",
                    acc=f"{100 * correct / total:.1f}%",
                )

        # Submit weights
        cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
        result = self._submit_weights(job_id, round_num, cpu_sd, n_samples)
        logger.info("training_complete", job=job_id, round=round_num, result=result.get("status"))

    def run(self):
        """Main loop: heartbeat + poll for tasks + train."""
        logger.info(
            "node_starting",
            server=self.config.server_url,
            device=str(self.device),
            freeze=self.config.freeze_backbone,
        )

        # Start heartbeat thread
        hb_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb_thread.start()

        logger.info("polling_for_tasks")

        while not self._stop_event.is_set():
            task = self._poll_for_task()
            if task is None:
                self._stop_event.wait(self.config.poll_interval)
                continue

            try:
                self._train_round(task)
            except Exception as e:
                logger.error("training_error", error=str(e))
                time.sleep(5)

    def stop(self):
        self._stop_event.set()


def main():
    parser = argparse.ArgumentParser(description="Nexus Node Agent")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="Server URL")
    parser.add_argument("--api-key", type=str, required=True,
                        help="Node API key (from registration)")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Local dataset directory (ImageFolder layout)")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Training epochs per round")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-samples", type=int, default=200,
                        help="Max training samples (0 = all)")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=5,
                        help="Seconds between task polls")
    args = parser.parse_args()

    config = NodeConfig(
        server_url=args.server,
        api_key=args.api_key,
        epochs_per_round=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_samples=args.max_samples,
        freeze_backbone=args.freeze_backbone,
        poll_interval=args.poll_interval,
    )

    node = NexusNode(config, data_dir=args.data_dir)
    try:
        node.run()
    except KeyboardInterrupt:
        logger.info("shutting_down")
        node.stop()


if __name__ == "__main__":
    main()
