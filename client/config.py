"""Node agent configuration — loaded from env, CLI args, or YAML file."""

from dataclasses import dataclass, field


@dataclass
class NodeConfig:
    # Server connection
    server_url: str = "http://localhost:8000"
    api_key: str = ""

    # Training
    epochs_per_round: int = 2
    batch_size: int = 32
    learning_rate: float = 0.001
    max_samples: int = 200  # 0 = no cap
    freeze_backbone: bool = False  # auto-enabled on CPU

    # Resource limits
    max_cpu_percent: int = 80
    max_gpu_percent: int = 90
    only_on_power: bool = False
    only_on_wifi: bool = False

    # Paths
    data_cache_dir: str = "~/.nexus/data"

    # Heartbeat
    heartbeat_interval: int = 30  # seconds
    poll_interval: int = 5  # seconds between task polls
