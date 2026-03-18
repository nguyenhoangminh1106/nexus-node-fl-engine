"""Weighted Federated Averaging (FedAvg) — core aggregation algorithm.

Extracted from the original server.py. This is the heart of the FL system:
given N trained state_dicts with their sample counts, produce one aggregated
state_dict where each node's contribution is weighted by its dataset size.

    global[key] = sum(n_i * w_i[key]) / sum(n_i)
"""

import structlog
import torch

logger = structlog.get_logger()


def weighted_fedavg(
    submissions: list[tuple[dict, int]],
) -> dict:
    """Run weighted FedAvg over a list of (state_dict, n_samples) pairs.

    Args:
        submissions: List of (state_dict, n_samples) tuples from participating
                     nodes. Each state_dict must have the same keys.

    Returns:
        A new aggregated state_dict.
    """
    if not submissions:
        raise ValueError("Cannot run FedAvg with zero submissions")

    total_n = sum(n for _, n in submissions)
    if total_n == 0:
        raise ValueError("Total sample count is zero")

    keys = list(submissions[0][0].keys())
    new_sd: dict[str, torch.Tensor] = {}

    for key in keys:
        new_sd[key] = sum(
            (n / total_n) * sd[key].float()
            for sd, n in submissions
        )

    logger.info(
        "fedavg_complete",
        num_nodes=len(submissions),
        total_samples=total_n,
    )
    return new_sd
