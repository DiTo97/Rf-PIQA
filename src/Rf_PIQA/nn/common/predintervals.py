import torch
from torch.nn import functional as F


def predinterval_to_point_estimate(outputs: torch.Tensor) -> torch.Tensor:
    U = outputs[:, 0]
    L = outputs[:, 1]
    v = outputs[:, 2]

    point_estimate = v * U + (1 - v) * L
    return point_estimate


def predinterval_to_confidence(
    outputs: torch.Tensor, alpha: float = 0.05
) -> torch.Tensor:
    U = outputs[:, 0]
    L = outputs[:, 1]

    distance = U - L

    confidence = F.sigmoid(-distance)
    confidence = confidence * (1 - alpha)

    return confidence
