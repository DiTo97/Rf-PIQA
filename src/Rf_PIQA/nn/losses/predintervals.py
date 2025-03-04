import torch
import torch.nn as nn
from torch.nn import functional as F

from Rf_PIQA.nn.common.predintervals import predinterval_to_point_estimate


def mpiw(U: torch.Tensor, L: torch.Tensor, k: torch.Tensor, eps: float = 1e-6) -> float:
    """The mean prediction interval width (MPIW) metric

    Notes
    -----
    The equation (4) in [1]_

    References
    ----------
    .. [1] E. Simhayev, K. Gilad, and R. Lior, 2020
       PIVEN: A DNN for Prediction Intervals with Specific Value Prediction
    """
    nume = torch.sum(torch.abs(U - L) * k)
    deno = torch.sum(k) + eps

    metric = nume / deno
    return metric


def picp(k: torch.Tensor) -> float:
    """The prediction interval coverage probability (PICP) metric

    Notes
    -----
    The equation (1) in [1]_

    References
    ----------
    .. [1] E. Simhayev, K. Gilad, and R. Lior, 2020
       PIVEN: A DNN for Prediction Intervals with Specific Value Prediction
    """
    metric = torch.mean(k)
    return metric


def predinterval_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    lambda_: float = 15.0,
    soft: float = 160.0,
    alpha: float = 0.05,
    eps: float = 1e-6,
) -> float:
    """The prediction interval (PI) loss
    
    Notes
    -----
    The loss introduced in section 4.2 in [1]_

    References
    ----------
    .. [1] E. Simhayev, K. Gilad, and R. Lior, 2020
       PIVEN: A DNN for Prediction Intervals with Specific Value Prediction
    """
    U = outputs[:, 0]  # U(x)
    L = outputs[:, 1]  # L(x)
    _ = outputs[:, 2]  # v(x)
    y = targets[:, 0]  # y(x)

    n = torch.Tensor(outputs.size(0))

    k_soft_upper = F.sigmoid(soft * (U - y))
    k_soft_lower = F.sigmoid(soft * (y - L))

    k_hard_upper = torch.clamp(torch.sign(U - y), 0.0)
    k_hard_lower = torch.clamp(torch.sign(y - L), 0.0)

    k_soft = k_soft_upper * k_soft_lower
    k_hard = k_hard_upper * k_hard_lower

    mpiw_capt = mpiw(U, L, k_hard, eps)
    picp_soft = picp(k_soft)

    penalty = 1 - alpha - picp_soft
    penalty = torch.square(torch.clamp(penalty, 0.0))

    interval_loss = mpiw_capt + torch.sqrt(n) * lambda_ * penalty

    return interval_loss


class PIVEN(nn.Module):
    """The PIVEN loss for regression with prediction intervals

    Notes
    -----
    The loss introduced in equation (8) in [1]_

    References
    ----------
    .. [1] E. Simhayev, K. Gilad, and R. Lior, 2020
       PIVEN: A DNN for Prediction Intervals with Specific Value Prediction
    """

    def __init__(
        self,
        lambda_: float = 15.0,
        soft: float = 160.0,
        alpha: float = 0.05,
        beta: float = 0.5,
        eps: float = 1e-6,
    ) -> None:
        super(PIVEN, self).__init__()

        self.lambda_ = lambda_
        self.soft = soft
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> float:
        interval_loss = predinterval_loss(
            outputs, targets, self.lambda_, self.soft, self.alpha, self.eps
        )  # equation (6)

        point_estimates = predinterval_to_point_estimate(outputs)  # equation (3)
        point_estimates = torch.reshape(point_estimates, (-1, 1))

        regloss = F.mse_loss(point_estimates, targets)  # equation (7)
        cumloss = self.beta * interval_loss + (1 - self.beta) * regloss  # equation (8)

        return cumloss
