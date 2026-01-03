import torch
import torch.linalg as LA
import torch.nn.functional as F
from torch import Tensor, nn


class TverskyLoss(nn.Module):
    """
    Tversky loss with soft labels support.

    Defined in https://arxiv.org/abs/1706.05721.

    Implementation is copied from https://github.com/Project-MONAI/MONAI/blob/1.5.0/monai/losses/tversky.py
    """

    def __init__(
        self,
        reduction: str = "mean",
        soft_labels: bool = True,
        smooth: float = 1e-5,
        alpha: float = 0.3,
        beta: float = 0.7,
    ):
        super().__init__()
        self.reduction = reduction
        self.smooth = smooth
        self.soft_labels = soft_labels
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        num_classes = inputs.shape[-1]

        probs = F.softmax(inputs, dim=-1).permute(0, 2, 1)

        if self.soft_labels:
            ground_truth = targets.permute(0, 2, 1)
        else:
            ground_truth = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 2, 1).float()

        pred = LA.vector_norm(probs, ord=1, dim=2)
        ground = LA.vector_norm(ground_truth, ord=1, dim=2)
        difference = LA.vector_norm(probs - ground_truth, ord=1, dim=2)

        tp = (pred + ground - difference) / 2
        fp = pred - tp
        fn = ground - tp

        tvl = 1.0 - (tp + self.smooth) / (tp + fp * self.alpha + fn * self.beta + self.smooth)

        if self.reduction == "mean":
            return tvl.mean()
        else:
            return tvl


class GeneralizedDiceLoss(nn.Module):
    """
    Computes the Generalized Dice Loss, adaptable for soft labels.

    As defined in: https://arxiv.org/abs/1707.03237

    Modified to be used for LLM logits.
    """

    def __init__(
        self,
        reduction: str = "mean",
        soft_labels: bool = True,
        smooth: float = 1e-5,
        weight_type: str = "square",
    ):
        """
        Args:
            reduction (str): Reduction to apply: 'none' | 'mean' | 'sum'.
            soft_labels (bool): Targets are soft probabilities.
            smooth (float): Smoothing for numerator, denominator, and weight calculations.
        """
        super().__init__()
        self.reduction = reduction
        self.smooth = smooth
        self.soft_labels = soft_labels
        assert weight_type in ("square", "linear")
        self.weight_type = weight_type

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """
        Computes the generalized (soft) dice loss
        Args:
            inputs (torch.Tensor): Raw logit predictions from LLM of shape (Batch, Token, Num_classes).
            targets (torch.Tensor): Ground truth.
                - If soft_labels is False: Class probs in `[B, T, C]` or class indices in `[B, T]` (will perform one hot for class indices)
                - If soft_labels is True: Class probs in `[B, T, C]`.
        """
        num_classes = inputs.shape[-1]

        probs = F.softmax(inputs, dim=-1).permute(0, 2, 1)  # B, C, T

        if self.soft_labels:
            ground_truth = targets.permute(0, 2, 1)  # B, C, T
        else:
            ground_truth = F.one_hot(targets.long(), num_classes=num_classes).permute(0, 2, 1)  # B, C, T

        intersection = torch.sum(probs * ground_truth, dim=2)

        sum_p = torch.sum(probs, dim=2)

        sum_r = torch.sum(ground_truth, dim=2)

        if self.weight_type == "square":
            weights = 1.0 / (torch.pow(sum_r, 2) + self.smooth)
        elif self.weight_type == "linear":
            weights = 1.0 / (sum_r + self.smooth)
        else:
            raise NotImplementedError(f"weight type {self.weight_type} is not implemented")

        numerator = torch.sum(weights * intersection, dim=1)

        denominator = torch.sum(weights * (sum_p + sum_r), dim=1)

        gdl = 1.0 - (2.0 * numerator + self.smooth) / (denominator + self.smooth)

        if self.reduction == "mean":
            return torch.mean(gdl)
        elif self.reduction == "sum":
            return torch.sum(gdl)
        else:
            return gdl
