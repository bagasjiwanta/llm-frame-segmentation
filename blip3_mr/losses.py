import torch
import torch.linalg as LA
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from blip3_mr.config import Config
from blip3_mr.dataset import DataInfo


class TverskyLoss(torch.nn.Module):
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


class GeneralizedSoftDiceLoss(torch.nn.Module):
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


# copied from transformers.loss.loss_utils.ForCausalLMLoss
# adapted to use weight
def weighted_cross_entropy(logits, labels, vocab_size, reduction="mean", weight=None):
    logits = logits.float()
    # print(logits.shape, labels.shape, vocab_size, logits.device, labels.device, weight.device)
    if weight is not None:
        assert isinstance(weight, torch.Tensor) and weight.ndim == 1 and weight.size(0) == vocab_size
        weight = weight.to(logits.device)

    shift_labels = F.pad(labels, (0, 1), value=-100)
    shift_labels = shift_labels[..., 1:].contiguous()

    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    shift_labels = shift_labels.to(logits.device)
    loss = nn.functional.cross_entropy(logits, shift_labels, weight=weight, reduction=reduction)

    return loss


def extract_binary_mask_from_logits(
    logits: Tensor, input_ids: Tensor, num_frame: int, num_class: int, datainfo: DataInfo
) -> Tensor:
    # The predictions placed right after the assistant token
    # Example: <assistant>0011101010
    # However, the logits at assistant token is responsible to generate the first segment token

    B = logits.size(0)
    argmax_dim = 1 if B > 1 else 0
    # start_mask = ((input_ids == datainfo.token_assistant).int().argmax(dim=argmax_dim) + 1).view(B, 1)
    start_mask = (input_ids == datainfo.token_assistant).int().argmax(dim=argmax_dim).view(B, 1)

    # --- Extract the logits in the right areas
    start_offsets = torch.arange(num_frame).unsqueeze(0).to(logits.device, non_blocking=True)  # 1, Frame

    batch_idxs = torch.arange(B).view(B, 1, 1).expand(B, num_frame, num_class)
    frame_idxs = (start_offsets + start_mask).view(B, num_frame, 1).expand(B, num_frame, num_class)
    probs_idxs = torch.tensor([datainfo.token_zero, datainfo.token_one]).expand(B, num_frame, num_class)

    moment_logits = logits[batch_idxs, frame_idxs, probs_idxs]  # (Batch, Frame, 2)
    # targets_onehot = F.one_hot(answers, num_classes=2).float()  # (Batch, Frame, 2)
    return moment_logits


def setup_finetune_losses(
    config: Config, datainfo: DataInfo, device: torch.device, num_frame: int
) -> tuple[Tensor | None, GeneralizedSoftDiceLoss, TverskyLoss, nn.BCEWithLogitsLoss]:
    ce_weight = None
    if config.ce_pos_weight != 1.0:
        vocab_size = len(datainfo.dataloader.collate_fn.tokenizer)
        ce_weight = torch.ones(vocab_size, dtype=torch.float32)
        ce_weight[datainfo.token_one] = config.ce_pos_weight
        ce_weight = ce_weight.to(device)
    bce_weight = torch.ones((num_frame,)) * config.bce_pos_weight

    gdl = GeneralizedSoftDiceLoss("mean", soft_labels=config.soft_loss, weight_type=config.gd_norm).to(device)
    tvl = TverskyLoss("mean", soft_labels=config.soft_loss, beta=config.tvl_beta, alpha=1.0 - config.tvl_beta)
    bce = torch.nn.BCEWithLogitsLoss(pos_weight=bce_weight).to(device)
    return ce_weight, gdl, tvl, bce
