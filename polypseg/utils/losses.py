from typing import Optional

import torch
from torch import nn

from polypseg.utils.functional import cross_entropy, dice_loss, sigmoid_focal_loss

__all__ = ["DiceLoss", "DiceCELoss", "CrossEntropyLoss", "FocalLoss"]


class CrossEntropyLoss(nn.Module):
    """Cross Entropy Loss
    Args:
        class_weights (torch.Tensor): class weights
        reduction (str): reduction type
        loss_weight (float): loss weight
    """

    def __init__(
        self,
        class_weights: Optional[torch.Tensor] = None,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.class_weight = class_weights
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        loss = self.loss_weight * cross_entropy(
            inputs,
            targets,
            weight,
            class_weight=self.class_weight,
            reduction=self.reduction,
            ignore_index=ignore_index,
        )

        return loss


class DiceLoss(nn.Module):
    """Dice Loss
    Args:
        reduction (str): reduction type
        loss_weight (float): weight for a loss function
        eps (float): epsilon
    """

    def __init__(
        self,
        reduction: str = "mean",
        loss_weight: Optional[float] = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ):
        loss = self.loss_weight * dice_loss(
            inputs,
            targets,
            weight=weight,
            reduction=self.reduction,
            eps=self.eps,
        )

        return loss


class DiceCELoss(nn.Module):
    """Dice + Cross Entropy Loss
    Args:
        reduction (str): reduction type
        dice_weight (float): dice loss weight
        ce_weight (float): cross entropy loss weight
        eps (float): epsilon
    """

    def __init__(
        self,
        reduction: str = "mean",
        dice_weight: float = 1.0,
        ce_weight: float = 1.0,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.reduction = reduction
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.eps = eps

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ):
        # calculate dice loss
        dice = dice_loss(
            inputs,
            targets,
            weight=weight,
            reduction=self.reduction,
            eps=self.eps,
        )
        # calculate cross entropy loss
        ce = cross_entropy(inputs, targets, weight=weight, reduction=self.reduction)
        # accumulate loss according to given weights
        loss = self.dice_weight * dice + ce * self.ce_weight

        return loss


class FocalLoss(nn.Module):
    """Sigmoid Focal Loss
    Args:
        gamma (float): gamma parameter
        alpha (float): alpha parameter
        reduction (str): reduction type
        loss_weight (float): focal loss weight
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        reduction: str = "mean",
        loss_weight: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ):
        loss = self.loss_weight * sigmoid_focal_loss(
            inputs,
            targets,
            weight,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction,
        )

        return loss
