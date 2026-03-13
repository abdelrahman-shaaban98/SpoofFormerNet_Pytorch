import torch
import torch.nn as nn
from typing import Optional


class SpoofingLoss(nn.Module):
    """
    Give the majority class less weight and the minority class more weight to handle class imbalance.
    Add label smoothing for regularization to enhance generalization.
    """
    def __init__(self, class_weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.1):
        super().__init__()

        self.loss = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, labels)