from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from torch.nn import functional as F


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        ls_cat_logits: Optional[List[torch.tensor]],
        # List with one element per categorical feature.
        # Element shape: (B, n_categories).
        ls_cat_targets: Optional[List[torch.tensor]],
        # List with one element per categorical feature.
        # Element shape: (B), containing the integer indices of the target.
        t_con_predictions: Optional[torch.tensor],
        # Shape: (B, n_features)
        t_con_targets: Optional[torch.tensor],
        # Shape: (B, n_features)
    ) -> Tuple[Optional[torch.tensor], Optional[torch.tensor]]:
        t_cat_losses, t_con_losses = None, None
        if ls_cat_logits is not None and ls_cat_targets is not None:
            ls_cat_losses = []
            for cat_inputs, cat_targets in zip(ls_cat_logits, ls_cat_targets):
                ls_cat_losses.append(
                    F.cross_entropy(cat_inputs, cat_targets.long(), reduction="none")
                )
            t_cat_losses = torch.stack(ls_cat_losses).t()
        if t_con_predictions is not None and t_con_targets is not None:
            t_con_losses = torch.pow(t_con_predictions - t_con_targets, 2)
        return t_cat_losses, t_con_losses
