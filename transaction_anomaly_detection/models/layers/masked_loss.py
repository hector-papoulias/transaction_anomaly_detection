from typing import Optional
import torch
import torch.nn as nn


class MaskedLCCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        t_logits: torch.tensor,  # Shape: (B, T, n_tokens)
        t_targets: torch.tensor,  # Shape: (B, T)
        t_mask: torch.tensor,  # Shape: (B, T)
        reduction: Optional[str] = None,
    ) -> torch.tensor:
        B, T, n_tokens = t_logits.size()
        t_loss = nn.functional.cross_entropy(
            t_logits.view(B * T, n_tokens), t_targets.view(B * T), reduction="none"
        ).view(B, T)
        t_loss = t_loss * t_mask  # t_loss shape: (B, T)
        if reduction == "sum":
            t_loss_sum = torch.sum(t_loss)
            return t_loss_sum  # t_loss_sum shape: ()
        if reduction == "mean":
            n_examples = len(torch.nonzero(t_mask))
            t_loss_mean = torch.sum(t_loss) / n_examples
            return t_loss_mean  # t_loss_mean shape: ()
        return t_loss  # t_loss shape: (B, T)
