import torch
import torch.nn as nn


class BatchSwapNoise(nn.Module):
    def __init__(self, swap_rate: float):
        super().__init__()
        self._swap_rate = swap_rate

    @property
    def swap_rate(self) -> float:
        return self._swap_rate

    def forward(self, x: torch.tensor) -> torch.tensor:
        if self.training:
            mask = torch.rand(x.size()) > (1 - self._swap_rate)
            l1 = torch.floor(torch.rand(x.size()) * x.size(0)).type(torch.LongTensor)
            l2 = mask.type(torch.LongTensor) * x.size(1)
            res = (l1 * l2).view(-1)
            idx = torch.arange(x.nelement()) + res
            idx[idx >= x.nelement()] = idx[idx >= x.nelement()] - x.nelement()
            return x.flatten()[idx].view(x.size())
        else:
            return x

    def __repr__(self):
        return f"BatchSwapNoise(swap_rate = {self._swap_rate})"
