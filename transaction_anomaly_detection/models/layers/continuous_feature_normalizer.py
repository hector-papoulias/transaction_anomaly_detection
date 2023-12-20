from typing import Optional
import torch
import torch.nn as nn


class ContinuousFeatureNormalizer(nn.Module):
    def __init__(self, t_means: torch.tensor, t_stds: torch.tensor):
        super().__init__()
        self._means = nn.Parameter(t_means, requires_grad=False)
        self._stds = nn.Parameter(t_stds, requires_grad=False)

    @property
    def means(self) -> nn.Parameter:
        return self._means

    @means.setter
    def means(self, t_means: torch.tensor):
        self._means = nn.Parameter(t_means, requires_grad=False)

    @property
    def stds(self) -> nn.Parameter:
        return self._stds

    @stds.setter
    def stds(self, t_stds: torch.tensor):
        self._stds = nn.Parameter(t_stds, requires_grad=False)

    def forward(
        self, x: torch.tensor, denormalize: Optional[bool] = False
    ) -> torch.tensor:
        if denormalize:
            return self._stds * x + self._means
        return (x - self._means) / self._stds
