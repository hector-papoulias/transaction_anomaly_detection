import torch
import torch.nn as nn


class ShiftedSigmoid(nn.Module):
    def __init__(self, t_min_values: torch.tensor, t_max_values: torch.tensor):
        super().__init__()
        self._min_vals = nn.Parameter(t_min_values, requires_grad=False)
        self._max_vals = nn.Parameter(t_max_values, requires_grad=False)

    @property
    def min_vals(self) -> nn.Parameter:
        return self._min_vals

    @min_vals.setter
    def min_vals(self, t_min_values: torch.tensor):
        self._min_vals = nn.Parameter(t_min_values, requires_grad=False)

    @property
    def max_vals(self) -> nn.Parameter:
        return self._max_vals

    @max_vals.setter
    def max_vals(self, t_max_values: torch.tensor):
        self._max_vals = nn.Parameter(t_max_values, requires_grad=False)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self._min_vals + (self._max_vals - self._min_vals) * torch.sigmoid(x)
