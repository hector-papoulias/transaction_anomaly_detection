from typing import Optional
import torch.nn as nn


class LinBnDrop(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        activation: Optional[nn.Module] = None,
        bn: Optional[bool] = False,
        dropout_rate: Optional[float] = 0,
    ):
        super().__init__()

        self._fc = nn.Linear(dim_in, dim_out)

        if bn is True:
            self._bn = nn.BatchNorm1d(dim_out)
        else:
            self._bn = None

        # Activation
        self._activation = activation

        # Dropout layer
        self._dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self._fc(x)
        if self._bn is not None:
            x = self._bn(x)
        if self._activation is not None:
            x = self._activation(x)
        x = self._dropout(x)
        return x
