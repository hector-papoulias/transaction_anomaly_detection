from typing import Optional
import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        max_len: int,
        d_model: int,
        n_encoder_layers: int,
        n_parallel_heads_per_layer: int,
        dim_feedforward: int,
        activation: nn.Module,
        layer_norm_eps: Optional[float] = 1e-5,
        dropout_rate: Optional[float] = 0,
    ):
        super().__init__()
        self._max_len = max_len
        self._d_model = d_model
        self._dim_feedforward = dim_feedforward
        self._t_positional_encoding = self._get_positional_encoding(
            n_embd=d_model, max_len=max_len
        )
        self._transformer_encoder_stack = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_parallel_heads_per_layer,
                    dim_feedforward=dim_feedforward,
                    activation=activation,
                    layer_norm_eps=layer_norm_eps,
                    dropout=dropout_rate,
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(n_encoder_layers)
            ]
        )

    def forward(self, t_embedded_tokens: torch.tensor) -> torch.tensor:
        t_in = t_embedded_tokens + self._t_positional_encoding
        return self._transformer_encoder_stack(t_in)

    @staticmethod
    def _get_positional_encoding(n_embd: int, max_len: int) -> torch.tensor:
        pos_encoding = torch.zeros(max_len, n_embd)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, n_embd, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / n_embd)
        )
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.detach()
