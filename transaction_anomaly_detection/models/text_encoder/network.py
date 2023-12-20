from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from transaction_anomaly_detection.models.layers.transformer_encoder import (
    TransformerEncoder,
)
from transaction_anomaly_detection.models.layers.masked_loss import MaskedLCCELoss


class BERTEncoder(nn.Module):
    def __init__(
        self,
        ls_standard_tokens: List[str],
        n_special_tokens: int,
        pad_token_encoding: int,
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
        # Save Init Params
        self._ls_standard_tokens = ls_standard_tokens
        self._max_len = max_len
        self._d_model = d_model
        self._n_encoder_layers = n_encoder_layers
        self._n_parallel_heads_per_layer = n_parallel_heads_per_layer
        self._dim_feedforward = dim_feedforward
        self._activation = activation
        self._layer_norm_eps = layer_norm_eps
        self._dropout_rate = dropout_rate

        # Save Derived Attributes
        self._n_tokens = len(ls_standard_tokens) + n_special_tokens
        self._embedding = nn.Embedding(
            num_embeddings=self._n_tokens,
            embedding_dim=self._d_model,
            padding_idx=pad_token_encoding,
        )
        self._transformer_encoder = TransformerEncoder(
            max_len=max_len,
            d_model=d_model,
            n_encoder_layers=n_encoder_layers,
            n_parallel_heads_per_layer=n_parallel_heads_per_layer,
            dim_feedforward=dim_feedforward,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            dropout_rate=dropout_rate,
        )
        self._t_encoded_to_logits = nn.Linear(
            in_features=self._d_model, out_features=self._n_tokens, bias=True
        )
        self._loss = MaskedLCCELoss()

    # Expose Init Params
    @property
    def ls_standard_tokens(self) -> List[str]:
        return self._ls_standard_tokens

    @property
    def max_len(self) -> int:
        return self._max_len

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def n_encoder_layers(self) -> List[int]:
        return self._n_encoder_layers

    @property
    def n_parallel_heads_per_layer(self) -> int:
        return self._n_parallel_heads_per_layer

    @property
    def dim_feedforward(self) -> int:
        return self._dim_feedforward

    @property
    def activation(self) -> nn.Module:
        return self._activation

    @property
    def layer_norm_eps(self) -> float:
        return self._layer_norm_eps

    @property
    def dropout_rate(self) -> float:
        return self._dropout_rate

    # Expose Derived Attributes
    @property
    def n_tokens(self) -> int:
        return self._n_tokens

    def get_n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        t_encoded_tokens: torch.tensor,  # t_encoded_tokens shape: (B, T)
        t_targets: Optional[torch.tensor] = None,  # t_targets shape: (B, T)
        t_mask: Optional[torch.tensor] = None,  # t_mask shape: (B, T)
        loss_reduction: Optional[str] = None,
    ) -> Tuple[torch.tensor, torch.tensor, Optional[torch.tensor]]:
        t_embedded_tokens = self._embedding(
            t_encoded_tokens
        )  # t_embedded_tokens shape: (B, T, d_model)
        t_bert_encoding = self._transformer_encoder.forward(
            t_embedded_tokens
        )  # t_bert_encoding shape: (B, T, d_model)
        t_logits = self._t_encoded_to_logits(
            t_bert_encoding
        )  # t_logits shape: (B, T, n_tokens)
        t_loss = None
        if t_targets is not None and t_mask is not None:
            t_loss = self._loss(
                t_logits=t_logits,
                t_targets=t_targets,
                t_mask=t_mask,
                reduction=loss_reduction,
            )
            # t_loss shape: ()
        return t_bert_encoding, t_logits, t_loss
