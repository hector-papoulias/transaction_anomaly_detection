from typing import List, Dict, Union, Optional
from tqdm.notebook import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as dist
from transaction_anomaly_detection.models.tools.tokenization.tokenizer import Tokenizer
from transaction_anomaly_detection.models.text_encoder.network import BERTEncoder
from transaction_anomaly_detection.models.text_encoder.mlm_trainer import MLMTrainer


class TextEncoder:
    def __init__(
        self,
        ls_tokens: List[str],
        max_n_standard_tokens: int,
        d_model: int,
        n_encoder_layers: List[int],
        n_parallel_heads_per_layer: int,
        dim_feedforward: int,
        activation: nn.Module,
        layer_norm_eps: Optional[float] = 1e-5,
        dropout_rate: Optional[float] = 0,
    ):
        self._max_n_standard_tokens = max_n_standard_tokens
        self._tokenizer = Tokenizer(
            ls_tokens=ls_tokens, pad_token="pad", unk_token="unk", mask_token="mask"
        )
        self._bert_encoder = BERTEncoder(
            n_tokens=len(self._tokenizer.vocabulary),
            pad_token_encoding=self._tokenizer.pad_token_encoding,
            max_len=1 + max_n_standard_tokens,
            d_model=d_model,
            n_encoder_layers=n_encoder_layers,
            n_parallel_heads_per_layer=n_parallel_heads_per_layer,
            dim_feedforward=dim_feedforward,
            activation=activation,
            layer_norm_eps=layer_norm_eps,
            dropout_rate=dropout_rate,
        )
        self._bert_encoder.eval()

    @property
    def max_n_standard_tokens(self) -> int:
        return self._max_n_standard_tokens

    @property
    def d_model(self) -> int:
        return self._bert_encoder.d_model

    @property
    def bert_encoder_architecture(self) -> str:
        return repr(self._bert_encoder)

    @property
    def vocabulary(self) -> List[str]:
        return self._tokenizer.vocabulary

    def get_n_params(self) -> int:
        return self._bert_encoder.get_n_params()
    @classmethod
    def _prepare_t_input(
        cls, input_text: Union[List[str], str], max_len: int, tokenizer: Tokenizer
    ) -> torch.tensor:
        if type(input_text) == str:
            input_text = [input_text]
        ls_t_examples = []
        for i, str_example in enumerate(input_text):
            t_example = cls._str_to_tensor(
                str_input=str_example, max_len=max_len, tokenizer=tokenizer
            )
            ls_t_examples.append(t_example)
        t_encoded_tokens = torch.stack(ls_t_examples)
        return t_encoded_tokens  # t_encoded_tokens shape: (B,T)

    @classmethod
    def _str_to_tensor(
        cls, str_input: str, max_len: int, tokenizer: Tokenizer
    ) -> torch.tensor:
        ls_tokens = tokenizer.str_to_ls_tokens(str_input=str_input)
        t_encoded_tokens = cls._ls_tokens_to_tensor(
            ls_tokens=ls_tokens, max_len=max_len, tokenizer=tokenizer
        )
        return t_encoded_tokens  # t_encoded_tokens shape: (T)

    @staticmethod
    def _ls_tokens_to_tensor(
        ls_tokens: List[str], max_len: int, tokenizer: Tokenizer
    ) -> torch.tensor:
        ls_tokens = tokenizer.pad(sequence=ls_tokens, pad_left=1)
        sz_example = len(ls_tokens)
        if sz_example > max_len:
            ls_tokens = ls_tokens[:max_len]
        if sz_example < max_len:
            ls_tokens = tokenizer.pad(
                sequence=ls_tokens, pad_right=max_len - sz_example
            )
        ls_encoded_tokens = tokenizer.encode(token_or_ls_tokens=ls_tokens)
        t_encoded_tokens = torch.tensor(ls_encoded_tokens)
        return t_encoded_tokens  # t_encoded_tokens shape: (T)
