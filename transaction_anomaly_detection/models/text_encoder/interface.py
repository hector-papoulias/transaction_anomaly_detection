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

    def encode(self, input_text: Union[str, List[str]]) -> torch.tensor:
        return self._encode(
            input_text=input_text,
            tokenizer=self._tokenizer,
            bert_encoder=self._bert_encoder,
        )

    def complete(
        self,
        ls_tokens: Union[str, List[Optional[str]]],
        argmax_logits: Optional[bool] = True,
    ) -> List[str]:
        return self._complete(
            ls_tokens=ls_tokens,
            tokenizer=self._tokenizer,
            bert_encoder=self._bert_encoder,
            argmax_logits=argmax_logits,
        )

    @classmethod
    @torch.no_grad()
    def _encode(
        cls,
        input_text: Union[str, List[str]],
        tokenizer: Tokenizer,
        bert_encoder: BERTEncoder,
    ) -> torch.tensor:
        t_encoded_tokens = cls._prepare_t_input(
            input_text=input_text, max_len=bert_encoder.max_len, tokenizer=tokenizer
        )
        # t_encoded_tokens shape: (B, T)
        t_bert_encoding, _, _ = bert_encoder(t_encoded_tokens=t_encoded_tokens)
        # t_bert_encoding shape: (B, T, d_model)
        return t_bert_encoding

    @classmethod
    @torch.no_grad()
    def _complete(
        cls,
        ls_tokens: Union[str, List[Optional[str]]],
        tokenizer: Tokenizer,
        bert_encoder: BERTEncoder,
        argmax_logits: bool,
    ) -> List[str]:
        ls_tokens = cls._replace_gaps_with_mask_token(
            ls_tokens=ls_tokens, mask_token=tokenizer.mask_token
        )
        t_encoded_tokens = cls._ls_tokens_to_tensor(
            ls_tokens=ls_tokens, max_len=bert_encoder.max_len, tokenizer=tokenizer
        ).unsqueeze(
            0
        )  # Shape: (1, T)
        _, t_logits, _ = bert_encoder(t_encoded_tokens=t_encoded_tokens)
        if argmax_logits:
            t_completions = torch.argmax(t_logits, dim=-1)
        else:
            categorical_distn = dist.Categorical(logits=t_logits)
            t_completions = categorical_distn.sample()
        t_mask = t_encoded_tokens == tokenizer.mask_token_encoding
        t_tokens_completed = t_encoded_tokens * ~t_mask + t_completions * t_mask
        ls_tokens_completed = t_tokens_completed.squeeze(0).tolist()
        return tokenizer.decode(ls_tokens_completed)

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

    @staticmethod
    def _replace_gaps_with_mask_token(
        ls_tokens: List[Optional[str]], mask_token: str
    ) -> List[str]:
        ls_tokens_filled = []
        for token in ls_tokens:
            if token is not None:
                ls_tokens_filled.append(token)
            else:
                ls_tokens_filled.append(mask_token)
        return ls_tokens_filled
