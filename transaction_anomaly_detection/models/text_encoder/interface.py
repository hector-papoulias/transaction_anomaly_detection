from typing import List, Dict, Union, Optional, Type, TypeVar
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributions as dist
from transaction_anomaly_detection.models.tools.tokenization.tokenizer import Tokenizer
from transaction_anomaly_detection.models.text_encoder.network import BERTEncoder
from transaction_anomaly_detection.models.text_encoder.mlm_trainer import MLMTrainer

TextEncoderType = TypeVar("TextEncoderType", bound="TextEncoder")


class TextEncoder:
    def __init__(
        self,
        ls_standard_tokens: List[str],
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
            ls_tokens=ls_standard_tokens,
            pad_token="pad",
            unk_token="unk",
            mask_token="mask",
        )
        self._bert_encoder = BERTEncoder(
            ls_standard_tokens=ls_standard_tokens,
            n_special_tokens=len(self._tokenizer.vocabulary) - len(ls_standard_tokens),
            pad_token_encoding=self._tokenizer.pad_token_encoding,
            max_len=self._max_n_standard_tokens_to_max_len(
                max_n_standard_tokens=max_n_standard_tokens
            ),
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
    def vocabulary(self) -> List[str]:
        return self._tokenizer.vocabulary

    @property
    def max_n_standard_tokens(self) -> int:
        return self._max_n_standard_tokens

    @property
    def d_model(self) -> int:
        return self._bert_encoder.d_model

    @property
    def n_encoder_layers(self) -> List[int]:
        return self._bert_encoder.n_encoder_layers

    @property
    def n_parallel_heads_per_layer(self) -> int:
        return self._bert_encoder.n_parallel_heads_per_layer

    @property
    def dim_feedforward(self) -> int:
        return self._bert_encoder.dim_feedforward

    @property
    def activation(self) -> nn.Module:
        return self._bert_encoder.activation

    @property
    def layer_norm_eps(self) -> float:
        return self._bert_encoder.layer_norm_eps

    @property
    def dropout_rate(self) -> float:
        return self._bert_encoder.dropout_rate

    @property
    def bert_encoder_architecture(self) -> str:
        return repr(self._bert_encoder)

    def get_n_params(self) -> int:
        return self._bert_encoder.get_n_params()

    def fit(
        self,
        corpus: List[str],
        val_ratio: float,
        sz_batch: int,
        learning_rate: float,
        patience: int,
        loss_delta_threshold: float,
        max_n_epochs: Optional[Union[int, float]] = np.nan,
        verbose: Optional[bool] = False,
    ) -> Dict[str, pd.Series]:
        t_dataset = self._prepare_t_input(
            input_text=corpus,
            max_len=self._bert_encoder.max_len,
            tokenizer=self._tokenizer,
        )
        self._bert_encoder, dict_loss_evolution = MLMTrainer.train(
            mlm_random_resample_low=min(self._tokenizer.regular_token_encodings),
            mlm_random_resample_high=1 + max(self._tokenizer.regular_token_encodings),
            mlm_mask_token_encoding=self._tokenizer.mask_token_encoding,
            bert_encoder=self._bert_encoder,
            t_dataset=t_dataset,
            val_ratio=val_ratio,
            sz_batch=sz_batch,
            learning_rate=learning_rate,
            patience=patience,
            loss_delta_threshold=loss_delta_threshold,
            max_n_epochs=max_n_epochs,
            verbose=verbose,
        )
        return dict_loss_evolution

    def encode(self, input_text: Union[str, List[str]]) -> torch.tensor:
        return self._encode(
            input_text=input_text,
            tokenizer=self._tokenizer,
            bert_encoder=self._bert_encoder,
        )

    def complete(
        self,
        ls_tokens: Union[str, List[Optional[str]]],
    ) -> List[str]:
        return self._complete(
            ls_tokens=ls_tokens,
            tokenizer=self._tokenizer,
            bert_encoder=self._bert_encoder,
            argmax_logits=True,
        )

    def export(self, path_export_dir: Path, model_name: str):
        os.makedirs(path_export_dir, exist_ok=True)
        path_model = self._get_path_model(
            path_export_dir=path_export_dir, model_name=model_name
        )
        torch.save(self._bert_encoder, path_model)

    @classmethod
    def load_exported_model(
        cls, path_export_dir: Path, model_name: str
    ) -> Type[TextEncoderType]:
        path_model = cls._get_path_model(
            path_export_dir=path_export_dir, model_name=model_name
        )
        bert_encoder = torch.load(path_model)
        kwargs = {
            "ls_standard_tokens": bert_encoder.ls_standard_tokens,
            "max_n_standard_tokens": cls._max_len_to_max_n_standard_tokens(
                bert_encoder.max_len
            ),
            "d_model": bert_encoder.d_model,
            "n_encoder_layers": bert_encoder.n_encoder_layers,
            "n_parallel_heads_per_layer": bert_encoder.n_parallel_heads_per_layer,
            "dim_feedforward": bert_encoder.dim_feedforward,
            "activation": bert_encoder.activation,
            "layer_norm_eps": bert_encoder.layer_norm_eps,
            "dropout_rate": bert_encoder.dropout_rate,
        }
        text_encoder = cls(**kwargs)
        text_encoder._bert_encoder = bert_encoder
        return text_encoder

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

    @staticmethod
    def _max_n_standard_tokens_to_max_len(max_n_standard_tokens: int) -> int:
        return max_n_standard_tokens + 1

    @staticmethod
    def _max_len_to_max_n_standard_tokens(max_len: int) -> int:
        return max_len - 1

    @staticmethod
    def _get_path_model(path_export_dir: Path, model_name: str) -> Path:
        filename = model_name + ".pth"
        return path_export_dir / filename
