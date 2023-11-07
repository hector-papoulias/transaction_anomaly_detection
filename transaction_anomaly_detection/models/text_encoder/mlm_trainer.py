from tqdm.notebook import tqdm
from typing import Tuple, Dict, Union, Optional
from math import ceil
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
from transaction_anomaly_detection.models.tools.training.trainer import Trainer
from transaction_anomaly_detection.models.tools.training.early_stopping.standard import (
    StandardStopper,
)
from transaction_anomaly_detection.models.text_encoder.network import BERTEncoder


@dataclass
class MLMBatch:
    tokens: torch.tensor  # Shape: (B, T)
    targets: torch.tensor  # Shape: (B, T)
    mask: torch.tensor  # Shape: (B, T)


class MLMTrainer(Trainer):
    @classmethod
    def _produce_mlm_batch(
        cls,
        t_encoded_tokens: torch.tensor,  # Shape (B, T)
        random_resample_low: int,
        random_resample_high: int,
        mask_token_encoding: int,
        mlm_modification_ratio: Optional[float] = 0.15,
        mlm_hide_ratio: Optional[float] = 0.8,
        mlm_resample_ratio: Optional[float] = 0.1,
    ) -> MLMBatch:
        B, T = t_encoded_tokens.size()
        n_elements_to_modify_per_slice = ceil(mlm_modification_ratio * T)
        n_elements_to_hide = ceil(mlm_hide_ratio * n_elements_to_modify_per_slice)
        n_elements_to_resample = ceil(
            mlm_resample_ratio * n_elements_to_modify_per_slice
        )

        t_targets = t_encoded_tokens

        t_triary_mask = cls._generate_random_triary_mask(
            t_in=t_encoded_tokens,
            dim=1,
            n_category_b_per_slice=n_elements_to_hide,
            n_category_c_per_slice=n_elements_to_resample,
        )
        t_mask_modify = ~(t_triary_mask == 0)
        t_mask_hide = t_triary_mask == 1
        t_mask_resample = t_triary_mask == 2
        t_batch = cls._modify_entries(
            t_in=t_encoded_tokens, new_vals=mask_token_encoding, t_mask=t_mask_hide
        )
        t_batch = cls._replace_entries_with_random_integers(
            t_in=t_batch,
            t_mask=t_mask_resample,
            low=random_resample_low,
            high=random_resample_high,
        )
        return MLMBatch(
            tokens=t_batch,  # Shape (B, T),
            targets=t_targets,  # Shape (B, T),
            mask=t_mask_modify,  # Shape (B, T)
        )

    @classmethod
    def _generate_random_triary_mask(
        cls,
        t_in: torch.tensor,  # Shape (B, T),
        dim: int,
        n_category_b_per_slice: int,
        n_category_c_per_slice: int,
    ) -> torch.tensor:
        t_mask = torch.zeros(t_in.size(), dtype=torch.int)
        t_mask = t_mask.index_fill(
            dim=dim,
            index=torch.tensor(
                [x for x in range(n_category_b_per_slice)], dtype=torch.int64
            ),
            value=1,
        )
        t_mask = t_mask.index_fill(
            dim=dim,
            index=torch.tensor(
                [
                    x
                    for x in range(
                        n_category_b_per_slice,
                        n_category_b_per_slice + n_category_c_per_slice,
                    )
                ],
                dtype=torch.int64,
            ),
            value=2,
        )
        return cls._shuffle_tensor_along_dimension(
            t_input=t_mask, dim=dim
        )  # Shape (B, T)

    @staticmethod
    def _shuffle_tensor_along_dimension(
        t_input: torch.tensor, dim: int
    ) -> torch.tensor:
        num_rows, num_cols = t_input.size()
        random_indices = torch.argsort(torch.rand(num_rows, num_cols), dim=dim)
        return torch.gather(t_input, dim=dim, index=random_indices)

    @classmethod
    def _replace_entries_with_random_integers(
        cls, t_in: torch.tensor, t_mask: torch.tensor, low: int, high: int
    ) -> torch.tensor:
        t_sample = torch.randint_like(t_in, low=low, high=high)
        return cls._modify_entries(t_in=t_in, new_vals=t_sample, t_mask=t_mask)

    @staticmethod
    def _modify_entries(
        t_in: torch.tensor, new_vals: Union[torch.tensor, float], t_mask: torch.tensor
    ) -> torch.tensor:
        return t_in * ~t_mask + new_vals * t_mask
