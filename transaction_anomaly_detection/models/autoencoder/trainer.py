from tqdm.notebook import tqdm
from typing import Dict, Optional, Tuple, Generator
import numpy as np
import pandas as pd
import torch
from transaction_anomaly_detection.models.autoencoder.network import Autoencoder
from transaction_anomaly_detection.models.tools.early_stopping import EarlyStopper

class AutoencoderTrainer:
    @torch.no_grad()
    @staticmethod
    def _compute_val_loss(
        autoencoder: Autoencoder, t_dataset_val: torch.tensor
    ) -> float:
        _, _, _, t_val_loss, _, _ = autoencoder.forward(
            t_in=t_dataset_val, compute_loss=True, loss_batch_reduction="mean"
        )
        return t_val_loss.item()
    @torch.no_grad()
    @staticmethod
    def _get_batch_generator(
        t_dataset: torch.tensor,  # t_dataset shape: (n_records, n_cat_feautres + n_con_eatures)
        n_batches: int,
        sz_batch: int,
        random_seed: Optional[int] = None,
    ) -> Generator[torch.tensor, None, None]:
        for i in range(n_batches):
            start_idx = i * sz_batch
            end_idx = (i + 1) * sz_batch
            t_batch = t_dataset[start_idx:end_idx, :]
            yield t_batch  # t_batch shape: (B, n_cat_feautres + n_con_eatures)

    @torch.no_grad()
    @staticmethod
    def _shuffle_dataset(t_dataset: torch.tensor) -> torch.tensor:
        shuffled_indices = torch.randperm(t_dataset.size(0))
        return t_dataset[shuffled_indices]

    @torch.no_grad()
    @staticmethod
    def _split_dataset(
        t_dataset: torch.tensor,
        val_ratio: float,
    ) -> Tuple[torch.tensor, torch.tensor]:
        n_records = t_dataset.size(0)
        split_size = int(val_ratio * n_records)
        t_dataset_train = t_dataset[split_size:]
        t_dataset_val = t_dataset[:split_size]
        return t_dataset_train, t_dataset_val

    @staticmethod
    def _get_n_batches(n_records: int, sz_batch: int) -> int:
        return (n_records + sz_batch - 1) // sz_batch
