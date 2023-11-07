from tqdm.notebook import tqdm
from typing import Dict, Tuple, Generator, Optional
from abc import ABC, abstractmethod
import pandas as pd
import torch
import torch.nn as nn


class Trainer(ABC):
    @classmethod
    @abstractmethod
    def train(
        cls, model: nn.Module, **kwargs
    ) -> Tuple[nn.Module, Dict[str, pd.Series]]:
        # Train the model. Return trained model, train loss evolution and val loss evolution.
        pass

    @staticmethod
    def get_progress_bar_desc(
        current_epoch: int,
        previous_epoch_val_loss: float,
        min_loss: float,
        best_epoch: int,
    ) -> str:
        progress_bar_desc = f"Current Epoch: {current_epoch}." + "\t"
        progress_bar_desc += (
            f"Previous Epoch Val Loss: {previous_epoch_val_loss}." + "\t"
        )
        progress_bar_desc += f"Min Val Loss: {min_loss} @ Epoch {best_epoch}."
        return progress_bar_desc

    @staticmethod
    def get_loss_evolution_update(
        epoch: int, train_loss: float, val_loss: float
    ) -> str:
        return f"Epoch {epoch}: train_loss = {train_loss}, val loss: {val_loss}"

    @staticmethod
    def get_train_recap(best_epoch: int, min_val_loss: float) -> str:
        return f"Min Val Loss @Epoch {best_epoch}: {min_val_loss} "

    @staticmethod
    @torch.no_grad()
    def get_batch_generator(
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

    @staticmethod
    @torch.no_grad()
    def shuffle_dataset(t_dataset: torch.tensor) -> torch.tensor:
        shuffled_indices = torch.randperm(t_dataset.size(0))
        return t_dataset[shuffled_indices]

    @staticmethod
    @torch.no_grad()
    def split_dataset(
        t_dataset: torch.tensor,
        val_ratio: float,
    ) -> Tuple[torch.tensor, torch.tensor]:
        n_records = t_dataset.size(0)
        split_size = int(val_ratio * n_records)
        t_dataset_train = t_dataset[split_size:]
        t_dataset_val = t_dataset[:split_size]
        return t_dataset_train, t_dataset_val

    @staticmethod
    def get_n_batches(n_records: int, sz_batch: int) -> int:
        return (n_records + sz_batch - 1) // sz_batch
