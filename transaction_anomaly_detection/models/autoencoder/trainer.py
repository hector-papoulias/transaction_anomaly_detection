from tqdm.notebook import tqdm
from typing import Dict, Optional, Tuple, Generator
import numpy as np
import pandas as pd
import torch
from transaction_anomaly_detection.models.autoencoder.network import Autoencoder
from transaction_anomaly_detection.models.tools.early_stopping import EarlyStopper


class AutoencoderTrainer:
    @classmethod
    def train(
        cls,
        autoencoder: Autoencoder,
        t_dataset: torch.tensor,
        val_ratio: float,
        sz_batch: int,
        learning_rate: float,
        patience: int,
        loss_delta_threshold: float,
        max_n_epochs: Optional[int] = None,
        verbose: Optional[bool] = False,
    ) -> Tuple[Autoencoder, Dict[str, pd.Series]]:
        dict_loss_evolution_train = {}
        dict_loss_evolution_val = {}
        optimizer = torch.optim.Adam(
            params=list(autoencoder.parameters()), lr=learning_rate
        )
        early_stopper = EarlyStopper(
            patience=patience,
            delta_threshold=loss_delta_threshold,
            max_n_epochs=max_n_epochs,
        )
        autoencoder.train()
        progress_bar = tqdm(total=1, desc="Training")
        while not early_stopper.stop:
            # Split, Shufle, Batch
            t_dataset_shuffled = cls._shuffle_dataset(t_dataset=t_dataset)
            t_dataset_train, t_dataset_val = cls._split_dataset(
                t_dataset=t_dataset, val_ratio=val_ratio
            )
            n_batches = cls._get_n_batches(
                n_records=len(t_dataset_train), sz_batch=sz_batch
            )
            gen_t_batches = cls._get_batch_generator(
                t_dataset=t_dataset_train, n_batches=n_batches, sz_batch=sz_batch
            )
            # Initialize Epoch Loss Records
            epoch_train_loss = 0
            if early_stopper.n_epochs_ellapsed == 0:
                epoch_val_loss = np.nan
            # Reset Progress Bar
            progress_bar.reset()
            progress_bar.total = n_batches
            progress_bar_desc = cls._get_progress_bar_desc(
                current_epoch=early_stopper.n_epochs_ellapsed,
                previous_epoch_val_loss=epoch_val_loss,
                min_loss=early_stopper.best_metric,
                best_epoch=early_stopper.best_epoch,
            )
            progress_bar.set_description(desc=progress_bar_desc)
            # Process Batches
            for t_batch in gen_t_batches:
                epoch_train_loss += cls._training_step(
                    autoencoder=autoencoder, optimizer=optimizer, t_batch=t_batch
                )
                progress_bar.update(1)
            # Track Loss Evolution
            epoch_train_loss = epoch_train_loss / n_batches
            epoch_val_loss = cls._compute_val_loss(
                autoencoder=autoencoder, t_dataset_val=t_dataset_val
            )

            dict_loss_evolution_train[
                f"Epoch {early_stopper.n_epochs_ellapsed}"
            ] = epoch_train_loss
            dict_loss_evolution_val[
                f"Epoch {early_stopper.n_epochs_ellapsed}"
            ] = epoch_val_loss
            if verbose:
                loss_evolution_update = cls._get_loss_evolution_update(
                    epoch=early_stopper.n_epochs_ellapsed,
                    train_loss=epoch_val_loss,
                    val_loss=epoch_val_loss,
                )
                print(loss_evolution_update)
            early_stopper.update(metric=epoch_val_loss, model=autoencoder)
        # Terminate Progress Bar
        progress_bar.close()
        # Select Best Model
        autoencoder = early_stopper.best_model
        autoencoder.eval()
        train_recap = cls._get_train_recap(
            best_epoch=early_stopper.best_epoch, min_val_loss=early_stopper.best_metric
        )
        dict_loss_evolution = {
            "train": pd.Series(dict_loss_evolution_train),
            "val": pd.Series(dict_loss_evolution_val),
        }
        print(train_recap)
        return autoencoder, dict_loss_evolution

    @staticmethod
    def _training_step(
        autoencoder: Autoencoder,
        optimizer: torch.optim.Optimizer,
        t_batch: torch.tensor,
    ) -> torch.tensor:
        optimizer.zero_grad()
        _, _, _, t_loss, _, _ = autoencoder(
            t_batch, compute_loss=True, loss_batch_reduction="mean"
        )
        t_loss.backward()
        optimizer.step()
        return t_loss.item()

    @torch.no_grad()
    @staticmethod
    def _compute_val_loss(
        autoencoder: Autoencoder, t_dataset_val: torch.tensor
    ) -> float:
        _, _, _, t_val_loss, _, _ = autoencoder.forward(
            t_in=t_dataset_val, compute_loss=True, loss_batch_reduction="mean"
        )
        return t_val_loss.item()

    @staticmethod
    def _get_progress_bar_desc(
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
    def _get_loss_evolution_update(
        epoch: int, train_loss: float, val_loss: float
    ) -> str:
        return f"Epoch {epoch}: train_loss = {train_loss}, val loss: {val_loss}"

    @staticmethod
    def _get_train_recap(best_epoch: int, min_val_loss: float) -> str:
        return f"Min Val Loss @Epoch {best_epoch}: {min_val_loss} "

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
