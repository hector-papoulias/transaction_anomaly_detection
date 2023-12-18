from tqdm.notebook import tqdm
from typing import Dict, Tuple, Generator, Union, Optional
import numpy as np
import pandas as pd
import torch
from transaction_anomaly_detection.models.tools.training.trainer import Trainer
from transaction_anomaly_detection.models.tools.training.early_stopping.standard import (
    StandardStopper,
)
from transaction_anomaly_detection.models.autoencoder.network import Autoencoder


class AutoencoderTrainer(Trainer):
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
        max_n_epochs: Optional[Union[int, float]] = np.nan,
        verbose: Optional[bool] = False,
    ) -> Tuple[Autoencoder, Dict[str, pd.Series]]:
        dict_loss_evolution_train = {}
        dict_loss_evolution_val = {}
        optimizer = torch.optim.Adam(
            params=list(autoencoder.parameters()), lr=learning_rate
        )
        early_stopper = StandardStopper(
            patience=patience,
            delta_threshold=loss_delta_threshold,
            max_n_epochs=max_n_epochs,
        )
        autoencoder.train()
        progress_bar = tqdm(total=1, desc="Training")
        while not early_stopper.stop:
            # Split, Shufle, Batch
            t_dataset_shuffled = cls.shuffle_dataset(t_dataset=t_dataset)
            t_dataset_train, t_dataset_val = cls.split_dataset(
                t_dataset=t_dataset, val_ratio=val_ratio
            )
            n_batches = cls.get_n_batches(
                n_records=len(t_dataset_train), sz_batch=sz_batch
            )
            gen_t_batches = cls.get_batch_generator(
                t_dataset=t_dataset_train, sz_batch=sz_batch
            )
            # Initialize Epoch Loss Records
            epoch_train_loss = 0
            if early_stopper.n_epochs_ellapsed == 0:
                epoch_val_loss = np.nan
            # Reset Progress Bar
            progress_bar.reset()
            progress_bar.total = n_batches
            progress_bar_desc = cls.get_progress_bar_desc(
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
                loss_evolution_update = cls.get_loss_evolution_update(
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
        train_recap = cls.get_train_recap(
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

    @staticmethod
    @torch.no_grad()
    def _compute_val_loss(
        autoencoder: Autoencoder, t_dataset_val: torch.tensor
    ) -> float:
        _, _, _, t_val_loss, _, _ = autoencoder.forward(
            t_in=t_dataset_val, compute_loss=True, loss_batch_reduction="mean"
        )
        return t_val_loss.item()
