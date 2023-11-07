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
    def train(
        cls,
        mlm_random_resample_low: int,
        mlm_random_resample_high: int,
        mlm_mask_token_encoding: int,
        bert_encoder: BERTEncoder,
        t_dataset: torch.tensor,
        val_ratio: float,
        sz_batch: int,
        learning_rate: float,
        patience: int,
        loss_delta_threshold: float,
        max_n_epochs: Optional[Union[int, float]] = np.nan,
        verbose: Optional[bool] = False,
    ) -> Tuple[BERTEncoder, Dict[str, pd.Series]]:
        dict_loss_evolution_train = {}
        dict_loss_evolution_val = {}
        optimizer = torch.optim.Adam(
            params=list(bert_encoder.parameters()), lr=learning_rate
        )
        early_stopper = StandardStopper(
            patience=patience,
            delta_threshold=loss_delta_threshold,
            max_n_epochs=max_n_epochs,
        )
        bert_encoder.train()
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
                t_dataset=t_dataset_train, n_batches=n_batches, sz_batch=sz_batch
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
                mlm_batch = cls._produce_mlm_batch(
                    t_encoded_tokens=t_batch,
                    random_resample_low=mlm_random_resample_low,
                    random_resample_high=mlm_random_resample_high,
                    mask_token_encoding=mlm_mask_token_encoding,
                )
                epoch_train_loss += cls._training_step(
                    bert_encoder=bert_encoder, optimizer=optimizer, mlm_batch=mlm_batch
                )
                progress_bar.update(1)
            # Track Loss Evolution
            epoch_train_loss = epoch_train_loss / n_batches
            val_data_mlm = cls._produce_mlm_batch(
                t_encoded_tokens=t_dataset_val,
                random_resample_low=mlm_random_resample_low,
                random_resample_high=mlm_random_resample_high,
                mask_token_encoding=mlm_mask_token_encoding,
            )
            epoch_val_loss = cls._compute_val_loss(
                bert_encoder=bert_encoder, val_data_mlm=val_data_mlm
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
            early_stopper.update(metric=epoch_val_loss, model=bert_encoder)
        # Terminate Progress Bar
        progress_bar.close()
        # Select Best Model
        bert_encoder = early_stopper.best_model
        bert_encoder.eval()
        train_recap = cls.get_train_recap(
            best_epoch=early_stopper.best_epoch, min_val_loss=early_stopper.best_metric
        )
        dict_loss_evolution = {
            "train": pd.Series(dict_loss_evolution_train),
            "val": pd.Series(dict_loss_evolution_val),
        }
        print(train_recap)
        return bert_encoder, dict_loss_evolution

    @staticmethod
    def _training_step(
        bert_encoder: BERTEncoder,
        optimizer: torch.optim.Optimizer,
        mlm_batch: MLMBatch,
    ) -> torch.tensor:
        optimizer.zero_grad()
        _, _, t_loss = bert_encoder.forward(
            t_encoded_tokens=mlm_batch.tokens,
            t_targets=mlm_batch.targets,
            t_mask=mlm_batch.mask,
            loss_reduction="mean",
        )
        t_loss.backward()
        optimizer.step()
        return t_loss.item()

    @staticmethod
    @torch.no_grad()
    def _compute_val_loss(bert_encoder: BERTEncoder, val_data_mlm: MLMBatch) -> float:
        _, _, t_val_loss = bert_encoder.forward(
            t_encoded_tokens=val_data_mlm.tokens,
            t_targets=val_data_mlm.targets,
            t_mask=val_data_mlm.mask,
            loss_reduction="mean",
        )
        return t_val_loss.item()

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
