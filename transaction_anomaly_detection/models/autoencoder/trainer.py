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
    def _split_dataset(
        t_dataset: torch.tensor,
        val_ratio: float,
    ) -> Tuple[torch.tensor, torch.tensor]:
        n_records = t_dataset.size(0)
        split_size = int(val_ratio * n_records)
        t_dataset_train = t_dataset[split_size:]
        t_dataset_val = t_dataset[:split_size]
        return t_dataset_train, t_dataset_val
