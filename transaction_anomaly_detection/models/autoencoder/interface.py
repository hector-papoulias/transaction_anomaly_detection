from typing import List, Dict, Optional, Union, Tuple, Generator, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transaction_anomaly_detection.models.tools.tokenizer import Tokenizer
from transaction_anomaly_detection.models.autoencoder.network import Autoencoder
from transaction_anomaly_detection.models.autoencoder.trainer import AutoencoderTrainer


class TransactionAnomalyDetector:
    def __init__(
        self,
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]],
        ls_con_features: List[str],
        encoder_layer_szs: List[int],
        ae_activation: nn.Module,
        dropout_rate: Optional[float] = 0,
        batchswap_noise_rate: Optional[float] = 0,
    ):
        self._autoencoder = Autoencoder(
            dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd,
            ls_con_features=ls_con_features,
            encoder_layer_szs=encoder_layer_szs,
            ae_activation=ae_activation,
            dropout_rate=dropout_rate,
            batchswap_noise_rate=batchswap_noise_rate,
        )
        self._autoencoder.eval()
        self._dict_cat_feature_to_tokenizer = self._get_dict_cat_feature_to_tokenizer(
            dict_cat_feature_to_ls_categories=self._autoencoder.dict_cat_feature_to_ls_categories
        )
        self._reconstruction_loss_threshold: float = np.nan

    @property
    def has_cat(self) -> bool:
        return self._autoencoder.has_cat

    @property
    def ls_cat_features(self) -> List[str]:
        return self._autoencoder.ls_cat_features

    @property
    def dict_cat_feature_to_ls_categories(self) -> Dict[str, List[str]]:
        return self._autoencoder.dict_cat_feature_to_ls_categories

    @property
    def dict_cat_feature_to_n_embd(self) -> Dict[str, int]:
        return self._autoencoder.dict_cat_feature_to_n_embd

    @property
    def has_con(self) -> bool:
        return self._autoencoder.has_con

    @property
    def ls_con_features(self) -> List[str]:
        return self._autoencoder.ls_con_features

    @property
    def df_con_stats(self) -> Optional[pd.DataFrame]:
        return self._autoencoder.df_con_stats

    @property
    def autoencoder_architecture(self) -> str:
        return repr(self._autoencoder)

    @property
    def reconstruction_loss_threshold(self) -> float:
        return self._reconstruction_loss_threshold
    @staticmethod
    def _get_dict_cat_feature_to_tokenizer(
        dict_cat_feature_to_ls_categories: Dict[str, List[str]]
    ) -> Dict[str, Tokenizer]:
        dict_cat_feature_to_tokenizer = {
            cat_feature: Tokenizer(ls_tokens=ls_categories)
            for cat_feature, ls_categories in dict_cat_feature_to_ls_categories.items()
        }
        return dict_cat_feature_to_tokenizer
