from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transaction_anomaly_detection.models.layers.batch_swap_noise import (
    BatchSwapNoise,
)
from transaction_anomaly_detection.models.layers.continuous_feature_normalizer import (
    ContinuousFeatureNormalizer,
)
from transaction_anomaly_detection.models.layers.lin_bin_drop import LinBnDrop
from transaction_anomaly_detection.models.layers.reconstruction_loss import (
    ReconstructionLoss,
)
from transaction_anomaly_detection.models.layers.shifted_sigmoid import ShiftedSigmoid


class Autoencoder(nn.Module):
    def __init__(
        self,
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]],
        ls_con_features: List[int],
        ae_activation: nn.Module,
    ):
        super().__init__()
        # Extract and Save Categorical Feature Specs
        self._dict_cat_feature_to_ls_categories: Dict[
            str, List[str]
        ] = self._get_dict_cat_feature_to_ls_categories(
            dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd
        )
        self._dict_cat_feature_to_n_embd: Dict[
            str, int
        ] = self._get_dict_cat_feature_to_n_embd(
            dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd
        )
        self._ls_cat_features: List[str] = self._get_ls_cat_features(
            dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd
        )
        self._ls_n_categories: List[int] = self._get_ls_n_categories(
            dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd
        )
        self._n_categorical_features: int = len(self._ls_cat_features)
        n_categories_total = sum(self._ls_n_categories)
        ls_n_embd = list(self._dict_cat_feature_to_n_embd.values())
        n_embed_total = sum(ls_n_embd)
        self._has_cat: bool = True if self._n_categorical_features > 0 else False
        # Extract and Save Continuous Feature Specs
        self._ls_con_features = ls_con_features
        self._n_continuous_features = len(self._ls_con_features)
        self._has_con = True if self._n_continuous_features > 0 else False
        if self._has_con:
            self._df_con_stats = self._get_default_df_con_stats(
                ls_con_features=self._ls_con_features
            )
        else:
            self._df_con_stats = None
        # Compute Standard Autoencoder External Dimension
        dim_ae_external = self._n_continuous_features + n_embed_total
        # Save Loss
        self._loss = ReconstructionLoss()
    @staticmethod
    def _get_dict_cat_feature_to_ls_categories(
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]]
    ) -> Dict[str, List[str]]:
        return {
            cat_feature: ls_categories_n_embd[0]
            for cat_feature, ls_categories_n_embd in dict_cat_feature_to_ls_categories_n_embd.items()
        }

    @staticmethod
    def _get_dict_cat_feature_to_n_embd(
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]]
    ) -> Dict[str, int]:
        return {
            cat_feature: ls_categories_n_embd[1]
            for cat_feature, ls_categories_n_embd in dict_cat_feature_to_ls_categories_n_embd.items()
        }

    @staticmethod
    def _get_ls_cat_features(
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]]
    ) -> List[str]:
        return list(dict_cat_feature_to_ls_categories_n_embd.keys())

    @staticmethod
    def _get_ls_n_categories(
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]]
    ) -> List[int]:
        return [
            len(ls_categories_n_embd[0])
            for ls_categories_n_embd in dict_cat_feature_to_ls_categories_n_embd.values()
        ]
    @classmethod
    def _get_default_df_con_stats(cls, ls_con_features: List[str]) -> pd.DataFrame:
        dict_con_feature_to_mean_std_min_max = {}
        for con_feature in ls_con_features:
            dict_con_feature_to_mean_std_min_max[con_feature] = (0, 1, -1, 1)
        df_con_stats = cls._dict_con_stats_to_df_con_stats(
            dict_con_feature_to_mean_std_min_max=dict_con_feature_to_mean_std_min_max
        )
        return df_con_stats

    @staticmethod
    def _dict_con_stats_to_df_con_stats(
        dict_con_feature_to_mean_std_min_max: Dict[
            str, Tuple[float, float, float, float]
        ]
    ) -> pd.DataFrame:
        df_con_stats = pd.DataFrame.from_dict(
            data=dict_con_feature_to_mean_std_min_max,
            orient="index",
            columns=["mean", "std", "min", "max"],
        )
        return df_con_stats
