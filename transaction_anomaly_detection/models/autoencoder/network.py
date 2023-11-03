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
    _dropout_rate_cat_decoder: float = 0.1

    def __init__(
        self,
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]],
        ls_con_features: List[int],
        ae_activation: nn.Module,
        encoder_layer_szs: List[int],
        dropout_rate: float,
        batchswap_noise_rate: float,
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
        # Save Batch Swap Noise Preprocessor
        self._batchswap_noise = BatchSwapNoise(swap_rate=batchswap_noise_rate)
        # Save Categorical Feature Preprocessor
        if self._has_cat:
            self._cat_preprocessor = self._get_cat_preprocessor(
                dict_cat_feature_to_ls_categories_n_embd=dict_cat_feature_to_ls_categories_n_embd,
                dropout_rate=dropout_rate,
            )
        # Save Continuous Feature Preprocessor
        if self._has_con:
            self._con_preprocessor = self._get_con_preprocessor(
                n_continuous_features=self._n_continuous_features,
                sr_feature_means=self._df_con_stats["mean"],
                sr_feature_stds=self._df_con_stats["std"],
            )
        # Save Standard AutoEncoder
        self._standard_autoencoder = self._get_standard_autoencoder(
            dim_external=dim_ae_external,
            encoder_layer_szs=encoder_layer_szs,
            activation=ae_activation,
            bn=True,
            dropout_rate=dropout_rate,
        )
        # Save Continuous Feature Postprocessor
        if self._has_con:
            self._con_postprocessor = self._get_con_postprocessor(
                dim_autoencoder_output=dim_ae_external,
                n_continuous_features=self._n_continuous_features,
                sr_min=self._df_con_stats["min"],
                sr_max=self._df_con_stats["max"],
                sr_mean=self._df_con_stats["mean"],
                sr_std=self._df_con_stats["std"],
            )
        # Save Categorical Feature Postprocessor
        if self._has_cat:
            self._cat_postprocessor = self._get_cat_postprocessor(
                dim_autoencoder_output=dim_ae_external,
                n_categories_total=n_categories_total,
                dropout_rate=self._dropout_rate_cat_decoder,
            )

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

    @staticmethod
    def _get_con_preprocessor(
        n_continuous_features: int,
        sr_feature_means: pd.Series,
        sr_feature_stds: pd.Series,
    ) -> nn.ModuleDict:
        con_preprocessor = nn.ModuleDict()
        con_preprocessor["normalizer"] = ContinuousFeatureNormalizer(
            t_means=torch.tensor(
                sr_feature_means.tolist(), dtype=torch.float, requires_grad=False
            ),
            t_stds=torch.tensor(
                sr_feature_stds.tolist(), dtype=torch.float, requires_grad=False
            ),
        )
        con_preprocessor["batch_norm"] = nn.BatchNorm1d(n_continuous_features)
        return con_preprocessor

    @staticmethod
    def _get_cat_preprocessor(
        dict_cat_feature_to_ls_categories_n_embd: Dict[str, Tuple[List[str], int]],
        dropout_rate: float,
    ) -> nn.ModuleDict:
        cat_preprocessor = nn.ModuleDict()
        cat_preprocessor["embeddings"] = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=len(ls_categories), embedding_dim=embedding_dim
                )
                for ls_categories, embedding_dim in dict_cat_feature_to_ls_categories_n_embd.values()
            ]
        )
        cat_preprocessor["dropout"] = nn.Dropout(dropout_rate)
        return cat_preprocessor

    @classmethod
    def _get_standard_autoencoder(
        cls,
        dim_external: int,
        encoder_layer_szs: List[int],
        activation: nn.Module,
        bn: bool,
        dropout_rate: float,
    ) -> nn.ModuleDict:
        standard_autoencoder = nn.ModuleDict()
        ls_encoder_layer_szs = [dim_external] + encoder_layer_szs
        standard_autoencoder["encoder"] = cls._get_mlp(
            layer_szs=ls_encoder_layer_szs,
            activation=activation,
            bn=bn,
            dropout_rate=dropout_rate,
        )
        standard_autoencoder["decoder"] = cls._get_mlp(
            layer_szs=ls_encoder_layer_szs[::-1],
            activation=activation,
            bn=bn,
            dropout_rate=dropout_rate,
        )
        return standard_autoencoder

    @staticmethod
    def _get_mlp(
        layer_szs: List[int], activation: nn.Module, bn: bool, dropout_rate: float
    ) -> nn.Sequential:
        mlp = nn.Sequential(
            *[
                LinBnDrop(
                    dim_in=dim_in,
                    dim_out=dim_out,
                    activation=activation,
                    bn=bn,
                    dropout_rate=dropout_rate,
                )
                for dim_in, dim_out in zip(layer_szs, layer_szs[1:])
            ]
        )
        return mlp

    @staticmethod
    def _get_cat_postprocessor(
        dim_autoencoder_output: int, n_categories_total: int, dropout_rate: float
    ) -> nn.ModuleDict:
        cat_postprocessor = nn.ModuleDict()
        cat_postprocessor["t_decoded_to_logits"] = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(
                in_features=dim_autoencoder_output,
                out_features=n_categories_total,
                bias=True,
            ),
        )
        return cat_postprocessor

    @classmethod
    def _get_con_postprocessor(
        cls,
        dim_autoencoder_output: int,
        n_continuous_features: int,
        sr_min: pd.Series,
        sr_max: pd.Series,
        sr_mean: pd.Series,
        sr_std: pd.Series,
    ) -> nn.ModuleDict:
        con_postprocessor = nn.ModuleDict()
        sr_low = cls._get_sr_low(sr_min=sr_min, sr_mean=sr_mean, sr_std=sr_std)
        sr_high = cls._get_sr_high(sr_max=sr_max, sr_mean=sr_mean, sr_std=sr_std)
        con_postprocessor["t_decoded_to_zscores"] = nn.Sequential(
            LinBnDrop(
                dim_in=dim_autoencoder_output,
                dim_out=n_continuous_features,
                activation=None,
                bn=False,
                dropout_rate=0,
            ),
            ShiftedSigmoid(
                t_min_values=torch.tensor(list(sr_low.values), requires_grad=False),
                t_max_values=torch.tensor(list(sr_high.values), requires_grad=False),
            ),
        )
        return con_postprocessor

    @staticmethod
    def _get_sr_low(
        sr_min: pd.Series, sr_mean: pd.Series, sr_std: pd.Series
    ) -> pd.Series:
        return (sr_min - sr_mean) / sr_std

    @staticmethod
    def _get_sr_high(
        sr_max: pd.Series, sr_mean: pd.Series, sr_std: pd.Series
    ) -> pd.Series:
        return (sr_max - sr_mean) / sr_std
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
