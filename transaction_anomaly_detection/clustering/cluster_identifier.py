from typing import List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
class ClusterIdentifier:

    @classmethod
    def _identify_clusters(
        cls,
        gdf_transactions=gdf_transactions,
        n_nbrs=n_nbrs,
        crs=crs,
        gdf_centers=gdf_centers,
    ) -> Tuple[pd.DataFrame, List[gpd.GeoDataFrame]]:
        distances, indices = cls._get_cluster_distances_indices(
            gdf_transactions=gdf_transactions,
            n_nbrs=n_nbrs,
            crs=crs,
            gdf_centers=gdf_centers,
        )
        df_cluster_stats, ls_clusters = cls.__format_cluster_output(
            gdf_transactions=gdf_transactions,
            cluster_distances=distances,
            cluster_indices=indices,
        )
        return df_cluster_stats, ls_clusters

    @staticmethod
    def _format_cluster_output(
        gdf_transactions: gpd.GeoDataFrame,
        cluster_distances: np.array,
        cluster_indices: np.array,
    ) -> Tuple[pd.DataFrame, List[gpd.GeoDataFrame]]:
        ls_clusters = []
        dict_cluster_stats = defaultdict(list)
        for i in range(len(cluster_indices)):
            gdf_cluster = gdf_transactions.iloc[cluster_indices[i]]
            n_spoofed = len(gdf_cluster[gdf_cluster["label"] == 1])
            n_nonspoofed = len(gdf_cluster[gdf_cluster["label"] == 0])
            class_balance = n_spoofed / (n_nonspoofed + n_spoofed)
            dict_cluster_stats["dist_mean"].append(cluster_distances[i].mean())
            dict_cluster_stats["dist_std"].append(cluster_distances[i].std())
            dict_cluster_stats["n_spoofed"].append(n_spoofed)
            dict_cluster_stats["n_nonspoofed"].append(n_nonspoofed)
            dict_cluster_stats["class_balance"].append(class_balance)
            ls_clusters.append(gdf_cluster)
        df_cluster_stats = pd.DataFrame.from_dict(dict_cluster_stats)
        df_cluster_stats["cluster_idx"] = df_cluster_stats.index
        df_cluster_stats.sort_values(by="dist_mean", ascending=True, inplace=True)
        return df_cluster_stats, ls_clusters

    @classmethod
    def _get_cluster_distances_indices(
        cls,
        gdf_transactions: gpd.GeoDataFrame,
        n_nbrs: int,
        crs: str,
        gdf_centers: Optional[gpd.GeoDataFrame] = None,
    ) -> Tuple[np.array, np.array]:
        # Change coordinates
        gdf_transactions_projected = gdf_transactions.to_crs(crs)

        # Get Ball Centers
        if gdf_centers is None:
            gdf_centers = gdf_transactions_projected[
                gdf_transactions_projected["label"] == 1
            ]
        arr_centers = np.column_stack(
            (gdf_centers["geometry"].x, gdf_centers["geometry"].y)
        )

        # Train Ball Tree
        ball_tree = BallTree(
            cls._sr_geometry_to_arr_coords(
                sr_geometry=gdf_transactions_projected["geometry"]
            )
        )

        # Query Ball Tree
        distances, indices = ball_tree.query(arr_centers, k=1 + n_nbrs)
        return distances, indices

    @staticmethod
    def _sr_geometry_to_arr_coords(sr_geometry: gpd.GeoSeries) -> np.array:
        return np.column_stack((sr_geometry.x, sr_geometry.y))
