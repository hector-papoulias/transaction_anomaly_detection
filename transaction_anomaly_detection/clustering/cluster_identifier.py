from typing import List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
class ClusterIdentifier:

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
