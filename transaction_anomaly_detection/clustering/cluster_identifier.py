from typing import List, Tuple, Optional
from collections import defaultdict
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.neighbors import BallTree
class ClusterIdentifier:
    @staticmethod
    def _sr_geometry_to_arr_coords(sr_geometry: gpd.GeoSeries) -> np.array:
        return np.column_stack((sr_geometry.x, sr_geometry.y))
