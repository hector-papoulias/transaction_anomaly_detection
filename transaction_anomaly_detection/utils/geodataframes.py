from typing import List, Optional
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def get_ls_geometry(df: pd.DataFrame) -> List[Point]:
    try:
        return [Point(xy) for xy in zip(df["long"], df["lat"])]
    except KeyError:
        raise KeyError(
            "The dataframe must contain lattitude and longitude features in columns labelled 'lat' and 'long' respectively."
        )


def upgrade_df_to_gdf(df: pd.DataFrame, crs: Optional[str] = None) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(df, geometry=get_ls_geometry(df=df), crs=crs)


def downgrade_gdf_to_df(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    return gdf.loc[:, gdf.columns != "geometry"]


def geospatial_dropna(gdf: gpd.GeoDataFrame):
    return gdf[gdf["geometry"].is_valid]
