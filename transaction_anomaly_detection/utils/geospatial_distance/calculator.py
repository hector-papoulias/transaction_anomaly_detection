from enum import Enum
from shapely.geometry import Point
from transaction_anomaly_detection.utils.geospatial_distance._metrics import (
    compute_geodesic_distance,
)


class GeospatialDistance(Enum):
    GEODESIC = "geodesic"


class GeospatialDistanceCalculator:
    _DICT_DISTANCE_TO_CALLABLE = {
        GeospatialDistance.GEODESIC: compute_geodesic_distance
    }

    @classmethod
    def compute_distance(
        cls, geospatial_distance: GeospatialDistance, p1: Point, p2: Point
    ) -> float:
        """
        Compute the specified distance between two points on the surface of the Earth.

        Parameters:
        geospatial_distance (GeospatialDistance): The distance to be computed.

        p1 (Point): The first point---encoded as a shapely point---storing the respective longitude (x) and lattitude (y) coordinates.

        p1 (Point): The second point---encoded as a shapely point---storing the respective longitude (x) and lattitude (y) coordinates.

        Returns:
        float: The distance between p1 and p2.

        """
        try:
            distance_function = cls._DICT_DISTANCE_TO_CALLABLE[geospatial_distance]
        except ValueError:
            raise ValueError("The requested distance is not supported.")
        return distance_function(p1=p1, p2=p2)
