from shapely.geometry import Point
from sklearn.metrics.pairwise import haversine_distances
from math import radians

_EARTH_RADIUS_KM = 6371


def compute_geodesic_distance(
    p1: Point, p2: Point, radius: float = _EARTH_RADIUS_KM
) -> float:
    """
    Compute the geodesic (great-circle) distance between two points on the round 2-sphere.

    Parameters:
    p1 (Point): The first point---encoded as a shapely point---storing the respective longitude (x) and lattitude (y) coordinates.

    p1 (Point): The second point---encoded as a shapely point---storing the respective longitude (x) and lattitude (y) coordinates.

    radius (float): The radius of the round 2-sphere.

    Returns:
    float: The geodesic distance between p1 and p2.
    """
    p1_radians = list(map(radians, [p1.y, p1.x]))
    p2_radians = list(map(radians, [p2.y, p2.x]))
    return radius * haversine_distances([p1_radians, p2_radians])[0, 1]
