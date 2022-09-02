# Code from https://gist.github.com/dwyerk/10561690

from shapely.ops import unary_union, polygonize
import shapely.geometry as geometry
from scipy.spatial import Delaunay
import numpy as np


def alpha_shape(points, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    @param points: numpy array of points.
    @param alpha: alpha value to influence the
        gooeyness of the border. Smaller numbers
        don't fall inward as much as larger numbers.
        Too large, and you lose everything!
    """
    if len(points) < 4:
        # When you have a triangle, there is no sense
        # in computing an alpha shape.
        return geometry.MultiPoint(list(points)).convex_hull

    # coords = np.array([point.coords[0] for point in points])
    # tri = Delaunay(coords)
    # triangles = coords[tri.vertices]
    tri = Delaunay(points)
    triangles = points[tri.vertices]
    a = np.sqrt((triangles[:, 0, 0] - triangles[:, 1, 0]) ** 2
                + (triangles[:, 0, 1] - triangles[:, 1, 1]) ** 2)
    b = np.sqrt((triangles[:, 1, 0] - triangles[:, 2, 0]) ** 2
                + (triangles[:, 1, 1] - triangles[:, 2, 1]) ** 2)
    c = np.sqrt((triangles[:, 2, 0] - triangles[:, 0, 0]) ** 2
                + (triangles[:, 2, 1] - triangles[:, 0, 1]) ** 2)
    s = (a + b + c) / 2.0
    areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
    circums = a * b * c / (4.0 * areas)
    filtered = triangles[circums < (1.0 / alpha)]
    edge1 = filtered[:, (0, 1)]
    edge2 = filtered[:, (1, 2)]
    edge3 = filtered[:, (2, 0)]
    edge_points = (np.unique(
                        np.concatenate((edge1, edge2, edge3)), axis=0)
                   .tolist())
    m = geometry.MultiLineString(edge_points)
    triangles = list(polygonize(m))
    return unary_union(triangles), edge_points
