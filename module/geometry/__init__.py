# large number of contents in this submodule are  modified from deepxde.geometry,
# see: https://github.com/lululxvi/deepxde/tree/master/deepxde/geometry
__all__ = [
    "CSGDifference",
    "CSGIntersection",
    "CSGUnion",
    "Cuboid",
    "Disk",
    "Geometry",
    "Hypercube",
    "Hypersphere",
    "Interval",
    "Polygon",
    "Rectangle",
    "Sphere",
    "Triangle",
    "sample",
    "FunctionBasedGeometry",
    "Ellipsoid"
    "GaussianMixture"
]

from .csg import CSGDifference, CSGIntersection, CSGUnion
from .geometry import Geometry
from .geometry_1d import Interval
from .geometry_2d import Disk, Polygon, Rectangle, Triangle
from .geometry_3d import Cuboid, Sphere
from .geometry_nd import Hypercube, Hypersphere
from .sampler import sample
from .geometry_custom import FunctionBasedGeometry,Ellipsoid,GaussianMixture