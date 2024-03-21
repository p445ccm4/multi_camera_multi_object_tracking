# vim: expandtab:ts=4:sw=4
import numpy as np


class Detection(object):
    """
    This class represents a bounding box detection in a single image.

    Parameters
    ----------
    xy : array_like
        World Coordinates in format `(x, y)`.
    feature : array_like
        A feature vector that describes the object contained in this image.
    bbox_area : float
        bounding box area

    Attributes
    ----------
    xy : array_like
        World Coordinates in format `(x, y)`.
    feature : array_like
        A feature vector that describes the object contained in this image.
    bbox_area : float
        bounding box area

    """

    def __init__(self, xy, feature, bbox_area):
        self.xy = np.asarray(xy, dtype=np.float)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.bbox_area = bbox_area
