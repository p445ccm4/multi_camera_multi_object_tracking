import numpy as np
from scipy.spatial.distance import cdist

def euclidean_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    Args:
        tracks : List[track.Track]
            A list of predicted tracks at the current time step.
        detections : List[detection.Detection]
            A list of detections at the current time step.
        track_indices : List[int]
            List of track indices that maps rows in `cost_matrix` to tracks in
            `tracks` (see description above).
        detection_indices : List[int]
            List of detection indices that maps columns in `cost_matrix` to
            detections in `detections` (see description above).

    Returns:
        cost_matrix : NxM dimensional cost matrix, where element (i, j) is the
        association cost between the i-th track in the given track indices and
        the j-th detection in the given detection_indices.
    """
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))

    track_xy = []
    for ID in track_indices:
        track_xy.append(tracks[ID].mean[:2])
    track_xy = np.stack(track_xy)
    detection_xy = []
    for ID in detection_indices:
        detection_xy.append(detections[ID].xy)
    detection_xy = np.stack(detection_xy)

    cost_matrix = cdist(track_xy, detection_xy, metric='euclidean')

    return cost_matrix
