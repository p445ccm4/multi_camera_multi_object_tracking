# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        self.metric = metric
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            # track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):

        def gated_metric(tracks, dets, track_indices, detection_indices):
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            cost_matrix = self.metric.distance(features, targets)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix

        def disappeared_in_middle(track):
            """
            Check if a track disappeared in the middle of the image.
            """
            # Assuming the image dimensions are known (img_width, img_height)
            middle_region = [480, 270, 1440, 810]
            cxcy = track.mean[:2]
            return (middle_region[0] < cxcy[0] < middle_region[2]) and (middle_region[1] < cxcy[1] < middle_region[3])

        def greedy_matching_based_on_appearance(tracks, detections, track_indices, unmatched_detections):
            # Create a cost matrix based on appearance features
            features = np.array([detections[i].feature for i in unmatched_detections])
            # print("features: ", features) #[[0.1, 0.2, 0.3, 0.4]]]
            targets = np.array([tracks[i].track_id for i in track_indices])
            # print("targets: ", targets) #[1]
            cost_matrix = self.metric.distance(features, targets)

            matches = []
            while cost_matrix.size != 0:
                # Find the index of the minimum cost
                min_index = np.unravel_index(np.argmin(cost_matrix, axis=None), cost_matrix.shape)
                match_threshold = 0.5  # Set a threshold for matching

                if cost_matrix[min_index] < match_threshold:
                    track_idx = track_indices[min_index[1]]
                    detection_idx = unmatched_detections[min_index[0]]
                    matches.append((track_idx, detection_idx))

                    # Remove matched track and detection from the pool
                    cost_matrix = np.delete(cost_matrix, min_index[0], 0)
                    cost_matrix = np.delete(cost_matrix, min_index[1], 1)
                    track_indices.remove(track_idx)
                    unmatched_detections.remove(detection_idx)
                else:
                    break  # No more good matches

            unmatched_tracks = track_indices
            return matches, unmatched_tracks, unmatched_detections

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]


        # Associate confirmed tracks using appearance features.
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        if len(confirmed_tracks)!=0:
            # New step: Greedy matching for unconfirmed tracks that disappeared in the middle
            middle_unconfirmed_tracks = [
                i for i in unconfirmed_tracks if disappeared_in_middle(self.tracks[i]) and self.tracks[i].track_id in self.metric.samples]

            # print("middle_unconfirmed_tracks: ", middle_unconfirmed_tracks)
            # print("unmatched_detections: ", unmatched_detections)

            greedy_matches, unmatched_tracks_greedy, unmatched_detections = \
                greedy_matching_based_on_appearance(
                    self.tracks, detections, middle_unconfirmed_tracks, unmatched_detections)
            matches_a += greedy_matches
            unmatched_tracks_a = list(set(unmatched_tracks_a) - set(k for k, _ in greedy_matches))

            # Update the list of unconfirmed tracks
            unconfirmed_tracks = [
                i for i in unconfirmed_tracks if i not in middle_unconfirmed_tracks]

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]
        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature, detection.confidence))
        self._next_id += 1
