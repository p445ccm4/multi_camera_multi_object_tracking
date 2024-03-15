# TODO: Read in X_w, Y_w, feature from local
# TODO: Compare them with existing tracks, match or create a new track
import os.path

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

from deep_sort.sort.nn_matching import NearestNeighborDistanceMetric

# TODO
class MCMOTrack(object):
    def __init__(self, X, Y, features, cam_id, local_id):
        self.X = X
        self.Y = Y
        self.features = features
        self.local_ids = {cam_id: local_id}

# TODO
class MCMOServer(object):
    def __init__(self):
        pass


max_distance = 0.2
metric = NearestNeighborDistanceMetric("cosine", max_distance, budget=100)
global_id = []
global_features = {}
t = 0

# Create the figure and axes
fig, ax = plt.subplots()
# Set the x-axis range
ax.set_xlim(-10, 10)
# Set the y-axis range
ax.set_ylim(-10, 10)
plt.ion()
# plotting the first frame
graph = ax.scatter([], [])
plt.pause(0.25)

while True:
    if not os.path.exists(f'data/{t}_identities.npy'):
        break
    # Load the numpy array from the file
    local_id = np.load(f'data/{t}_identities.npy')
    world_coordinates_npy = np.load(f'data/{t}_world_coordinates.npy')
    world_coordinates_npy *= 0.001
    world_coordinates = dict(zip(local_id, world_coordinates_npy))
    features_npy = np.load(f'data/{t}_features.npy')
    features_npy = features_npy.reshape(-1, 512)
    features = dict(zip(local_id, features_npy))
    print(f'{t}: {world_coordinates}')
    print(f'{t}: {global_features.keys()}')
    t += 1

    cost_matrix = metric.distance(features_npy, global_id)
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5
    cost_matrix = np.nan_to_num(cost_matrix)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches, unmatched_tracks, unmatched_detections = [], [], []
    for col, detection_idx in enumerate(local_id):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)
    for row, track_idx in enumerate(global_id):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)
    for row, col in zip(row_indices, col_indices):
        track_idx = global_id[row]
        detection_idx = local_id[col]
        if cost_matrix[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    matches, unmatched_tracks, unmatched_detections = matches, unmatched_tracks, unmatched_detections

    for (temp_global_id, temp_local_id) in matches:
        feature = features[temp_local_id] / np.linalg.norm(features[temp_local_id])
        smooth_feat = 0.9 * global_features[temp_global_id] + (1 - 0.9) * feature
        smooth_feat /= np.linalg.norm(smooth_feat)
        global_features[temp_global_id] = smooth_feat
        # active_targets.append(temp_global_id)
    for temp_local_id in unmatched_detections:
        temp_global_id = len(global_id)
        global_id.append(temp_global_id)
        global_features[temp_global_id] = features[temp_local_id]
        # active_targets.append(temp_global_id)
    # for temp_global_id in unmatched_tracks:
    #     global_id.remove(temp_global_id)
    #     global_features.pop(temp_global_id)

    metric.partial_fit(global_features.values(), global_features.keys(), global_features.keys())

    # removing the older graph
    graph.remove()
    ax.clear()

    # Set the x-axis range
    ax.set_xlim(-10, 10)
    # Set the y-axis range
    ax.set_ylim(-10, 10)
    # plotting newer graph
    xs, ys, IDs = [], [], []
    for ID, value in world_coordinates.items():
        IDs.append(ID)
        xs.append(value[0])
        ys.append(value[1])
    # Label the points
    for x, y, ID in zip(xs, ys, IDs):
        ax.annotate(ID, xy=(x, y))
    graph = ax.scatter(xs, ys, c='g')

    # calling pause function for 0.25 seconds
    plt.pause(0.25)
