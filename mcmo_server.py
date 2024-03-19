# Read in X_w, Y_w, feature from local
# TODO: Compare X_w, Y_w with existing tracks, match or create a new track
# TODO: Compare features with existing tracks if there are no match, or more than one matches
import os.path
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from scipy.spatial.distance import cdist

max_coordinates_distance = 1
max_feature_distance = 0.2
feature_comparer = NearestNeighborDistanceMetric("cosine", max_feature_distance, budget=100)
global_id = []
global_world_coordinates_npy = np.array([])
global_features_npy = np.array([])
global_bbox_area_npy = np.array([])
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
    if not os.path.exists(f'data/{t}_world_coordinates.npy'):
        break
    # Load the numpy array from the file
    world_coordinates_npy = np.load(f'data/{t}_world_coordinates.npy').reshape(-1, 3)
    world_coordinates_npy *= 0.001
    features_npy = np.load(f'data/{t}_features.npy')
    features_npy = features_npy.reshape(-1, 512)
    bbox_areas_npy = np.load(f'data/{t}_bbox_areas.npy')
    print(f'{t}: {world_coordinates_npy}')
    print(f'{t}: {global_world_coordinates_npy}')
    t += 1

    # compare world coordinates Euclidean distance
    if len(global_world_coordinates_npy) == 0:
        matches, unmatched_tracks, unmatched_detections = [], [], [i for i in range(len(world_coordinates_npy))]
    elif len(world_coordinates_npy) == 0:
        matches, unmatched_tracks, unmatched_detections = [], [i for i in range(len(global_world_coordinates_npy))], []
    else:
        # # Generate some random old and new coordinates
        # global_world_coordinates_npy = np.array([[0, 0],
        #                                          [100, 100],
        #                                          [200, 200]])
        # world_coordinates_npy = np.array([[0, 0.1],
        #                                   [300, 300],
        #                                   [200, 200.5]])

        # Compute the Euclidean distance matrix.
        distance_matrix = cdist(global_world_coordinates_npy, world_coordinates_npy, 'euclidean')

        row_indices, col_indices = linear_sum_assignment(distance_matrix)
        matches, unmatched_tracks, unmatched_detections = [], [], []
        ambiguous_global_id = []
        ambiguous_local_id = np.where(np.sum(distance_matrix < 1, axis=0) >= 2)[0]
        for col in range(len(world_coordinates_npy)):
            if col not in col_indices:
                unmatched_detections.append(col)
        for row in range(len(global_world_coordinates_npy)):
            if row not in row_indices:
                unmatched_tracks.append(row)
        for row, col in zip(row_indices, col_indices):
            track_idx = row
            detection_idx = col
            if distance_matrix[row, col] > max_coordinates_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            elif detection_idx in ambiguous_local_id:
                ambiguous_global_id.append(track_idx)
            else:
                matches.append((track_idx, detection_idx))

        # TODO: compare feature cosine distance
        cost_matrix = feature_comparer.distance(features_npy[ambiguous_local_id], ambiguous_global_id)
        cost_matrix[cost_matrix > max_feature_distance] = max_feature_distance + 1e-5
        cost_matrix = np.nan_to_num(cost_matrix)
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        amb_matches, amb_unmatched_tracks, amb_unmatched_detections = [], [], []
        for col, detection_idx in enumerate(ambiguous_local_id):
            if col not in col_indices:
                amb_unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(ambiguous_global_id):
            if row not in row_indices:
                amb_unmatched_tracks.append(track_idx)
        for row, col in zip(row_indices, col_indices):
            track_idx = ambiguous_global_id[row]
            detection_idx = ambiguous_local_id[col]
            if cost_matrix[row, col] > max_feature_distance:
                amb_unmatched_tracks.append(track_idx)
                amb_unmatched_detections.append(detection_idx)
            else:
                amb_matches.append((track_idx, detection_idx))

        matches += amb_matches
        unmatched_tracks += amb_unmatched_tracks
        unmatched_detections += amb_unmatched_detections

    active_targets = []
    # update global variables
    for (temp_global_id, temp_local_id) in matches:
        feature = features_npy[temp_local_id] / np.linalg.norm(features_npy[temp_local_id])
        smooth_feat = 0.9 * global_features_npy[temp_global_id] + (1 - 0.9) * feature
        smooth_feat /= np.linalg.norm(smooth_feat)
        global_features_npy[temp_global_id] = smooth_feat
        global_world_coordinates_npy[temp_global_id] = world_coordinates_npy[temp_local_id]
        active_targets.append(temp_global_id)
    for temp_local_id in unmatched_detections:
        temp_global_id = len(global_id)
        global_id.append(temp_global_id)
        global_features_npy = np.append(global_features_npy, features_npy[temp_local_id]).reshape(-1, 512)
        global_world_coordinates_npy = np.append(global_world_coordinates_npy, world_coordinates_npy[temp_local_id]).reshape(-1, 3)
        global_bbox_area_npy = np.append(global_bbox_area_npy, bbox_areas_npy[temp_local_id])
        active_targets.append(temp_global_id)
    # for temp_global_id in unmatched_tracks:
    #     global_id.remove(temp_global_id)
    #     global_features.pop(temp_global_id)

    feature_comparer.partial_fit(global_features_npy, global_id, active_targets)

    # removing the older graph
    graph.remove()
    ax.clear()

    # Set the x-axis range
    ax.set_xlim(-10, 10)
    # Set the y-axis range
    ax.set_ylim(-10, 10)
    # plotting newer graph
    xs, ys, IDs = [], [], []
    for ID in active_targets:
        IDs.append(ID)
        xs.append(global_world_coordinates_npy[ID][0])
        ys.append(global_world_coordinates_npy[ID][1])
    # Label the points
    for x, y, ID in zip(xs, ys, IDs):
        ax.annotate(ID, xy=(x, y))
    graph = ax.scatter(xs, ys, c='g')

    # calling pause function for 0.25 seconds
    plt.pause(0.25)
