# Read in X_w, Y_w, feature from local
# Do tracking by comparing X_w, Y_w and feature
# Fuse inputs from multiple cameras on the same target
import argparse

import eventlet
import socketio
from matplotlib import pyplot as plt

from deep_sort.sort.detection import Detection
from deep_sort.sort.nn_matching import NearestNeighborDistanceMetric
from deep_sort.sort.tracker import Tracker

max_coordinates_distance = 1
max_feature_distance = 0.2
metric = NearestNeighborDistanceMetric("cosine", max_feature_distance, budget=100)
tracker = Tracker(metric, max_euclidean_distance=2, max_age=60, n_init=3)

# Create the figure and axes
fig, ax = plt.subplots()
# Set the x-axis range
ax.set_xlim(-5, 5)
# Set the y-axis range
ax.set_ylim(-5, 5)
plt.ion()
# plotting the first frame
graph = ax.scatter([], [])
plt.pause(0.25)

# Create a SocketIO server
sio = socketio.Server()
app = socketio.WSGIApp(sio)

all_detections = {}
@sio.event
def update(sid, world_coordinates, features, bbox_areas):
    """
    get data from cameras one by one, then parse into the tracker for updates
    Args:
        sid:
        world_coordinates:
        features:
        bbox_areas:

    Returns:

    """

    # Get the objects that were sent in the emit() call
    detections = [Detection(xy, feature, bbox_area) for xy, feature, bbox_area in zip(world_coordinates, features, bbox_areas)]
    all_detections[sid] = detections

    if len(all_detections) == n_cam:
        tracker.predict()
        tracker.update(all_detections)

        global graph
        # removing the older graph
        graph.remove()
        ax.clear()
        # Set the x-axis range
        ax.set_xlim(-5, 5)
        # Set the y-axis range
        ax.set_ylim(-5, 5)

        # plotting newer graph
        xs, ys, IDs = [], [], []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            IDs.append(track.track_id)
            xs.append(track.mean[0])
            ys.append(-track.mean[1])
        # Label the points
        for x, y, ID in zip(xs, ys, IDs):
            ax.annotate(ID, xy=(x, y))
        graph = ax.scatter(xs, ys, c='g')

        # calling pause function for 0.25 seconds
        plt.pause(1e-9)
        all_detections.clear()


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCMO server")
    parser.add_argument("n_cam", type=int, help="Number of Cameras")
    args = parser.parse_args()
    n_cam = args.n_cam

    # Start the SocketIO server
    eventlet.wsgi.server(eventlet.listen(("127.0.0.1", 5000)), app)
