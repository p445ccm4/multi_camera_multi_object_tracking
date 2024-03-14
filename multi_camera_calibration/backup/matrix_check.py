import numpy as np
import cv2

# Load the rotation vector and translation vector of the reference camera in the world coordinate system
rvec_world_o = np.load("rvec2_new_origin.npy").reshape(-1, 1)
tvec_world_o = np.load("tvec2_new_origin.npy").reshape(-1, 1)

# Convert the rotation vector to the rotation matrix
R_world_o, _ = cv2.Rodrigues(rvec_world_o)

# Create the transformation matrix of the reference camera
T_world_ref = np.hstack((R_world_o, tvec_world_o))
T_world_ref = np.vstack((T_world_ref, [0, 0, 0, 1]))

world = np.array([0, 0, 0, 1])
# Compute the transformation matrix of the second camera in the world coordinate system
world_sec = world @ T__ref