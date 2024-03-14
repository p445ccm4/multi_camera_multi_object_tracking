import numpy as np
import cv2
import sys

# Input file paths from the terminal
rotation_vec_file = sys.argv[1]
translation_vec_file = sys.argv[2]
relative_rot_vec_file = sys.argv[3]
relative_trans_file = sys.argv[4]

# Load the rotation vector and translation vector of the reference camera in the world coordinate system
rvec_world_ref = np.load(rotation_vec_file).reshape(-1, 1)
tvec_world_ref = np.load(translation_vec_file).reshape(-1, 1)

# Convert the rotation vector to the rotation matrix
R_world_ref, _ = cv2.Rodrigues(rvec_world_ref)

# Create the transformation matrix of the reference camera
T_world_ref = np.hstack((R_world_ref, tvec_world_ref))
T_world_ref = np.vstack((T_world_ref, [0, 0, 0, 1]))



# Load the relative rotation vector `rvec_rel` and translation vector `T` from `cv2.stereoCalibrate`
rvec_rel = np.load(relative_rot_vec_file).reshape(-1, 1)
T_rel = np.load(relative_trans_file).reshape(-1, 1)

# Convert the relative rotation vector to the relative rotation matrix
R_rel, _ = cv2.Rodrigues(rvec_rel)

# Create the transformation matrix of the second camera relative to the reference camera
T_rel_sec = np.hstack((R_rel, T_rel))
T_rel_sec = np.vstack((T_rel_sec, [0, 0, 0, 1]))

# Compute the transformation matrix of the second camera refer to cam1 camera coordinate system
T_world_sec = T_rel_sec @ T_world_ref

print("transfer matrix cam2 camera coords to cam1 world coordinate system: \n", T_world_sec)
np.save('T_sec2world', T_world_sec)

# # Extract the rotation matrix and translation vector of the second camera in the world coordinate system
# R_world_sec = T_world_sec[:3, :3]
# R_world_sec_vec, _ = cv2.Rodrigues(R_world_sec)
# tvec_world_sec = T_world_sec[:3, 3].reshape(-1, 1)

# print("Rotation matrix of the second camera in the world coordinate system:", R_world_sec_vec)
# print("Translation vector of the second camera in the world coordinate system:", tvec_world_sec)
# np.save('rvec2_new_origin.npy', R_world_sec_vec)
# np.save('tvec2_new_origin.npy', tvec_world_sec)

