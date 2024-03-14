import cv2
import numpy as np
import glob
import argparse
import re

# Parse command line arguments
parser = argparse.ArgumentParser(description="Stereo calibration using raw image pairs")
parser.add_argument("camera1_folder", type=str, help="Path to the folder containing images from Camera 1 (e.g., 'camera1_images/*.jpg')")
parser.add_argument("camera2_folder", type=str, help="Path to the folder containing images from Camera 2 (e.g., 'camera2_images/*.jpg')")
parser.add_argument("camera1_internal_matrix", help="numpy array of camera1 internal matrix")
parser.add_argument("camera2_internal_matrix", help="numpy array of camera2 internal matrix")
parser.add_argument("camera1_distortion", help="camera1_distortion vector")
parser.add_argument("camera2_distortion", help="camera2_distortion vector")
parser.add_argument("pattern_width", type=int, help="Number of corners in the calibration pattern along the width")
parser.add_argument("pattern_height", type=int, help="Number of corners in the calibration pattern along the height")
parser.add_argument("square_size", type=float, help="Size of a side of a square in the chessboard (in any unit, e.g., mm)")
args = parser.parse_args()

#load the numpy arrays
camera_matrix1 = np.load(args.camera1_internal_matrix)
camera_matrix2 = np.load(args.camera2_internal_matrix)
dist_coeffs1 = np.load(args.camera1_distortion)
dist_coeffs2 = np.load(args.camera2_distortion)

# Parameters for the calibration pattern (chessboard)
pattern_size = (args.pattern_width, args.pattern_height)
square_size = args.square_size

# Prepare the object points, like (0,0,0), (1,0,0), (2,0,0), ..., (6,5,0)
objp = np.zeros((np.prod(pattern_size), 3), np.float32)
objp[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points from all images
objpoints = []  # 3d points in real world space
imgpoints1 = []  # 2d points in image plane for camera 1
imgpoints2 = []  # 2d points in image plane for camera 2

# List of images for Camera 1 and Camera 2
camera1_images = glob.glob(args.camera1_folder + "/" + '*.jpg')
camera2_images = glob.glob(args.camera2_folder + "/" + '*.jpg')
camera1_images.sort(key=lambda f: int(re.sub('\D', '', f)))
camera2_images.sort(key=lambda f: int(re.sub('\D', '', f)))
print(camera1_images)
print(camera2_images)

# Ensure the same number of images for both cameras
assert len(camera1_images) == len(camera2_images), "Camera 1 and Camera 2 should have the same number of images"

for fname1, fname2 in zip(camera1_images, camera2_images):
    img1 = cv2.imread(fname1)
    img2 = cv2.imread(fname2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret1, corners1 = cv2.findChessboardCorners(gray1, pattern_size, None)
    ret2, corners2 = cv2.findChessboardCorners(gray2, pattern_size, None)

    # If found, add object points, image points
    if ret1 and ret2:
        objpoints.append(objp)
        imgpoints1.append(corners1)
        imgpoints2.append(corners2)

# # Calibrate each camera individually
# ret1, camera_matrix1, dist_coeffs1, rvecs1, tvecs1 = cv2.calibrateCamera(objpoints, imgpoints1, gray1.shape[::-1], None, None)
# ret2, camera_matrix2, dist_coeffs2, rvecs2, tvecs2 = cv2.calibrateCamera(objpoints, imgpoints2, gray2.shape[::-1], None, None)


# # print("Camera 1 intrinsic parameters:\n", camera_matrix1)
# print("dist_coeffs1:\n", dist_coeffs1)
# print("\n")
# # print("Camera 2 intrinsic parameters:\n", camera_matrix2)
# print("dist_coeffs2:\n", dist_coeffs2)
# print("\n")

# Stereo calibration
ret, M1, d1, M2, d2, R, T, E, F  = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2,
    camera_matrix1, dist_coeffs1,
    camera_matrix2, dist_coeffs2,
    gray1.shape[::-1], None, None, None, None,
    cv2.CALIB_FIX_INTRINSIC, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-5)
)

# print("Camera 1 intrinsic parameters:\n", M1)
# print("dist_coeffs1:\n", d1)
# print("\n")
# print("dist_coeffs2:\n", d2)
# print("Camera 2 intrinsic parameters:\n", M2)
print("Rotation Matrix (R):\n", R)
revc, _ = cv2.Rodrigues(R)
print("Rotation vector (R):\n", revc)
print("Translation Vector (T):\n", T)

# # savwe the numpy arrays
np.save("rvec_relative.npy", revc)
np.save("tvec_relative.npy", T)
