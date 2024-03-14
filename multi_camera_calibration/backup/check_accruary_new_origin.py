# Import required modules 
import cv2 
import numpy as np 
import os 
import glob 
from check_accruary import draw_axes, image_to_world, world_to_image  
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="check single camera accruary")
parser.add_argument("internal_matrix", help="numpy array of internal matrix")
parser.add_argument("rotation_vector",  help="numpy array of rotation vector")
parser.add_argument("translation_vector", help="numpy array of translation vector")
parser.add_argument("distortion", help="distortion vector")
parser.add_argument("new_origin_rotation_vector",  help="numpy array of new_origin rotation vector")
parser.add_argument("new_origin_translation_vector", help="numpy array of new_origin translation vector")
parser.add_argument("image", type=str, help="Path to the images from Camera that contine the origin")
args = parser.parse_args()

# Load the saved array
matrix = np.load(args.internal_matrix)
# print(matrix.shape)
r_vec = np.load(args.rotation_vector)
# print(r_vec.shape)
t_vec = np.load(args.translation_vector)
# print(t_vec.shape)
new_o_r_vec = np.load(args.new_origin_rotation_vector)
new_o_t_vec = np.load(args.new_origin_translation_vector)


def world_to_camera_coordinate(point_world, rvec_cam1, tvec_cam1):
    # Convert the rotation vector to a rotation matrix
    R_cam1, _ = cv2.Rodrigues(rvec_cam1)

    # Create the transformation matrix of cam1
    T_world_to_cam = np.hstack((R_cam1, tvec_cam1))
    T_world_to_cam = np.vstack((T_world_to_cam, [0, 0, 0, 1]))
    print(T_world_to_cam)

    # Convert the world coordinate of the point to homogeneous coordinates
    point_world_homogeneous = np.append(point_world, 1)

    # Compute the camera coordinate of the point
    point_cam_homogeneous = T_world_to_cam @ point_world_homogeneous
    point_cam = point_cam_homogeneous[:3]

    return point_cam

def camera_to_world_coordinate(point_cam, rvec_cam, tvec_cam):
    # Convert the rotation vector to a rotation matrix
    R_cam, _ = cv2.Rodrigues(rvec_cam)

    # Create the transformation matrix of the camera
    T_cam_to_world = np.hstack((R_cam, tvec_cam))
    T_cam_to_world = np.vstack((T_cam_to_world, [0, 0, 0, 1]))
    T_cam_to_world = np.linalg.inv(T_cam_to_world)

    # Convert the camera coordinate of the point to homogeneous coordinates
    point_cam_homogeneous = np.append(point_cam, 1)

    # Compute the world coordinate of the point
    point_world_homogeneous = T_cam_to_world @ point_cam_homogeneous
    point_world = point_world_homogeneous[:3]

    return point_world

# Function to handle mouse events
def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, img 

    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        img = original_img.copy()
        #get camera 2 image coordinate
        input_img_coords = np.array((mouseX, mouseY), dtype=np.float32)  # Example input image coordinates

        #convert to camera 2 world coordinate
        cam2_world_coords = image_to_world(input_img_coords, matrix, r_vec, t_vec)
        #cam2_world_coords[2] = 0

        #convert to camera 2 camera coordinate
        cam2_cam_coords = world_to_camera_coordinate(cam2_world_coords, r_vec, t_vec)

        #convert to camera 1 world coordinate
        cam1_world_coords = camera_to_world_coordinate(cam2_cam_coords, new_o_r_vec, new_o_t_vec)

        #draw circle and put text
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(img, f'{(str(cam1_world_coords[0])),(str(cam1_world_coords[1]))}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        

# Read the image
original_img = cv2.imread(args.image)
img = original_img.copy()

# Create a named window and bind the mouse callback function to it
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_circle)

while True:
    # Display the image
    cv2.imshow('Image', img)

    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Close all windows
cv2.destroyAllWindows()
