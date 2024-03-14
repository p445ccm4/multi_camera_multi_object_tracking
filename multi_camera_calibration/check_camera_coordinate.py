# Import required modules 
import cv2 
import numpy as np 
import os 
import glob  
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description="check single camera accruary")
parser.add_argument("internal_matrix", help="numpy array of internal matrix")
parser.add_argument("rotation_vector",  help="numpy array of rotation vector")
parser.add_argument("translation_vector", help="numpy array of translation vector")
parser.add_argument("distortion", help="distortion vector")
parser.add_argument("relative_rotation_vector",  help="numpy array of relative rotation vector")
parser.add_argument("relative_translation_vector", help="numpy array of relative translation vector")
parser.add_argument("image", type=str, help="Path to the images from Camera that contine the origin")
args = parser.parse_args()

# Load the saved array
matrix = np.load(args.internal_matrix)
r_vec = np.load(args.rotation_vector)
t_vec = np.load(args.translation_vector)
distortion = np.load(args.distortion)
new_r_r_vec = np.load(args.relative_rotation_vector)
new_r_t_vec = np.load(args.relative_translation_vector)

def world_to_image(world_coord, rvec, tvec, camera_matrix, dist_coeffs):
    world_coord = np.array([world_coord], dtype=np.float32)
    image_points, _ = cv2.projectPoints(world_coord, rvec, tvec, camera_matrix, dist_coeffs)
    return tuple(map(int, image_points[0][0]))

def image_to_world(i, M, r_vec, T):
    """
    Convert image coordinate (u, v) to 2D world coordinate (X, Y) using camera calibration parameters,
    assuming Z = 0.

    :param i: Image coordinate [u,v] (2x1)
    :param M: Camera intrinsic matrix (3x3)
    :param D: Distortion coefficients (1x5 or 1x4)
    :param r_vec: Rotation vector (3x1)
    :param T: Translation vector (3x1)
    :return w: World coordinate (X, Y)
    """
    i_vector = np.append(i,1).reshape(3, 1)		
    M_inv = np.linalg.inv(M)
    rot_mat, _ = cv2.Rodrigues(r_vec) # Convert rotation vector to rotation matrix
    rot_mat_inv = np.linalg.inv(rot_mat)
    left_side = np.matmul(rot_mat_inv, np.matmul(M_inv,i_vector)) #(3x1)
    right_side = np.matmul(rot_mat_inv , T) #(3x1)
    s = (0 + right_side[2, 0]) / left_side[2, 0] #0 mean the Z of the world coordinate locate in the same plane with the orgin
    # w = (s * left_side - right_side) # unit is how many box(need to multiply by the length of the box)
    w = np.matmul(rot_mat_inv, (s * np.matmul(M_inv, i_vector)) - T) # unit is how many box(need to multiply by the length of the box)
    return w

def draw_axes(img, M, r_vec, T, length=150):
    """
    Draw XYZ axes on the image.

    :param img: Input image
    :param M: Camera intrinsic matrix (3x3)
    :param r_vec: Rotation vector (3x1)
    :param T: Translation vector (3x1)
    :param length: Length of the axes (default: 1( 2.5cm ))
    """
    # Define the axes' endpoints in world coordinates
    origin = np.array([0, 0, 0, 1]).reshape(4, 1)
    x_axis = np.array([length, 0, 0, 1]).reshape(4, 1)
    y_axis = np.array([0, length, 0, 1]).reshape(4, 1)
    z_axis = np.array([0, 0, length, 1]).reshape(4, 1)

    # Transform world coordinates to image coordinates
    rot_mat, _ = cv2.Rodrigues(r_vec)
    transform = np.column_stack((rot_mat, T))
    img_origin = np.dot(M, np.dot(transform, origin))
    img_x_axis = np.dot(M, np.dot(transform, x_axis))
    img_y_axis = np.dot(M, np.dot(transform, y_axis))
    img_z_axis = np.dot(M, np.dot(transform, z_axis))

    # Normalize the coordinates
    img_origin = (img_origin / img_origin[2]).astype(int)
    img_x_axis = (img_x_axis / img_x_axis[2]).astype(int)
    img_y_axis = (img_y_axis / img_y_axis[2]).astype(int)
    img_z_axis = (img_z_axis / img_z_axis[2]).astype(int)

    # Draw the axes
    img = cv2.line(img, tuple(img_origin[:2].ravel()), tuple(img_z_axis[:2].ravel()), (0, 0, 255), 2)  # Z - blue
    cv2.putText(img, 'Z', tuple(img_z_axis[:2].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    img = cv2.line(img, tuple(img_origin[:2].ravel()), tuple(img_x_axis[:2].ravel()), (255, 0, 0), 2)  # X - red
    cv2.putText(img, 'X', tuple(img_x_axis[:2].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    img = cv2.line(img, tuple(img_origin[:2].ravel()), tuple(img_y_axis[:2].ravel()), (0, 255, 0), 2)  # Y - green
    cv2.putText(img, 'Y', tuple(img_y_axis[:2].ravel()), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img



def world_to_camera_coordinate(point_world, rvec_cam1, tvec_cam1):
    # Convert the rotation vector to a rotation matrix
    R_cam1, _ = cv2.Rodrigues(rvec_cam1)

    # Create the transformation matrix of cam1
    T_world_to_cam = np.hstack((R_cam1, tvec_cam1))
    T_world_to_cam = np.vstack((T_world_to_cam, [0, 0, 0, 1]))
    # print(T_world_to_cam)

    # Convert the world coordinate of the point to homogeneous coordinates
    point_world_homogeneous = np.append(point_world, 1)

    # Compute the camera coordinate of the point
    point_cam_homogeneous = T_world_to_cam @ point_world_homogeneous
    point_cam = point_cam_homogeneous[:3]

    return point_cam

def cam1_to_cam2_camera_coordinate(cam1_point, rvec_rel, tvec_rel):
    # Convert the rotation vector to a rotation matrix
    R_cam1, _ = cv2.Rodrigues(rvec_rel)

    # Create the transformation matrix of cam1
    T_cam1_to_cam2 = np.hstack((R_cam1, tvec_rel))
    T_cam1_to_cam2 = np.vstack((T_cam1_to_cam2, [0, 0, 0, 1]))
    # print(T_world_to_cam)

    # Convert the world coordinate of the point to homogeneous coordinates
    point_cam1_homogeneous = np.append(cam1_point, 1)

    # Compute the camera coordinate of the point
    point_cam_homogeneous = T_cam1_to_cam2 @ point_cam1_homogeneous
    cam2_point = point_cam_homogeneous[:3]

    return cam2_point

# Function to handle mouse events
def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, img 

    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        img = original_img_undist_add_origin.copy()
        input_img_coords = np.array((mouseX, mouseY), dtype=np.float32)  # Example input image coordinates
        world_coords = image_to_world(input_img_coords, matrix, r_vec, t_vec)
        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        cv2.putText(img, f'{(str(world_coords[0])),(str(world_coords[1]))}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        camera_coords = world_to_camera_coordinate(world_coords, r_vec, t_vec)
        cam2_camera_coords = cam1_to_cam2_camera_coordinate(camera_coords, new_r_r_vec, new_r_t_vec)
        print("camera2_coords: ", cam2_camera_coords)


if __name__ == "__main__": 
    # Read the image
    original_img = cv2.imread(args.image)
    original_img_undist = cv2.undistort(original_img, matrix, distortion)
    original_img_undist_add_origin = draw_axes(original_img_undist, matrix, r_vec, t_vec)
    img = original_img_undist_add_origin.copy()

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
