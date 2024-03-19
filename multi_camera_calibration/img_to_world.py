import os.path

import cv2
import numpy as np


class ImgToWorld(object):
    def __init__(self, cam_id):
        cam_param_path = f'cam_param{cam_id}'
        self.cam_id = cam_id
        # Load the saved array
        self.matrix = np.load(os.path.join(cam_param_path, 'internal_matrix.npy'))
        self.r_vec = np.load(os.path.join(cam_param_path, 'rotation_vector.npy'))
        self.t_vec = np.load(os.path.join(cam_param_path, 'translation_vector.npy'))
        self.distortion = np.load(os.path.join(cam_param_path, 'distortion.npy'))

    def image_to_world(self, i):
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
        M = self.matrix
        r_vec = self.r_vec
        T = self.t_vec

        i_vector = np.append(i, 1).reshape(3, 1)
        M_inv = np.linalg.inv(M)
        rot_mat, _ = cv2.Rodrigues(r_vec)  # Convert rotation vector to rotation matrix
        rot_mat_inv = np.linalg.inv(rot_mat)
        left_side = np.matmul(rot_mat_inv, np.matmul(M_inv, i_vector))  # (3x1)
        right_side = np.matmul(rot_mat_inv, T)  # (3x1)
        s = (0 + right_side[2, 0]) / left_side[
            2, 0]  # 0 mean the Z of the world coordinate locate in the same plane with the orgin
        # w = (s * left_side - right_side) # unit is how many box(need to multiply by the length of the box)
        w = np.matmul(rot_mat_inv, (s * np.matmul(M_inv,
                                                  i_vector)) - T)  # unit is how many box(need to multiply by the length of the box)
        return w

    def draw_axes(self, img, length=250):
        """
        Draw XYZ axes on the image.

        :param img: Input image
        :param M: Camera intrinsic matrix (3x3)
        :param r_vec: Rotation vector (3x1)
        :param T: Translation vector (3x1)
        :param length: Length of the axes (default: 1( 2.5cm ))
        """
        M = self.matrix
        r_vec = self.r_vec
        T = self.t_vec

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

    def get_world_coordinates(self, outputs):
        """
        Args:
            outputs: nx5 numpy array, each row contains (x1, y1, x2, y2, id)

        Returns:
            world_coordinates: nx2 numpy array, each row contains (X_w, Y_w)
        """

        world_coordinates = []
        for x1, y1, x2, y2 in outputs[:, :4]:
            image_x = (x1 + x2) / 2
            image_y = max(y1, y2)
            world_coordinate = self.image_to_world([image_x, image_y])
            world_coordinates.append(world_coordinate)
        if len(world_coordinates) > 0:
            world_coordinates = np.stack(world_coordinates, axis=0)

        return world_coordinates
