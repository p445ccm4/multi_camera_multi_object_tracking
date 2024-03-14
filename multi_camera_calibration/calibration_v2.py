# Import required modules 
import cv2 
import numpy as np 
import os 
import glob 
import re
import argparse



if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="single camera calibration using raw image")
    parser.add_argument("image_folder", type=str, help="Path to the folder containing images from Camera")
    parser.add_argument("pattern_width", type=int, help="Number of corners in the calibration pattern along the width")
    parser.add_argument("pattern_height", type=int, help="Number of corners in the calibration pattern along the height")
    parser.add_argument("square_size", type=float, help="Size of a side of a square in the chessboard (in any unit, e.g., mm)")
    args = parser.parse_args()

    # Define the dimensions of checkerboard 
    CHECKERBOARD = (args.pattern_width, args.pattern_height) 
    square_size = args.square_size


    # stop the iteration when specified 
    # accuracy, epsilon, is reached or 
    # specified number of iterations are completed. 
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) 


    # Vector for 3D points 
    threedpoints = [] 

    # Vector for 2D points 
    twodpoints = [] 


    # 3D points real world coordinates 
    objectp3d = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32) 
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) 
    objectp3d = objectp3d * square_size
    prev_img_shape = None


    # Load calibration images from the folder
    # folder_path = 
    image_files = glob.glob(args.image_folder + "/" + '*.jpg')
    image_files.sort(key=lambda f: int(re.sub('\D', '', f)))
    print(image_files)

    for filename in image_files: 
        image = cv2.imread(filename) 
        grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

        # Find the chess board corners 
        # If desired number of corners are 
        # found in the image then ret = true 
        ret, corners = cv2.findChessboardCorners( 
                        grayColor, CHECKERBOARD, 
                        cv2.CALIB_CB_ADAPTIVE_THRESH 
                        + cv2.CALIB_CB_FAST_CHECK +
                        cv2.CALIB_CB_NORMALIZE_IMAGE) 

        # If desired number of corners can be detected then, 
        # refine the pixel coordinates and display 
        # them on the images of checker board 
        if ret == True: 
            threedpoints.append(objectp3d) 

            # Refining pixel coordinates 
            # for given 2d points. 
            corners2 = cv2.cornerSubPix( 
                grayColor, corners, (11, 11), (-1, -1), criteria) 

            twodpoints.append(corners2) 

            # Draw and display the corners 
            image = cv2.drawChessboardCorners(image, 
                                            CHECKERBOARD, 
                                            corners2, ret) 

        cv2.imshow('img', image) 
        cv2.waitKey(0) 

    cv2.destroyAllWindows() 

    h, w = image.shape[:2] 


    # Calibrate the camera
    ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera( 
        threedpoints, twodpoints, grayColor.shape[::-1], None, None) 


    # Displaying required output 
    print(" Camera matrix:") 
    print(matrix)

    print("\n Distortion coefficient:") 
    print(distortion) 

    print("\n Rotation Vectors:") 
    print(r_vecs) 

    # 把旋转向量转换为旋转矩阵
    print("\n Rotation Matrix of image0:") 
    rot_mat, _ = cv2.Rodrigues(r_vecs[0])
    print(f"rot_mat:\n {rot_mat}")

    print("\n Translation Vectors:") 
    # print(t_vecs) 

    print("\n r_vecs:") 
    print(r_vecs) 
    # print(len(r_vecs))


    print("\n t_vecs:") 
    print(t_vecs) 
    # print(len(t_vecs))  
    print("\n")
    # # Project a image point [u,v] into the real-world coordinate system
    # ToWorldPoint = [800, 400]
    # input_img_coords = np.array(ToWorldPoint, dtype=np.float32)  # Example input image coordinates
    # world_coords = image_to_world(input_img_coords, matrix, r_vecs[0], t_vecs[0])


    # # Project the world origin (0, 0, 0) back onto the image
    # world_origin = np.array([0, 0, 0], dtype=np.float32)
    # image_origin = world_to_image(world_origin, r_vecs[0], t_vecs[0], matrix, distortion)

    # # Load the image
    # img = cv2.imread(image_files[0])
    
    # #h,  w = img.shape[:2]
    
    # #newcameramtx, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w,h), 1, (w,h))

    # # Undistort the image
    # img = cv2.undistort(img, matrix, distortion)

    # # Draw the axes
    # image_with_axes = draw_axes(img,matrix, r_vecs[0], t_vecs[0], 500)

    # # Draw a circle at the image origin with a radius of 5 and color (0, 0, 255) (red)
    # cv2.circle(img, image_origin, 3, (0, 0, 255), -1)

    # # # Draw a circle at ToWorldPoint with a radius of 5 and color (0, 255, 0) (blue)
    # # cv2.circle(img, ToWorldPoint, 3, (255, 0, 0), -1)

    # # Save the modified image
    # # cv2.imwrite(f'{ToWorldPoint}.jpg', img)
    # cv2.imwrite('image_origin.jpg', img)


    # Save the array to a binary file
    np.save(os.path.join(args.image_folder, 'internal_matrix.npy'), matrix)
    np.save(os.path.join(args.image_folder, 'rotation_vector.npy'),r_vecs[0])
    np.save(os.path.join(args.image_folder, 'translation_vector.npy'), t_vecs[0])
    np.save(os.path.join(args.image_folder, 'distortion'), distortion)
