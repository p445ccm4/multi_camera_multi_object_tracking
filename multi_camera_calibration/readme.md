# 1. Delete all image in calibration_image folder.
   use capture.py to capture the chessboard image (>=20images) 
   press "c" to capture common image, image will save as calibration{i}.jpg. 
   press "f" to capture orgin image, image will save as calibration0.jpg.
   press "q" to exit the program.
   **first image (orgin image) can take individual.**
   **first image must be orgin image, other image must be common image.**
	
	python3 capture.py
	
   All image will save in calibration_image folder.
	
	
# 2. use calibration_v2.py to calibrate single camera. 
   **(in general only need to do in reference camera C1)**
   it will save the internal and orgin rotation, translation np array as .npy format.
	
python3 calibration_v2.py calibration/image/folder/path pattern_width pattern_height square_size
**pattern_width and pattern_height is the nuumber of inner corner, square_size is the size of each box in mm**
	
	python3 calibration_v2.py calibration_image_1 5 4 75
	
   function will output distortion.npy , internal_matrix.npy , rotation_vector.npy , translation_vector.npy 
	
	
# 3. run check_accuracy.py to evaluate the performance. use 'esc' to exit
   check on any place on image and measure the distance to origin in real world see if it similar to the output by the function
	
	python3 check_accuracy.py internal_matrix.npy rotation_vector.npy translation_vector.npy distortion.npy image_origin.jpg
	


# 4. move the origin image out of the folder and rename them e.g. C1_calibration1.jpg C2_calibration1.jpg
**stereo_calibration result may not accurate must check**
   run stereo_calibration.py to have the relation between two camera
	
python3 stereo_calibration.py calibration/image/folder/path1 calibration/image/folder/path2 camera1_internal_matrix camera2_internal_matrix camera1_distortion camera2_distortion pattern_width pattern_height square_size

	python3 stereo_calibration.py calibration_image_1 calibration_image_2 c1_internal_matrix.npy c2_internal_matrix.npy c1_distortion.npy c2_distortion.npy 5 4 75
	
   function will output rvec_relative.npy , tvec_relative.npy

# 4.1. To check the relative vector. run check_camera_coordinate.py and check_accuracy.py at the same time:
**use a set of common image, kick on the same place(world coordinate) of both image and check their terminal see if the vetcor the same**
	
 	python3 check_camera_coordinate.py c1_internal_matrix.npy c1_rotation_vector.npy c1_translation_vector.npy c1_distortion.npy rvec_relative.npy tvec_relative.npy c1_calibration1.jpg 

	python3 check_accuracy.py c2_internal_matrix.npy c2_rotation_vector.npy c2_translation_vector.npy c2_distortion.npy c2_look_c1_origin.jpg 


# 5. run camera_orgin_transfer.py to make c2 have the same orgin

	python3 camera_orgin_transfer.py c1_rotation_vector.npy c1_translation_vector.npy rvec_relative.npy tvec_relative.npy
   
   function will get T_sec2world.npy
	

# 6. run check_accuracy_world.py to evaluate the performance. use 'esc' to exit
   check on any place on image and measure the distance to origin in real world see if it similar to the output by the function
	
	python3 check_accuracy_world.py c2_internal_matrix.npy c2_rotation_vector.npy c2_translation_vector.npy c2_distortion.npy T_sec2world.npy c2_look_c1_origin.jpg
