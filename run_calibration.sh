export cam=0
echo calibrating cam$cam
python3 multi_camera_calibration/capture.py $cam
python3 multi_camera_calibration/calibration_v2.py cam_param$cam 4 5 125
python3 multi_camera_calibration/check_accuracy.py cam_param$cam