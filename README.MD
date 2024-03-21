# Multi Camera Multi Object Tracking

This repository enables (DeepSort) tracking across multiple calibrated cameras.

## MCMOLocal will do the following things:

1. Do object detection on the captured image
2. Transform the camera coordinate at the middle-bottom of bounding boxes into world coordinates
3. Crop the image according to bounding boxes and encode it into 512-d features using osnet
4. Send world coordinates and features to the server

## MCMOServer will do the following things:

1. Do tracking on each data using DeepSort
2. Merge matches

# Dependencies

`python3 -m pip install -r requirements.txt`

# Camera calibration

```bash
# do it for all connected cameras 
# (export cam=0, 1, 2...)
export cam=0
# place the checkboard on the ground, visible to all cameras.
# press c to capture, press q to quit 
python3 multi_camera_calibration/capture.py $cam
# if one of the images shows correct corner locations, done
# otherwise, re-do the capture part
python3 multi_camera_calibration/calibration_v2.py cam_param$cam [number of checkerboard corners per row] [number of checkerboard corners per column] [actual distance between corners]
python3 multi_camera_calibration/check_accuracy.py cam_param$cam
```

# Running

```bash
# run each of the following commands in separate terminal
# run it just once
python3 mcmo_server.py [no of cameras]
# run it for each camera (camera number = 0, 1, 2...)
python3 mcmo_local.py [camera number]
```

## Example

Assume I have 3 cameras connected to the computer.

```bash
# do it for all connected cameras 
# (export cam=0, 1, 2...)
export cam=0
# place the checkboard on the ground, visible to all cameras.
# press c to capture, press q to quit 
python3 multi_camera_calibration/capture.py $cam
# if one of the images shows correct corner locations, done
# otherwise, re-do the capture part
python3 multi_camera_calibration/calibration_v2.py cam_param$cam 4 5 125
python3 multi_camera_calibration/check_accuracy.py cam_param$cam
export cam=1
# place the checkboard on the ground, visible to all cameras.
# press c to capture, press q to quit 
python3 multi_camera_calibration/capture.py $cam
# if one of the images shows correct corner locations, done
# otherwise, re-do the capture part
python3 multi_camera_calibration/calibration_v2.py cam_param$cam 4 5 125
python3 multi_camera_calibration/check_accuracy.py cam_param$cam
export cam=2
# place the checkboard on the ground, visible to all cameras.
# press c to capture, press q to quit 
python3 multi_camera_calibration/capture.py $cam
# if one of the images shows correct corner locations, done
# otherwise, re-do the capture part
python3 multi_camera_calibration/calibration_v2.py cam_param$cam 4 5 125
python3 multi_camera_calibration/check_accuracy.py cam_param$cam

# by now, the root repository should have 3 new directories named "cam_param0", "cam_param1" and "cam_param2"
# run each of the following commands in separate terminal
# run it just once
python3 mcmo_server.py 3
# run it for each camera (camera number = 0, 1, 2...)
python3 mcmo_local.py 0
python3 mcmo_local.py 1
python3 mcmo_local.py 2
````

# Note

1. The ID in local will keep changing because no tracking is performed at local. The ID order are just referred by
   engine output.