"""
An example that uses TensorRT's Python api to make inferences.
"""
import ctypes
import os
import pickle
import shutil
import random
import socket

import sys
import threading
import time
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

from deep_sort.deep.feature_extractor import Extractor
from yolov7trt import YoLov7TRT
from deep_sort.deep_sort import DeepSort
from draw import draw_boxes
from multi_camera_calibration.img_to_world import ImgToWorld


if __name__ == "__main__":
    # load custom plugin and engine
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "build/yolov7.engine"

    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    # load coco labels

    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

    # if os.path.exists('output/'):
    #     shutil.rmtree('output/')
    # os.makedirs('output/')
    # a YoLov7TRT instance
    yolov7_wrapper = YoLov7TRT(engine_file_path)
    # tracker = DeepSort("deep_sort/deep/checkpoint/osnet_x0_25.engine", max_dist=0.2, min_confidence=0.4, nms_max_overlap=1, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True)
    feature_extractor = Extractor("deep_sort/deep/checkpoint/osnet_x0_25.engine", use_cuda=True)
    cam = ImgToWorld(cam_id=0)

    # read video and do inference than save the result video
    # cap = cv2.VideoCapture("video/MOT_test_video.mp4")
    cap = cv2.VideoCapture(0)
    # Set the input resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    # Define the codec and create VideoWriter object  (mp4)
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('video/output.mp4',fourcc, 20.0, (int(cap.get(3)),int(cap.get(4))))

    # # Create a socket
    # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # # Connect to the receiver
    # s.connect(('127.0.0.1', 5000))
    t = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read() # 640x480
        if ret == False:
            break
        # Our operations on the frame come here
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_boxes, result_scores, result_classid = yolov7_wrapper.infer(img)

        # select person class
        mask = result_classid == 0
        result_boxes = result_boxes[mask]
        result_scores = result_scores[mask]

        if len(result_boxes) > 0:
            # extract features
            features = []
            bbox_areas = []
            for box in result_boxes:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                im = img[y1:y2, x1:x2]

                feature = feature_extractor([im])
                features.append(feature)
                bbox_area = abs((x1-x2) * (y1-y2))
                bbox_areas.append(bbox_area)

            features = np.concatenate(features)
            bbox_areas = np.asarray(bbox_areas)

            world_coordinates = cam.get_world_coordinates(result_boxes)
            # draw boxes and axes for visualization
            frame = draw_boxes(frame, result_boxes, world_coordinates=world_coordinates)
            frame = cam.draw_axes(frame)

            np.save(f'data/{t}_world_coordinates.npy', world_coordinates)
            np.save(f'data/{t}_features.npy', features)
            np.save(f'data/{t}_bbox_areas.npy', bbox_areas)
            t += 1

            # # Serialize the object using pickle
            # data = pickle.dumps(obj)
            # # Send the serialized object to the receiver
            # s.send(data)
            # print(obj)
        else:
            frame = frame
            frame = cam.draw_axes(frame)

        # save the result video
        # out.write(frame)
        cv2.imshow("result", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destroy the instance
    yolov7_wrapper.destroy()
    # Close the socket
    # s.close()
