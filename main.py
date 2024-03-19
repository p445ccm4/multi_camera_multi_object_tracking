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

from yolov7trt import YoLov7TRT
from deep_sort.deep_sort import DeepSort
from draw import draw_boxes
from multi_camera_calibration.img_to_world import ImgToWorld


# def plot_one_box(x, img, color=None, label=None, line_thickness=None):
#     """
#     description: Plots one bounding box on image img,
#                  this function comes from YoLov7 project.
#     param: 
#         x:      a box likes [x1,y1,x2,y2]
#         img:    a opencv image object
#         color:  color to draw rectangle, such as (0,255,0)
#         label:  str
#         line_thickness: int
#     return:
#         no return

#     """
#     tl = (
#         line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
#     )  # line/font thickness
#     color = color or [random.randint(0, 255) for _ in range(3)]
#     c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
#     cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
#     if label:
#         tf = max(tl - 1, 1)  # font thickness
#         t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
#         c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
#         cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
#         cv2.putText(
#             img,
#             label,
#             (c1[0], c1[1] - 2),
#             0,
#             tl / 3,
#             [225, 255, 255],
#             thickness=tf,
#             lineType=cv2.LINE_AA,
#         )


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
    tracker = DeepSort("deep_sort/deep/checkpoint/osnet_x0_25.engine", max_dist=0.2, min_confidence=0.4, nms_max_overlap=1, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True)
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
        # do tracking
            outputs, features = tracker.update(result_boxes, result_scores, img)
        else:
            outputs, features = np.array([]), np.array([])

        if len(outputs) > 0:
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            world_coordinates = cam.get_world_coordinates(outputs)
            # draw boxes and axes for visualization
            frame = draw_boxes(frame, bbox_xyxy, identities, world_coordinates)
            frame = cam.draw_axes(frame)

            # np.save(f'data/{t}_world_coordinates.npy', world_coordinates)
            # np.save(f'data/{t}_features.npy', features)
            # np.save(f'data/{t}_identities.npy', identities)
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
