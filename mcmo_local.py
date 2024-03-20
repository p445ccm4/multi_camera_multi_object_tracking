"""
An example that uses TensorRT's Python api to make inferences.
"""
import argparse
import ctypes
import socketio
import sys
import cv2
import numpy as np
from deep_sort.deep.feature_extractor import Extractor
from yolov7trt import YoLov7TRT
from draw import draw_boxes
from multi_camera_calibration.img_to_world import ImgToWorld


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MCMO local")
    parser.add_argument("cam_no", type=int, help="Camera number")
    args = parser.parse_args()
    cam_no = args.cam_no

    # load custom plugin and engine
    PLUGIN_LIBRARY = "build/libmyplugins.so"
    engine_file_path = "build/yolov7.engine"

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

    # a YoLov7TRT instance
    yolov7_wrapper = YoLov7TRT(engine_file_path)
    feature_extractor = Extractor("deep_sort/deep/checkpoint/osnet_x0_25.engine", use_cuda=True)
    cam = ImgToWorld(cam_id=cam_no)

    # read video and do inference than save the result video
    # cap = cv2.VideoCapture("video/MOT_test_video.mp4")
    cap = cv2.VideoCapture(cam_no)
    # Set the input resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Create a SocketIO client
    sio = socketio.Client()
    # Connect to the SocketIO server
    sio.connect('http://127.0.0.1:5000')

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
            world_coordinates = world_coordinates[:, :2].reshape(-1, 2) * 0.001
            # draw boxes and axes for visualization
            frame = draw_boxes(frame, result_boxes, world_coordinates=world_coordinates)
            frame = cam.draw_axes(frame)

            # np.save(f'data/{t}_world_coordinates.npy', world_coordinates)
            # np.save(f'data/{t}_features.npy', features)
            # np.save(f'data/{t}_bbox_areas.npy', bbox_areas)
            # t += 1

            sio.emit('update', (world_coordinates.tolist(), features.tolist(), bbox_areas.tolist()))
        else:
            frame = frame
            frame = cam.draw_axes(frame)

        cv2.imshow("result", frame)
        if cv2.waitKey(200) & 0xFF == ord('q'):
            break

    # destroy the instance
    yolov7_wrapper.destroy()
    # Disconnect from the SocketIO server
    sio.disconnect()