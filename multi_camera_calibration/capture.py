import argparse
import os.path

import cv2
if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="single camera calibration using raw image")
    parser.add_argument("cam_no", type=int, help="Camera number")
    args = parser.parse_args()

    cam_no = args.cam_no
    # Open the default camera (0)
    cap = cv2.VideoCapture(cam_no)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        exit()

    os.makedirs(f'cam_param{cam_no}', exist_ok=True)
    i=0
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame from the camera.")
            break

        # Display the frame on the screen
        cv2.imshow('Press "c" to capture a frame, "q" to quit', frame)

        # Wait for a key press and check if it's 'c' or 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            i+=1
            cv2.imwrite(f'cam_param{cam_no}/calibration{i}.jpg', frame)
            print(f"Frame captured and saved as calibration_image/calibration{i}.jpg")
        elif key == ord('f'):
            cv2.imwrite(f'cam_param{cam_no}/calibration0.jpg', frame)
            print(f"Frame captured and saved as calibration_image/calibration0.jpg")
        elif key == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()
