# Import required modules 
import cv2 
import numpy as np 
import os 
import glob 
from calibration_v1 import draw_axes, image_to_world, world_to_image  

# Open the default camera (0)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()


# Load the saved array
matrix = np.load('internal_matrix.npy')
print(matrix.shape)
r_vec = np.load('rotation_vector.npy')
print(r_vec.shape)
t_vec = np.load('translation_vector.npy')
print(t_vec.shape)

# Define box length
box_length = 75 #mm

world_coords = None
mouseX = 0
mouseY = 0
# Function to handle mouse events
def draw_circle(event, x, y, flags, param):
    global mouseX, mouseY, world_coords 

    if event == cv2.EVENT_LBUTTONDOWN:
        mouseX, mouseY = x, y
        # img = original_img.copy()
        input_img_coords = np.array((mouseX, mouseY), dtype=np.float32)  # Example input image coordinates
        world_coords = image_to_world(input_img_coords, matrix, r_vec, t_vec) * box_length
        # cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        # cv2.putText(img, f'{(str(world_coords[0])),(str(world_coords[1]))}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        # print(world_coords)
        

# Read the image
origin_img = cv2.imread('image_origin.jpg')

# Create a named window and bind the mouse callback function to it
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_circle)

while True:
    # Display the image
    ret, img = cap.read()
    if not ret:
        print("Error: Could not read frame from the camera.")
        break
    if world_coords is not None:
        cv2.circle(img, (mouseX, mouseY), 1, (0, 0, 255), -1)
        cv2.putText(img, f'{(str(world_coords[0])),(str(world_coords[1]))}', (mouseX, mouseY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow('Image', img)
    # Break the loop when 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Close all windows
cv2.destroyAllWindows()
cap.release()
