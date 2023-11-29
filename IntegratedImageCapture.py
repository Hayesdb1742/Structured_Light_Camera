import numpy as np
import cv2 as cv
import open3d as o3d
import math
import os
import sys
import matplotlib.pyplot as plt
import subprocess
from StructuredLight import SLscan as SL


path = os.getcwd()
numberRotations= sys.argv[1]

# ## EXAMPLE USAGE FOR ONE VIEW:
# directory = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImagesPi"
# scanner = SLscan(1920,1080,directory)
# N = scanner.generate_gray_code_patterns()
# ## PROJECT AND TAKE PICTURES HERE ##


directory = os.getcwd()
width = 1920
height = 1080
scanner = SL(width,height,directory)
N = scanner.generate_gray_code_patterns()     ##only needs to generate one time, just needs to be saved in PI, N is number of images

# Initialize the camera
camera = cv.VideoCapture(0)

# Set camera properties
camera_frame = (1920,1080) #x,y dimensions (same as projector)
camera.set(cv.CAP_PROP_FRAME_WIDTH, camera_frame[0])
camera.set(cv.CAP_PROP_FRAME_HEIGHT, camera_frame[1]) 
# Capture Initial image with no projection

#Capture blankImage
blankPoints = np.zeros((1080,1920,3), dtype=np.uint8)
blackProjection = subprocess.Popen("feh --fullscreen -Y black_screen.jpg", shell=True)
cv.waitKey(500)
for i in range(20):
    ret, frame = camera.read()
ret, frame = camera.read()
filename = f'./grayCodePics/{numberRotations}/blank_image.png'
directory = path #change to PI directory
# Capture Image with all White projection
if not ret:
    print("Image was not captured")
    exit(0)
else:
    print("blank image captured...")
cv.imwrite(filename,frame)
subprocess.run(f"kill -9 {blackProjection.pid+1}", shell=True)
blackProjection.terminate()

projection = subprocess.Popen(f"feh --fullscreen -Y white_image.png", shell=True)
cv.waitKey(500)
for i in range(20):
    ret, frame = camera.read()
ret, frame = camera.read()
filename = f"./grayCodePics/{numberRotations}/full_image.png"
if not ret:
    print("Image not saved")
else:
    print("full image saved...")
cv.imwrite(filename, frame)
subprocess.run(f"kill -9 {projection.pid +1}", shell=True)
projection.terminate()

# Loop through and capture multiple images after projection
for i in range(N):
    # Project Image # 
    # To-Do: Project image
    projection = subprocess.Popen(f"feh --fullscreen -Y gray_pattern{i+1}.png", shell=True)
    cv.waitKey(500)
    #Also need to add while loop
    for j in range(20):
        ret, frame = camera.read()
    # Capture a single frame from the camera
    ret, frame = camera.read()
    if not ret:
        print(f"Projection Image: {i} was not captured")
        exit(0)
    else:
        print(f"Succesfully captured {i}")
    filename = f"./grayCodePics/{numberRotations}/image_{i}.png"
    directory = path #change to PI directory
    imgSave = cv.imwrite(filename, frame)
    if not imgSave:
        print("Test image did not save")
    subprocess.run(f"kill -9 {projection.pid +1}", shell=True)
    projection.terminate()

# Release the camera
camera.release()

blankImage = cv.imread(f"./grayCodePics/{numberRotations}/blank_image.png")
fullImage = cv.imread(f"./grayCodePics/{numberRotations}/full_image.png")
if not blankImage.any():
    print("Blank Image not found")
if not fullImage.any():
    print("Full Image not found")
    
# captured_patterns = scanner.load_images()
# decoded = scanner.preprocess(captured_patterns,threshold=100)
# scanner.calculate_3D_points(decoded)
# scanner.visualize_point_cloud()
# scanner.save(view=0)