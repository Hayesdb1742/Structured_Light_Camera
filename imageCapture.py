import numpy as np
import cv2 as cv
import math
import os
import sys
import matplotlib.pyplot as plt
import subprocess


N = 10 #user defined numImages (more precise for greater N, max for HD at 10)
projector_size = (1920, 1080) #projector defined
path = os.getcwd()
numberRotations= sys.argv[1]

def createGrayPattern(N, projector_size):
    num_patterns = 2**N
    pattern = np.zeros((projector_size[1], projector_size[0]), dtype=np.uint8)
    
    for i in range(num_patterns):
        gray_code = i ^ (i >> 1)  # Calculate Gray code
        stripe_width = math.ceil(projector_size[0] / num_patterns)
        
        if gray_code % 2 == 0:
            pattern[:, i * stripe_width: (i + 1) * stripe_width] = 255
    
    directory = path  # Change to Pi directory
    filename = "gray_pattern{}.png".format(N)
    cv.imwrite(directory + "/" + filename, pattern)
    
    
def pbpthreshold(image,threshold):
    #Compares image to thresholding image, set binary, if < than threshold image pixel = 0, if >, pixel goes to 255.
    flatimage = image.flatten()
    flatthreshold = threshold.flatten()
    if len(image) != len(threshold):
        print("Image and Threshold Matrix are incompatible sizes")
        return
    for i in range(len(flatimage)):
        if flatimage[i] <= flatthreshold[i]:
            flatimage[i] = 0
        elif flatimage[i] > flatthreshold[i]:
            flatimage[i] = 255
            
    image = flatimage.reshape(image.shape)
    return image


def decodeGrayPattern(image_array, N):
    height, width = image_array[0].shape

    # Initialize a 2D array to store the final decoded values
    decoded_image = np.zeros((height, width), dtype=np.uint16)

    for y in range(height):
        for x in range(width):
            binary_str = ''
            for i in range(N):
                # Convert 255 to 1 and 0 to 0
                binary_str += '1' if image_array[i][y, x] == 255 else '0'
                 
            # Convert binary string to its integer representation
            decimal = int(binary_str, 2)
            
            # Convert integer representation of binary to Gray code integer
            gray_decimal = binaryToGray(decimal)
            
            decoded_image[y, x] = gray_decimal
    
    
    return decoded_image


def binaryToGray(binary_integer):
    binary_integer ^= (binary_integer >> 1)
    return binary_integer


def visualize_point_cloud(points_3D):
    # Convert points from 3D matrix to list of 3D coordinates
    points = points_3D.reshape(-1, 3)
    # Filter out zero points
    points = points[np.any(points != [0, 0, 0], axis=1)]

    # Convert the points to an open3d point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    # Visualize the point cloud
    o3d.visualization.draw_geometries([point_cloud])


    
for i in range(N):
    createGrayPattern(i + 1, projector_size)
   
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
    
blankImage = cv.cvtColor(blankImage, cv.COLOR_BGR2GRAY)
fullImage = cv.cvtColor(fullImage,cv.COLOR_BGR2GRAY)
N = 10
image_array = np.empty((N,fullImage.shape[0],fullImage.shape[1]), dtype=blankImage.dtype)
avg_thresh = cv.addWeighted(blankImage,0.5,fullImage,0.5,0) #add white and blank images for thresholding and average
avg_thresh = cv.divide(avg_thresh,2)                           #divide to finish averaging (per pixel thresholding)

