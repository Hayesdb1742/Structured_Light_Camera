import numpy as np
import cv2 as cv
import math
from matplotlib import pyplot as plt

N = 10 #user defined numImages (more precise for greater N, max for HD at 10)
projector_size = (1920, 1080) #projector defined

# def createGrayPattern(N, projector_size):
#     num_patterns = 2**N
#     pattern = np.zeros((projector_size[1], projector_size[0]), dtype=np.uint8)
    
#     for i in range(num_patterns):
#         gray_code = i ^ (i >> 1)  # Calculate Gray code
#         stripe_width = math.ceil(projector_size[0] / num_patterns)
        
#         if gray_code % 2 == 0:
#             pattern[:, i * stripe_width: (i + 1) * stripe_width] = 255
     
#     directory = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\GrayCodedPictures"  # Change to Pi directory
#     filename = "gray_pattern{}.png".format(N)
#     cv.imwrite(directory + "\\" + filename, pattern)
    

# for i in range(N):
#     createGrayPattern(i + 1, projector_size)
   
# # Initialize the camera
# camera = cv.VideoCapture(0)

# # Set camera properties
# camera_frame = (1920,1080) #x,y dimensions (same as projector)
# camera.set(cv.CAP_PROP_FRAME_WIDTH, camera_frame[0])
# camera.set(cv.CAP_PROP_FRAME_HEIGHT, camera_frame[1]) 
# # Capture Initial image with no projection
# ret, frame = camera.read()
# filename = "blank_image.png"
# directory = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\GrayCodedPictures" #change to PI directory

# # Capture Image with all White projection
# ret, frame = camera.read()
# filename = "full_image.png"

# # Loop through and capture multiple images after projection
# for i in range(N):
#     # Project Image # 
    
#     # Capture a single frame from the camera
#     ret, frame = camera.read()
#     filename = "image_{}.png".format(i+1)
#     directory = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\GrayCodedPictures" #change to PI directory
#     cv.imwrite(directory + "\\" + filename, frame)

# # Release the camera
# camera.release()

# blankImage = cv.imread('blank_image.png')
# fullImage = cv.imread('full_image.png')
#Loop over all captured images
image_array = np.empty((N,camera_frame[1],camera_frame[0]), dtype=object)
# binary_array = np.zeros((N,2**N), dtype=int)
# x_array = np.zeros((N,2**N), dtype=int) 
# avg_thresh = cv.addWeighted(blankImage, 0.5, fullImage, 0.5,0) #add white and blank images for thresholding and average
# avg_thresh = cv.divide(avg_thresh,2)                           #divide to finish averaging (per pixel thresholding)

# def pbpthreshold(image,threshold):
#     #Compares image to thresholding image, set binary, if < than threshold image pixel = 0, if >, pixel goes to 255.
#     image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     flatimage = image.flatten()
#     flatthreshold = threshold.flatten()
#     if len(image) != len(threshold):
#         print("Image and Threshold Matrix are incompatible sizes")
#         return
#     for i in range(len(flatimage)):
#         if flatimage[i] < flatthreshold[i]:
#             flatimage[i] = 0
#         elif flatimage[i] >= flatthreshold[i]:
#             flatimage[i] = 255
            
#     image = flatimage.reshape(image.shape)
#     return image


for i in range(N):
    # load image array and pre-process images
    filein = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\GrayCodedPictures\\gray_capture{}.png".format(i+1)
    image_array[i] = cv.imread(filein)
    #Subtract image with no projection to remove Reflections
    image_array[i] = cv.subtract(image_array[i],blankImage)
    
    #Convert to black and white based on grascale (average per pixel thresholding)
    image_array[i] = pbpthreshold(image_array[i],avg_thresh)

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
                
            # Convert binary to Gray code
            decimal = int(binary_str, 2)
            gray_decimal = binaryToGray(decimal)
            
            decoded_image[y, x] = gray_decimal

    return decoded_image

def binaryToGray(n):
    n = int(n, 2)
    n ^= (n >> 1)

    return n


for i in range(N):
    # load image array and pre-process images
    filein = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\GrayCodedPictures\\gray_capture{}.png".format(i+1)
    image_array[i] = cv.imread(filein)
    #Subtract image with no projection to remove Reflections
    image_array[i] = cv.subtract(image_array[i],blankImage)
    
    #Convert to black and white based on grascale (average per pixel thresholding)
    image_array[i] = pbpthreshold(image_array[i],avg_thresh)

# Call the decoding function
decoded_image = decodeGrayPattern(image_array, N)

# Optionally, save the decoded image
cv.imwrite("decoded_image.png", decoded_image)