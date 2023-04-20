import cv2
import numpy as np
import math

N = 10 #user defined numImages (more precise for greater N, max for HD at 10)
projector_size = (1920, 1080) #projector defined

#stores binary patterns on PI
def createBinaryPattern(N,projector_size):
    num_stripes = 2**N
    stripe_width = math.ceil(projector_size[0] / num_stripes)
    pattern = np.zeros((projector_size[1],projector_size[0]), dtype=np.uint8)
    for i in range(num_stripes):
        if i % 2 == 0:
            pattern[:, i*stripe_width : (i+1)*stripe_width] = 255
    directory = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\BinaryCodingPictures" #change to PI directory
    filename = "binary_pattern{}.png".format(N)                         
    cv2.imwrite(directory + "\\" + filename, pattern) 
 
for i in range(N):
    createBinaryPattern(i+1, projector_size)


## CODE HERE FOR PROJECTOR SETTINGS ##


# Initialize the camera
camera = cv2.VideoCapture(0)

# Set camera properties
camera_frame = (1920,1080) #x,y dimensions (same as projector)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_frame[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_frame[1])

# Capture Initial image with no projection
ret, frame = camera.read()
filename = "blank_image.png"
directory = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\BinaryCodingPictures" #change to PI directory

# Loop through and capture multiple images after projection
for i in range(N):
    # Project Image # 
    
    # Capture a single frame from the camera
    ret, frame = camera.read()
    filename = "image_{}.png".format(i+1)
    directory = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\BinaryCodingPictures" #change to PI directory
    cv2.imwrite(directory + "\\" + filename, frame)

# Release the camera
camera.release()

blankImage = cv2.imread('blank_image.png')
#Loop over all captured images
image_array = np.empty((N,camera_frame[1],camera_frame[0]), dtype=object)
binary_array = np.zeros((N,2**N), dtype=int)
x_array = np.zeros((N,2**N), dtype=int)

for i in range(N):
    filein = "image_{}.png".format(i+1)
    image_array[i] = cv2.imread(filein)
    #Subtract image with no projection to remove Reflections
    image_array[i] = cv2.subtract(image_array[i],blankImage)
    
    #Convert to black and white based on grascale
    gray = cv2.cvtColor(image_array[i], cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,127,220,cv2.THRESH_BINARY)
    
    stripe_width = math.ceil(projector_size[0] / (2**i))
    for j in range(2**i):
        x = stripe_width * j + math.ceil(stripe_width/2)
        x_array[i,j] = x
        binary_array[i,j] = np.sum(thresh[:,j*stripe_width:(j+1)*stripe_width]) // 255



 