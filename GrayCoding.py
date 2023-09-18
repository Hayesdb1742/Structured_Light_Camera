import numpy as np
import cv2 as cv
import math

N = 10 #user defined numImages (more precise for greater N, max for HD at 10)
projector_size = (1920, 1080) #projector defined

def createGrayPattern(N, projector_size):
    num_patterns = 2**N
    pattern = np.zeros((projector_size[1], projector_size[0]), dtype=np.uint8)
    
    for i in range(num_patterns):
        gray_code = i ^ (i >> 1)  # Calculate Gray code
        stripe_width = math.ceil(projector_size[0] / num_patterns)
        
        if gray_code % 2 == 0:
            pattern[:, i * stripe_width: (i + 1) * stripe_width] = 255
    
    directory = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\GrayCodedPictures"  # Change to your directory
    filename = "gray_pattern{}.png".format(N)
    cv.imwrite(directory + "\\" + filename, pattern)

for i in range(N):
    createGrayPattern(i + 1, projector_size)
    