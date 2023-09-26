import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from datetime import datetime
import time

def pbpthreshold(image,threshold):
    #Compares image to thresholding image, set binary, if < than threshold image pixel = 0, if >, pixel goes to 255.
    flatimage = image.flatten()
    flatthreshold = threshold.flatten()
    if len(image) != len(threshold):
        print("Image and Threshold Matrix are incompatible sizes")
        return
    for i in range(len(flatimage)):
        if flatimage[i] < flatthreshold[i]:
            flatimage[i] = 0
        elif flatimage[i] >= flatthreshold[i]:
            flatimage[i] = 255
            
    image = flatimage.reshape(image.shape)
    return image

# Initialize the camera
camera = cv.VideoCapture(0)

# Set camera properties
camera_frame = (1920,1080) #x,y dimensions (same as projector)
camera.set(cv.CAP_PROP_FRAME_WIDTH, camera_frame[0])
camera.set(cv.CAP_PROP_FRAME_HEIGHT, camera_frame[1]) 
# Capture Initial image with no projection
ret, frame = camera.read()
filename = "test_image.png"
directory = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding" #change to PI directory
cv.imwrite(directory + "\\" + filename, frame) 
print('ready to take next frame')
time.sleep(5)
ret, frame2 = camera.read()
filename2 = "blank_image.png"
cv.imwrite(directory + "\\" + filename2, frame2)
camera.release()
print('pictures taken')

start = datetime.now()
testimage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\" + filename)
testimage = cv.cvtColor(testimage, cv.COLOR_BGR2GRAY)
print('test image converted')
noise_thresh = 255*np.random.rand(testimage.shape[0],testimage.shape[1])
#noise_thresh = pbpthreshold(testimage,noise_thresh)
print('pixel thresholding')
cv.imshow('Random pixel by pixel Threshold',noise_thresh)
print('image displayed') 
blank_image = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\" + filename2)
blank_image = cv.cvtColor(blank_image, cv.COLOR_BGR2GRAY)
print('blank image converted')
#Subtract image with no projection to remove Reflections
if testimage.shape != blank_image.shape:
    print('the test and blank image shapes are incompatible')
time.sleep(1)
test1 = cv.subtract(testimage,blank_image)

#ret, thresh = cv.threshold(gray,127,220,cv.THRESH_BINARY) 
test2 = pbpthreshold(test1,noise_thresh)

cv.imshow('test after removing added light',test1)
cv.waitKey(10000)
cv.imshow('test after noise thresholding subtracted images',test2)
end = datetime.now()
cv.waitKey(10000)

cv.destroyAllWindows()

et = (end - start).total_seconds() 
print(f"Execution time : in seconds: {(end - start).total_seconds() }, in time format: {(end - start)}")




