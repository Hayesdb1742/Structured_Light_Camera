import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math

def triangulate_points(decoded_image, K, R, t, P_proj):
    """
    :param decoded_image: The image with decoded pixel values (disparity values).
    :param K: Intrinsic matrix of the camera.
    :param R: Rotational matrix.
    :param t: Translational vector.
    :param P_proj: Projection matrix of the projector (or the second camera).
    :return: 3D points.
    """
    height, width = decoded_image.shape

    # Create the camera's projection matrix
    P_cam = np.dot(K, np.hstack((R, t)))

    # Placeholder for the 3D points
    points_3D = np.zeros((height, width, 3), dtype=np.float32)

    # Process each pixel
    for y in range(height):
        for x in range(width):
            # Construct homogeneous coordinates for the pixel
            pixel_homogeneous = np.array([x, y, 1])
            disparity = decoded_image[y, x]

            # If disparity is 0, skip triangulation for this pixel
            if disparity == 0:
                continue

            # Homogeneous coordinates for the corresponding point in the projector
            projector_homogeneous = np.array([x - disparity, y, 1])

            # Stack the points for triangulation
            A = np.vstack((pixel_homogeneous @ P_cam, projector_homogeneous @ P_proj))

            # Obtain the 3D point by solving AX=0
            _, _, Vt = np.linalg.svd(A)
            point_homogeneous = Vt[-1]

            # Convert from homogeneous to 3D Cartesian coordinates
            points_3D[y, x, :] = point_homogeneous[:3] / point_homogeneous[3]

    return points_3D

def triangulate_using_opencv(decoded_image, K, R, t, P_proj):
    height, width = decoded_image.shape

    # Create the camera's projection matrix
    P_cam = np.dot(K, np.hstack((R, t)))

    # Lists to hold corresponding points from camera and projector
    camera_points = []
    projector_points = []

    # Loop through each pixel
    for y in range(height):
        for x in range(width):
            disparity = decoded_image[y, x]
            if disparity == 0:
                continue
            camera_points.append([x, y])
            projector_points.append([x - disparity, y])

    camera_points = np.array(camera_points, dtype=np.float32)
    projector_points = np.array(projector_points, dtype=np.float32)

    # Triangulate using OpenCV
    points_homogeneous = cv.triangulatePoints(P_cam, P_proj, camera_points.T, projector_points.T)
    
    # Convert from homogeneous to Cartesian coordinates
    points_3D = points_homogeneous[:3, :] / points_homogeneous[3, :]

    return points_3D.T

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


blankImage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\blankImage.bmp")
fullImage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\fullImage.bmp")
blankImage = cv.cvtColor(blankImage, cv.COLOR_BGR2GRAY)
fullImage = cv.cvtColor(fullImage,cv.COLOR_BGR2GRAY)
N = 10
image_array = np.empty((N,fullImage.shape[0],fullImage.shape[1]), dtype=object)
avg_thresh = cv.addWeighted(blankImage,0.5,fullImage,0.5,0) #add white and blank images for thresholding and average
avg_thresh = cv.divide(avg_thresh,2)                           #divide to finish averaging (per pixel thresholding)

im1 = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\1.bmp")
im1 = cv.cvtColor(im1,cv.COLOR_BGR2GRAY)
image_thresh = pbpthreshold(im1,avg_thresh)
cv.imshow('im1',image_thresh)
cv.waitKey(0)

# for i in range(N):
#     # load image array and pre-process images
#     filein = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\{}.bmp".format(i+1)
#     image = cv.imread(filein)
#     image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#     #Convert to black and white based on grascale (average per pixel thresholding)
#     image_thresh = pbpthreshold(image_gray,avg_thresh)
#     image_array[i] = image_thresh
    
cv.imshow('Img3',image_array[3])
cv.waitKey(0)