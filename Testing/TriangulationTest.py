import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import open3d as o3d

def triangulate_points(decoded_image, K, R, t, P_proj):
    """
    :param decoded_image: The image with decoded pixel values (disparity values).
    :param K: Intrinsic matrix of the camera.
    :param R: Rotational matrix.
    :param t: Translational vector.
    :param P_proj: Projection matrix of the projector (or the second camera).
    :return: 3D points.
    """
    # Ensure K is 3x3
    assert K.shape == (3, 3), "Intrinsic matrix K must be 3x3"
    # Ensure R is 3x3
    assert R.shape == (3, 3), "Rotation matrix R must be 3x3"
    # Ensure t is 3x1
    assert t.shape == (3, 1) or t.shape == (3,), "Translation vector t must be 3x1 or 3"

    # Handle if t is given as a 1D array
    if t.ndim == 1:
        t = t[:, np.newaxis]
    
    
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

    # Reshape to 3D array 
    points_3D_reshaped = points_3D.T.reshape(height, width, 3)
    
    return points_3D_reshaped

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


blankImage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\blankImage.bmp")
fullImage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\fullImage.bmp")
blankImage = cv.cvtColor(blankImage, cv.COLOR_BGR2GRAY)
fullImage = cv.cvtColor(fullImage,cv.COLOR_BGR2GRAY)
N = 10
image_array = np.empty((N,fullImage.shape[0],fullImage.shape[1]), dtype=blankImage.dtype)
avg_thresh = cv.addWeighted(blankImage,0.5,fullImage,0.5,0) #add white and blank images for thresholding and average
avg_thresh = cv.divide(avg_thresh,2)                           #divide to finish averaging (per pixel thresholding)


for i in range(N):
    # load image array and pre-process images
    filein = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\{}.bmp".format(i+1)
    image = cv.imread(filein)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #Convert to black and white based on grascale (average per pixel thresholding)
    image_thresh = pbpthreshold(image_gray,avg_thresh)
    image_array[i] = image_thresh
    
decoded_image = decodeGrayPattern(image_array,N)
# plt.imshow(decoded_image)
# plt.show()
K = np.array([[1,4,5],[2,1,1],[0,0,1]])
R = np.array([[3,2,0],[1,1,0],[0,0,1]])
t = np.array([[0],[0],[1]])
P_proj = np.array([[1,1,0,-10],[0,1,0,0],[0,0,1,3]])

points_3D = triangulate_points(decoded_image,K,R,t,P_proj)
print(points_3D.shape)
visualize_point_cloud(points_3D)
