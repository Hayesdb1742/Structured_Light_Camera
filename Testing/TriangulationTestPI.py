import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import math
import open3d as o3d

# THIS RAYPLANE DOESN"T QUITE WORK 
# def ray_plane_intersection(decoded_gray_image, camera_matrix, projector_matrix, rotation_matrix, translation_vector):
    
#     inv_camera_matrix = np.linalg.inv(camera_matrix) # Invert the camera matrix 
#     # create array to save intersection points
#     intersection_points = np.zeros((decoded_gray_image.shape[0], decoded_gray_image.shape[1], 3))
#     rt_matrix = np.column_stack((rotation_matrix, translation_vector)) #-RT matrix
    
#     # loop through every pixel in decoded_gray_image
#     for i in range(decoded_gray_image.shape[0]):
#         for j in range(decoded_gray_image.shape[1]):
#             proj_col = decoded_gray_image[i, j]
#             if proj_col < 0:  # If the value is negative, it's invalid
#                 continue

#             # Create a 3D point in the projector space for the given column at an arbitrary depth (homogeneous depth=1 (third array value))
#             projector_point = np.array([proj_col, 0, 1])

#             # Convert to Heterogeneous points
#             projector_point_3D = np.linalg.inv(projector_matrix) @ projector_point #Proj_max^-1 * projector_point
#             projector_point_3D /= projector_point_3D[2]  # Normalize to ensure the Z coordinate is 1
#             projector_point_in_camera_coords = rt_matrix @ np.append(projector_point_3D[:3], 1)

#             plane_normal = projector_point_in_camera_coords[:3]  # Calculate the normal of the plane in the camera coordinate system

#             # Unproject the camera point to a ray in 3D space
#             camera_ray = inv_camera_matrix @ np.array([j, i, 1])
#             camera_ray /= camera_ray[2]  # Normalize to Z=1

#             # Calculate the ray-plane intersection
#             # Plane equation: plane_normal . (X - projector_point_in_camera_coords) = 0
#             # Ray equation: camera_origin + t * camera_ray
#             # Solve for t: t = plane_normal . (projector_point_in_camera_coords - camera_origin) / (plane_normal . camera_ray)
#             camera_origin = np.zeros(3)  # Set camera equal to origin
#             numerator = np.dot(plane_normal, (projector_point_in_camera_coords[:3] - camera_origin))
#             denominator = np.dot(plane_normal, camera_ray[:3])
#             if denominator == 0:
#                 continue  # Avoid division by zero; ray is invalid or parallel to the plane

#             t = numerator / denominator
#             intersection_point = camera_origin + t * camera_ray[:3]

#             intersection_points[i, j, :] = intersection_point

#     return intersection_points



def ray_plane_intersection(decoded_gray_image, camera_matrix, projector_matrix, rotation_matrix, translation_vector):
    # Make sure the translation vector is a column vector
    if translation_vector.ndim == 1 or translation_vector.shape[0] == 1:
        translation_vector = translation_vector.reshape(3, 1)
    height, width = decoded_gray_image.shape[:2]
    
    # Generate a grid of (u, v) coordinates
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()
    proj_cols = decoded_gray_image.flatten()
    
    # Filter out invalid column indices
    valid_cols = proj_cols >= 0
    u_valid = u[valid_cols]
    v_valid = v[valid_cols]
    proj_cols_valid = proj_cols[valid_cols]
    
    # Normalize pixel coordinates to camera space
    camera_points_normalized = np.linalg.inv(camera_matrix).dot(
        np.vstack((u_valid, v_valid, np.ones_like(u_valid)))
    )
    
    # Projector points in homogeneous coordinates
    projector_points_homogeneous = np.vstack((proj_cols_valid, np.zeros_like(proj_cols_valid), np.ones_like(proj_cols_valid)))
    # Convert projector points to 3D space in camera coordinates
    projector_points_3D = np.linalg.inv(projector_matrix).dot(projector_points_homogeneous)
    projector_points_3D /= projector_points_3D[2, :]  # Normalize z to 1
    
    # Concatenate rotation matrix and translation vector to form the extrinsic matrix
    extrinsic_matrix = np.hstack((rotation_matrix, translation_vector))
    
    # Transform projector points to camera coordinate system
    projector_points_camera_coords = extrinsic_matrix.dot(
        np.vstack((projector_points_3D[:3, :], np.ones(projector_points_3D.shape[1])))
    )
    # Assuming the planes are perpendicular to the projector's y-axis
    plane_normal_projector = np.array([0, 1, 0])  # Normal along Y-axis for vertical stripes
    plane_normal_camera = rotation_matrix.dot(plane_normal_projector)
    # Ray directions for each camera point
    ray_directions = camera_points_normalized[:3, :] - np.zeros((3, 1))  # Camera origin is (0, 0, 0)

    # Calculate ray-plane intersections
    dot_normals = plane_normal_camera.T.dot(ray_directions)
    valid_rays = dot_normals != 0  # Avoid division by zero
    # Calculating intersection 't' for each ray
    t = np.zeros(dot_normals.shape)
    t[valid_rays] = (plane_normal_camera.T.dot(projector_points_camera_coords[:3, :] - translation_vector)) / dot_normals[valid_rays]

    # Intersection points in camera coordinates
    intersection_points = ray_directions * t + np.zeros((3, 1))  # Adding camera origin
    # Prepare the output array
    points_3D = np.zeros((height * width, 3))
    points_3D[valid_cols, :] = intersection_points.T  # Transpose to match the shape
    points_3D = points_3D.reshape((height, width, 3))
    
    return points_3D




def undistortPoints(src,dst,cameraMatrix,distCoeffs):
    """
    :param src: Input Distorted Image.
    :param dst: Output Corrected image with same size and type as src.
    :param CameraMatrix: Intrinsic camera Matrix K
    :param distCoeffs: distortion coefficients (k1,k2,p1,p2,[k3]) of 4,5, or 8 elements.
    :return: undistorted image.
    """
    image = cv.undistort(src,dst,cameraMatrix,distCoeffs)
    return image
    

def visualize_point_cloud(points_3D):
    points = points_3D.reshape(-1, 3) # Convert points from 3D matrix to list of 3D coordinates
    points = points[np.any(points != [0, 0, 0], axis=1)] # Filter out zero points

    # Convert the points to an open3d point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])

    
# def createMesh(points,radius=1.0,iterations=100,confidence=0.9): ##Might want to use open3D instead
#     """
#         :param points: 3D points
#         :param radius: Optional, default=1
#         :param iterations: Optional, default=100
#         :param confidence: Optional, default=0.9
#     """
#     # Compute the normals for the point cloud
#     normals = cv.computeNormals(points)
#     # Create the surface mesh
#     mesh = cv.surfaceReconstruction(points, normals, radius, iterations, confidence)

    
def generate_gray_code_patterns(width, height):
    """
    :param width: Projector Width
    :param Height: Projector Height
    """
    # Calculate number of bits required to represent the width
    num_bits = math.ceil(math.log2(width))
    # Calculate the offset to center the pattern
    offset = (2 ** num_bits - width) // 2

    # Initialize pattern storage
    pattern = np.zeros((height, width, num_bits), dtype=np.uint8)
    directory = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\GrayCodedPictures"  # Change to Pi directory
    # Generate binary and Gray code numbers
    binary_numbers = np.array([list(format(i, '0' + str(num_bits) + 'b')) for i in range(2 ** num_bits)], dtype=np.uint8)
    gray_codes = np.bitwise_xor(binary_numbers[:, :-1], binary_numbers[:, 1:]) #XOR bitwise for gray coding
    gray_codes = np.c_[binary_numbers[:, 0], gray_codes]  # Add the first bit back
    
    # Fill in the pattern
    for i in range(num_bits):
        pattern[:, :, i] = np.tile(gray_codes[(np.arange(width) + offset), i].reshape(1, -1), (height, 1))
        filename = "gray_pattern{}.png".format(i)
        cv.imwrite(directory + "\\" + filename, 255*pattern[:,:,i]) #Need to multiply by 255 for openCV to save as white
    blankImage = np.zeros((height,width), dtype=np.uint8)
    fullImage = 255*np.ones((height,width),dtype=np.uint8)
    cv.imwrite(directory + "\\blankImage.png", blankImage)
    cv.imwrite(directory + "\\fullImage.png", fullImage)
        
    return pattern, offset


def convert_gray_code_to_decimal(gray_code_patterns):
    height, width, num_bits = gray_code_patterns.shape
    # num_bits -= 1 ### DELETE THIS AFTER TESTING
    binary_patterns = np.zeros((height, width, num_bits), dtype=np.uint8)
    binary_patterns[:, :, 0] = gray_code_patterns[:, :, 0]
    for i in range(1, num_bits):
        binary_patterns[:, :, i] = np.bitwise_xor(binary_patterns[:, :, i-1], gray_code_patterns[:, :, i])
        
    decimal_values = np.zeros((height, width), dtype=int)
    for i in range(num_bits):
        decimal_values += (2 ** (num_bits - 1 - i)) * binary_patterns[:, :, i]
    decimal_values += 1
    decimal_values -= int(224) # adjust decimal values according to aspect ratio to get columns 1 through 1920 (should be 64 for 1920 by 1080, 224 for 1600 by 1200). 
    return decimal_values 


def pbpthreshold(image,threshold):
    """
    :param image: Input image.
    :param threshold: Threshold image.
    """
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
            flatimage[i] = 1
            
    image = flatimage.reshape(image.shape)
    return image

width = 1600
height = 1200
grayPattern, offsets = generate_gray_code_patterns(width, height)

blankImage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\blankImage.png")//255 #to convert from 0-255 to 0-1
fullImage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\fullImage.png")//255
blankImage = cv.cvtColor(blankImage, cv.COLOR_BGR2GRAY)
fullImage = cv.cvtColor(fullImage,cv.COLOR_BGR2GRAY)
image_array = np.empty((grayPattern.shape[2],fullImage.shape[0],fullImage.shape[1]), dtype=grayPattern.dtype)
avg_thresh = cv.addWeighted(blankImage,0.5,fullImage,0.5,0) #add white and blank images for thresholding and average
avg_thresh = cv.divide(avg_thresh,2)                           #divide to finish averaging (per pixel thresholding)

# Viewing the patterns
# cv2.imshow('Pattern', grayPattern[:,:,0] * 255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

for i in range(grayPattern.shape[2]-1):

    # load image array and pre-process images
    filein = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\{}.png".format(i+1)
    image = cv.imread(filein)//255
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #Convert to black and white based on grascale (average per pixel thresholding)
    image_thresh = pbpthreshold(image_gray,avg_thresh)
    image_array[i] = image_thresh
    captured_patterns = np.transpose(image_array,(1,2,0)) #reorder shape for binary decoding 

# cv.imshow('book',image_array[2])
# cv.waitKey(5000)
# Convert Gray code to decimal
col_values = convert_gray_code_to_decimal(captured_patterns)
print(col_values)
col_values = convert_gray_code_to_decimal(grayPattern)
print(col_values)
decoded_image = convert_gray_code_to_decimal(captured_patterns)


K = np.array([[1642.17076,0,1176.14705],[0, 1642.83775, 714.90826],[0,0,1]])
R = np.array([[1,0,0],[0,1,0],[0,0,1]])
t = np.array([[0],[0],[1]])
P_proj = np.array([[1642.17076,0,0],[0,1642.83775,0],[0,0,1]])

print(col_values.shape,decoded_image.shape)
points_3D = ray_plane_intersection(decoded_image,K,P_proj,R,t)
visualize_point_cloud(points_3D)
