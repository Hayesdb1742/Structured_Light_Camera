import numpy as np
import cv2 as cv
import math
import open3d as o3d

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
    decimal_values -= int(64) # adjust decimal values according to aspect ratio to get columns 1 through 1920 (should be 64 for 1920 by 1080, 224 for 1600 by 1200). 
    return decimal_values 

def ray_to_plane_intersection(u, cam_inv, R, T, col_val, proj_inv):
    ## CAMERA
    u_vec =  np.append(u,1) ## Camera point (u,v,1)
  
    T= np.reshape(T,-1) #Translation Vector
    qc = np.array([0,0,0]) #Set camera pinhole at origin
    rc = (cam_inv).dot(u_vec) #Camera Ray
    
    ## PROJECTOR
    p_proj = np.array([col_val, 0, 1])
    rp = (proj_inv).dot(p_proj) #Projector Ray
    up = np.array([0,-1,0])
    
    n = np.cross(rp,up) #Plane normal for each projector ray
    qp = -T + qc #Center of projection for projector
    # change to camera coordinate system:
    
    n = R.dot(n)
   
    nc = n[0:3]
    qpc = qp[0:3] #center of projection from camera persective
    
    ## Intersection in terms of Camera Coord
    lambda_val = (nc.T).dot(qpc - qc) / nc.T.dot(rc) 
    
    intersection_point = qc + lambda_val * rc 
    
    return intersection_point

def calculate_3D_points(decoded_image, camera_matrix, R, T, proj_matrix):
    height, width = decoded_image.shape
    points_3D = []  # Initialize as an empty list

    cam_inv = np.linalg.inv(camera_matrix)
    proj_inv = np.linalg.inv(camera_matrix)
    
    # Iterate over each pixel in the decoded image
    for i in range(height):
        for j in range(width):
            projector_column = decoded_image[i, j]
            if np.isnan(projector_column) or projector_column < 0:
                continue  # Skip invalid points

            # Image point in pixel coordinates
            u = np.array([j, i])  # 2D pixel coordinates

            # Calculate the intersection point
            intersection_point = ray_to_plane_intersection(u, cam_inv, R, T, projector_column, proj_inv)
            points_3D.append(intersection_point)

    return points_3D

def visualize_point_cloud(points_3D):
    # Convert points_3D to a numpy array if it's not already
    if not isinstance(points_3D, np.ndarray):
        points_3D = np.array(points_3D)

    # Ensure the array shape is correct (n, 3)
    if points_3D.ndim != 2 or points_3D.shape[1] != 3:
        raise ValueError("points_3D should be of shape (n, 3)")

    # Remove any points that are NaN or inf
    points_3D = points_3D[~np.isnan(points_3D).any(axis=1)]
    points_3D = points_3D[~np.isinf(points_3D).any(axis=1)]

    # Check if the points_3D is not empty
    if points_3D.size == 0:
        raise ValueError("points_3D is empty")

    # Convert the points to an open3d point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3D)
    
    o3d.visualization.draw_geometries([point_cloud])

def overlay_mask_with_threshold(image1, image2, threshold=10): 
    ## Create an image mask from the full Image to block out shadows and non projected regions
    diff = cv.absdiff(image1, image2)
    gray_diff = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
    # Create a mask where the difference is below the threshold (similar pixels)
    _, mask = cv.threshold(gray_diff, threshold, 255, cv.THRESH_BINARY_INV)
    
    # Create a red overlay ## FOR Testing and display purposes, verify Mask covers shadoweed and non-projected regions
    # red_overlay = np.zeros_like(image1, image1.dtype)
    # red_overlay[:, :] = [0, 0, 255]  # Red color

    # # Apply the mask to the red overlay
    # red_mask = cv.bitwise_and(red_overlay, red_overlay, mask=mask)
    # # Overlay the red mask on the original image
    # overlay_result = cv.addWeighted(image1, 1, red_mask, 1, 0)

    # # # Display the result
    # # cv.imshow('Red Overlay on Image with Threshold', overlay_result)
    # # cv.waitKey(0)
    # # cv.destroyAllWindows()

    return mask

def apply_mask(image,mask):
    masked_image = image.astype(float)
    masked_image[mask==255] = np.nan
    
    return masked_image
    

width = 1600
height = 1200
grayPattern, offsets = generate_gray_code_patterns(width, height)

blankImage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\blankImage.png")
fullImage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\fullImage.png") 

blankImage = cv.cvtColor(blankImage, cv.COLOR_BGR2GRAY)/255
fullImage = cv.cvtColor(fullImage,cv.COLOR_BGR2GRAY)/255
image_array = np.empty((grayPattern.shape[2],fullImage.shape[0],fullImage.shape[1]), dtype=grayPattern.dtype)
avg_thresh = cv.addWeighted(blankImage,0.5,fullImage,0.5,0) #add white and blank images for thresholding and average


for i in range(grayPattern.shape[2]-1):

    # load image array and pre-process images
    filein = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\{}.png".format(i+1)
    image = cv.imread(filein)
    image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)/255
    #Convert to black and white based on grascale (average per pixel thresholding)
    image_thresh = pbpthreshold(image_gray,avg_thresh)
    image_array[i] = image_thresh
    captured_patterns = np.transpose(image_array,(1,2,0)) #reorder shape for binary decoding 


decoded_image = convert_gray_code_to_decimal(captured_patterns)
blankImage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\blankImage.png")
fullImage = cv.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\fullImage.png") 
mask = overlay_mask_with_threshold(fullImage, blankImage, threshold=10)

K = np.array([[1642.17076,0,1176.14705],[0, 1642.83775, 714.90826],[0,0,1]])
R = np.array([[1,0,0],[0,1,0],[0,0,1]])
t = np.array([[-50],[0],[0]])
P_proj = np.array([[1642.17076,0,1920/2],[0,1642.83775,1080/2],[0,0,1]])

points_3D = calculate_3D_points(decoded_image, K, R, t, P_proj)

visualize_point_cloud(points_3D)



