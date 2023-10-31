import numpy as np
import cv2
import math

def generate_gray_code_patterns(width, height):
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
        cv2.imwrite(directory + "\\" + filename, 255*pattern[:,:,i]) #Need to multiply by 255 for openCV to save as white
    blankImage = np.zeros((height,width), dtype=np.uint8)
    fullImage = 255*np.ones((height,width),dtype=np.uint8)
    cv2.imwrite(directory + "\\blankImage.png", blankImage)
    cv2.imwrite(directory + "\\fullImage.png", fullImage)
        
    return pattern, offset


def convert_gray_code_to_decimal(gray_code_patterns):
    height, width, num_bits = gray_code_patterns.shape
    
    binary_patterns = np.zeros((height, width, num_bits), dtype=np.uint8)
    binary_patterns[:, :, 0] = gray_code_patterns[:, :, 0]
    for i in range(1, num_bits):
        binary_patterns[:, :, i] = np.bitwise_xor(binary_patterns[:, :, i-1], gray_code_patterns[:, :, i])
        
    decimal_values = np.zeros((height, width), dtype=int)
    for i in range(num_bits):
        decimal_values += (2 ** (num_bits - 1 - i)) * binary_patterns[:, :, i]
    decimal_values += 1
    decimal_values -= int(width/30) # adjust decimal values according to aspect ratio to get columns 1 through 1920. 
    return decimal_values 


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
            flatimage[i] = 1
            
    image = flatimage.reshape(image.shape)
    return image

width = 1920
height = 1080


grayPattern, offsets = generate_gray_code_patterns(width, height)

blankImage = cv2.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\GrayCodedPictures\\blankImage.png")//255 #to convert from 0-255 to 0-1
fullImage = cv2.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\GrayCodedPictures\\fullImage.png")//255
blankImage = cv2.cvtColor(blankImage, cv2.COLOR_BGR2GRAY)
fullImage = cv2.cvtColor(fullImage,cv2.COLOR_BGR2GRAY)
image_array = np.empty((grayPattern.shape[2],fullImage.shape[0],fullImage.shape[1]), dtype=grayPattern.dtype)
avg_thresh = cv2.addWeighted(blankImage,0.5,fullImage,0.5,0) #add white and blank images for thresholding and average
avg_thresh = cv2.divide(avg_thresh,2)                           #divide to finish averaging (per pixel thresholding)

# Viewing the patterns
# cv2.imshow('Pattern', grayPattern[:,:,0] * 255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


for i in range(grayPattern.shape[2]):
    # load image array and pre-process images
    filein = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\GrayCodedPictures\\gray_pattern{}.png".format(i)
    image = cv2.imread(filein)//255
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Convert to black and white based on grascale (average per pixel thresholding)
    image_thresh = pbpthreshold(image_gray,avg_thresh)
    image_array[i] = image_thresh
    captured_patterns = np.transpose(image_array,(1,2,0)) #reorder shape for binary decoding 

# Convert Gray code to decimal
col_values = convert_gray_code_to_decimal(captured_patterns)
print(col_values)

# def undistort_points(x, y, k, p): #Where k and p are distortion coefficients
#     # Initializing variables
#     x_undistorted = x.copy()
#     y_undistorted = y.copy()
    
#     # Iterative undistortion as direct undistortion is non-trivial
#     for _ in range(5):  # Iterating 5 times usually suffices
#         r2 = x_undistorted**2 + y_undistorted**2
#         r4 = r2**2
#         r6 = r2**3
#         radial_distortion = 1 + k[0]*r2 + k[1]*r4 + k[2]*r6
        
#         dx = 2*p[0]*x_undistorted*y_undistorted + p[1]*(r2 + 2*x_undistorted**2)
#         dy = p[0]*(r2 + 2*y_undistorted**2) + 2*p[1]*x_undistorted*y_undistorted
        
#         x_undistorted = (x - dx) / radial_distortion
#         y_undistorted = (y - dy) / radial_distortion
        
#     return x_undistorted, y_undistorted
    
# def line_plane_intersection(cam_intrinsics, cam_rotation, cam_translation, 
#                             proj_intrinsics, proj_rotation, proj_translation, 
#                             image_points, proj_points, distortion_coeffs):
#     # Undistort the image points
#     x_undistorted, y_undistorted = undistort_points(image_points[:, 0], image_points[:, 1], distortion_coeffs[:1,4], distortion_coeffs[2:3])
    
#     # Construct the camera matrix
#     cam_matrix = cam_intrinsics @ np.hstack((cam_rotation, cam_translation))
    
#     # Project the points to normalized image coordinates
#     image_points_normalized = np.linalg.inv(cam_intrinsics) @ np.vstack((x_undistorted, y_undistorted, np.ones_like(x_undistorted)))
    
#     # Construct the projector matrix
#     proj_matrix = proj_intrinsics @ np.hstack((proj_rotation, proj_translation))
    
#     # Project the projector points to world coordinates
#     proj_points_world = np.linalg.inv(proj_matrix) @ np.vstack((proj_points[:, 0], proj_points[:, 1], np.ones_like(proj_points[:, 0]), np.zeros_like(proj_points[:, 0])))
#     proj_points_world /= proj_points_world[3, :]
    
#     # Calculate the line (ray) for each camera point
#     cam_origin = -cam_rotation.T @ cam_translation
#     rays = image_points_normalized - cam_origin[:, None]
    
#     # Normalize the rays
#     rays /= np.linalg.norm(rays, axis=0)
    
#     # Calculate the intersection of each ray with the projector plane
#     plane_normal = np.mean(np.cross(proj_points_world[:, :-1] - proj_points_world[:, 1:], proj_points_world[:, 1:] - proj_points_world[:, 2:], axis=0), axis=0)
#     plane_point = np.mean(proj_points_world[:, :3], axis=0)
    
#     t = ((plane_point - cam_origin) @ plane_normal) / (rays.T @ plane_normal)
#     intersection_points = cam_origin[:, None] + rays * t
    
#     return intersection_points


