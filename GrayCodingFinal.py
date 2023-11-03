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
    # num_bits -= 1 ### DELETE THIS AFTER TESTING
    binary_patterns = np.zeros((height, width, num_bits), dtype=np.uint8)
    binary_patterns[:, :, 0] = gray_code_patterns[:, :, 0]
    for i in range(1, num_bits):
        binary_patterns[:, :, i] = np.bitwise_xor(binary_patterns[:, :, i-1], gray_code_patterns[:, :, i])
        
    decimal_values = np.zeros((height, width), dtype=int)
    for i in range(num_bits):
        decimal_values += (2 ** (num_bits - 1 - i)) * binary_patterns[:, :, i]
    decimal_values += 1
    decimal_values -= int(width/30) # adjust decimal values according to aspect ratio to get columns 1 through 1920. (FOR 1920x1080 ONLY) 
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

blankImage = cv2.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\blankImage.png")//255 #to convert from 0-255 to 0-1
fullImage = cv2.imread("C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\fullImage.png")//255
blankImage = cv2.cvtColor(blankImage, cv2.COLOR_BGR2GRAY)
fullImage = cv2.cvtColor(fullImage,cv2.COLOR_BGR2GRAY)
image_array = np.empty((grayPattern.shape[2],fullImage.shape[0],fullImage.shape[1]), dtype=grayPattern.dtype)
avg_thresh = cv2.addWeighted(blankImage,0.5,fullImage,0.5,0) #add white and blank images for thresholding and average
avg_thresh = cv2.divide(avg_thresh,2)                           #divide to finish averaging (per pixel thresholding)

# Viewing the patterns
# cv2.imshow('Pattern', grayPattern[:,:,0] * 255)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


for i in range(grayPattern.shape[2]-1):

    # load image array and pre-process images
    filein = "C:\\Users\\nludw\\Documents\\Capstone\\Binary Coding\\Testing\\TestImages\\{}.png".format(i+1)
    image = cv2.imread(filein)//255
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Convert to black and white based on grascale (average per pixel thresholding)
    image_thresh = pbpthreshold(image_gray,avg_thresh)
    image_array[i] = image_thresh
    captured_patterns = np.transpose(image_array,(1,2,0)) #reorder shape for binary decoding 

# # Convert Gray code to decimal
col_values = convert_gray_code_to_decimal(captured_patterns)
print(col_values)



