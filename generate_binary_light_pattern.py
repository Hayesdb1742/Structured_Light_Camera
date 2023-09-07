import numpy as np
import math
import cv2

# Set the number of stripes
N = 128  # Change this to the desired number of stripes
numStripes = [128,64,32,16,8,4,2,1]

# Create a binary pattern of alternating black and white stripes
pattern = np.zeros((500, 500))
for index, stripes in enumerate(numStripes):
    if stripes <= N:
        for i in range(stripes):
            stripe = np.ones((500, int(500 / stripes))) * 255 * ((-1) ** i)
            pattern[:, i * int(500 / stripes):(i+1) * int(500 / stripes)] = stripe

        # Convert the pattern to 8-bit grayscale and save as an image
        pattern = np.uint8(pattern)
        cv2.imwrite('binary_pattern%s.png' % stripes, pattern)
  