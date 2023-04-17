import numpy as np
import cv2

# Set the number of stripes
N = 128  # Change this to the desired number of stripes

# Create a binary pattern of alternating black and white stripes
pattern = np.zeros((500, 500))
for i in range(N):
    stripe = np.ones((500, int(500 / N))) * 255 * ((-1) ** i)
    pattern[:, i * int(500 / N):(i+1) * int(500 / N)] = stripe

# Convert the pattern to 8-bit grayscale and save as an image
pattern = np.uint8(pattern)
cv2.imwrite('binary_pattern1.png', pattern)
