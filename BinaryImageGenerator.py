import cv2
import numpy as np

# Set the size of the pattern
width = 1920
height = 1080

# Create the pattern by alternating between black and white stripes
N = 4
num_stripes = 2**N
stripe_width = int(width / num_stripes)
pattern = np.zeros((height, width), dtype=np.uint8)
for i in range(num_stripes):
    if i % 2 == 0:
        pattern[:, i * stripe_width:(i + 1) * stripe_width] = 255

# Display the pattern
cv2.imshow("Pattern", pattern)
cv2.waitKey(0)
