import cv2
import numpy as np

# Set the size of the pattern: 1920x1080 standard HD for projector
width = 1920
height = 1080

# Create the pattern by alternating between black and white stripes
N = 8 #Num pics taken (improves resolution for higher number of pics)
num_stripes = 2**N
stripe_width = int(width / num_stripes)
pattern = np.zeros((height, width), dtype=np.uint8) #Base matrix
for i in range(num_stripes):
    if i % 2 == 0:
        pattern[:, i * stripe_width:(i + 1) * stripe_width] = 255 #takes each stripe and changes the value in the matrix to white 255

# Display the pattern
cv2.imshow("Pattern", pattern)
cv2.waitKey(0)


# Capture an image of the pattern
capture = cv2.VideoCapture(0)
ret, image = capture.read()
capture.release()

# Convert the image to black and white based off grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
thresh = cv2.threshold(gray, 127, 220, cv2.THRESH_BINARY)

# Compute the binary code based on the pattern
#binary = np.zeros((height, width), dtype=np.uint8)
#for i in range(num_stripes):
    #stripe = thresh[:, i * stripe_width:(i + 1) * stripe_width]
    #binary[:, i * stripe_width:(i + 1) * stripe_width] = thresh

# Compute the pixel locations based on the binary code
#locations = np.zeros((height, width, 2), dtype=np.float32)
#for y in range(height):
    #for x in range(width):
        #binary_code = "".join([str(int(binary[y, i * stripe_width + stripe_width // 2] == 255)) for i in range(num_stripes)])
        #locations[y, x] = [int(binary_code, 2), x]