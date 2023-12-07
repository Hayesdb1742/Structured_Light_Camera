import numpy as np
import cv2 as cv
import math

class SLscan:
    def __init__(self, cam_resolution, proj_resolution, directory):
        """
        :param cam_resolution: TUPLE of camera resolution
        :param proj_resolution: TUPLE of projector resolution
        :param directory: String of desired directory for saving
        """
        cam_w, cam_h = cam_resolution[0],cam_resolution[1]
        proj_w, proj_h = proj_resolution[0],proj_resolution[1]
        
        self.cam_w = cam_w #currently assumes same height and width for camera and projector (1920x1080)
        self.cam_h = cam_h
        self.proj_w = proj_w
        self.proj_h = proj_h
        
        # Calculate number of bits required to represent the width
        self.N = math.ceil(math.log2(self.proj_w)) ##PROPER CALCULATION FOR PROJECTOR RESOLUTION
        
        coeiff = np.load(directory + "/" + "calib.npz") ## Load Calibration Coeiff (Must be saved in directory)
        self.lst = coeiff.files
        self.R = coeiff['R']
        self.t = coeiff['T']
        self.P_proj = coeiff['proj_matrix']
        self.K = coeiff['cam_matrix']
        self.dist_coeff = coeiff['cam_dist']
        self.directory = directory ## SET TO PI Directory
        
    def generate_gray_code_patterns(self):
        num_bits = self.N
        # Calculate the offset to center the pattern
        offset = (2 ** num_bits - self.proj_w) // 2

        # Initialize pattern storage
        pattern = np.zeros((self.proj_h, self.proj_w, num_bits), dtype=np.uint8)
        # Generate binary and Gray code numbers
        binary_numbers = np.array([list(format(i, '0' + str(num_bits) + 'b')) for i in range(2 ** num_bits)], dtype=np.uint8)
        gray_codes = np.bitwise_xor(binary_numbers[:, :-1], binary_numbers[:, 1:]) #XOR bitwise for gray coding
        gray_codes = np.c_[binary_numbers[:, 0], gray_codes]  # Add the first bit back
        
        # Fill in the pattern
        for i in range(num_bits):
            pattern[:, :, i] = np.tile(gray_codes[(np.arange(self.proj_w) + offset), i].reshape(1, -1), (self.proj_h, 1))
            filename = "gray_pattern{}.png".format(i)
            cv.imwrite(self.directory + "/" + filename, 255*pattern[:,:,i]) 
        blankImage = np.zeros((self.proj_h,self.proj_w), dtype=np.uint8)
        fullImage = 255*np.ones((self.proj_h,self.proj_w),dtype=np.uint8)
        cv.imwrite(self.directory + "/blank_image.png", blankImage)
        cv.imwrite(self.directory + "/full_image.png", fullImage)

    def convert_gray_code_to_decimal(self,gray_code_patterns): 
        num_bits, cam_h, cam_w = gray_code_patterns.shape
        
        binary_patterns = np.zeros((num_bits, cam_h, cam_w), dtype=np.uint8)
        binary_patterns[0, :, :] = gray_code_patterns[0, :, :]
        for i in range(1, num_bits):
            binary_patterns[i, :, :] = np.bitwise_xor(binary_patterns[i-1, :, :], gray_code_patterns[i, :, :])
            
        decimal_values = np.zeros((cam_h, cam_w), dtype=int)
        for i in range(num_bits):
            decimal_values += (2 ** (num_bits - 1 - i)) * binary_patterns[i, :, :]
        
        correct_offset = ((2**num_bits) - self.proj_w)/2 #adjust for centering offset, changes based on width
        decimal_values -= int(correct_offset) # adjust decimal values according to above correction to get columns 0 through 1919 (should be 64 for 1920 by 1080, 224 for 1600 by 1200). 

        return decimal_values 

    def undistort(self, input):
        """
        :param src: Input Distorted Image.
        :param dst: Output Corrected image with same size and type as src.
        :param CameraMatrix: Intrinsic camera Matrix K
        :param distCoeffs: distortion coefficients (k1,k2,p1,p2,[k3]) of 4,5, or 8 elements.
        :return: undistorted image.
        """
      
        image = cv.undistort(input,self.K,self.dist_coeff)
        return image

    def pbpthreshold(self, image,avg_thresh):
        """
        :param image: Input image.
        :param threshold: Threshold image.
        """
        threshold = avg_thresh
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
    
    def apply_mask(self, image, mask):
        masked_image = image.astype(float)
        masked_image[mask==255] = np.nan
        return masked_image
    
    def shadow_mask(self, threshold=10, display=False): 
        
        ## Create an image mask from the full Image to block out shadows and non projected regions
        diff = cv.absdiff(self.fullImage, self.blankImage)
        
        # Create a mask where the difference is below the threshold (similar pixels)
        _, mask = cv.threshold(diff, threshold, 255, cv.THRESH_BINARY_INV)
        
        if display:   
            # Create a red overlay ## FOR Testing and display purposes, verify Mask covers shadoweed and non-projected regions
            full_color = cv.cvtColor(self.fullImage, cv.COLOR_GRAY2BGR)
            red_overlay = np.zeros_like(full_color)
            red_overlay[:, :] = [0, 0, 255]  # Red color

            # Apply the mask to the red overlay
            red_mask = cv.bitwise_and(red_overlay, red_overlay, mask=mask)
            # Overlay the red mask on the original image
            overlay_result = cv.addWeighted(full_color, 1, red_mask, 1, 0)

            # # Display the result
            cv.imshow('SL Mask', overlay_result)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return mask

    def load_images(self):
        #Open cv does everything in dtype uint8, all processing done in uint8 to avoid switching back and forth wasting time/mem
        blankImage = cv.imread(self.directory+"/"+"blank_image.png") # all black projection image capture
        blankImage = self.undistort(blankImage)
        fullImage = cv.imread(self.directory+"/"+"full_image.png") # all white projection image capture       
        fullImage = self.undistort(fullImage)
        self.blankImage = cv.cvtColor(blankImage, cv.COLOR_BGR2GRAY)
        self.fullImage =  cv.cvtColor(fullImage,cv.COLOR_BGR2GRAY)
        avg_thresh = cv.addWeighted(self.blankImage,0.5,self.fullImage,0.5,0)
        image_array = np.empty((self.N,self.cam_h,self.cam_w),dtype=np.uint8)
    
        for i in range(self.N):
            filein = "image_{}.png".format(i)
            image = cv.imread(self.directory+"/"+filein)
            image = self.undistort(image) 
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            image_thresh = self.pbpthreshold(image_gray,avg_thresh)
            image_array[i] = image_thresh
            captured_patterns = np.array(image_array)
        
        return captured_patterns
            
    def preprocess(self,captured_patterns,threshold=50, display=False):
        
        decoded_image = self.convert_gray_code_to_decimal(captured_patterns)
        mask = self.shadow_mask(threshold,display)
        masked_col_values = self.apply_mask(decoded_image,mask)
        return masked_col_values

    def ray_to_plane_intersection(self,u, proj_col):
        ## CAMERA
        u_vec =  np.append(u,1) ## Camera point (u,v,1)
    
        t = np.reshape(self.t,-1) #Translation Vector
        qc = np.array([0,0,0]) #Set camera pinhole at origin
        rc = (self.cam_inv).dot(u_vec) #Camera Ray
        
        ## PROJECTOR
        p_proj = np.array([proj_col, 0, 1])
        rp = (self.proj_inv).dot(p_proj) #Projector Ray
        up = np.array([0,-1,0])
        
        n = np.cross(rp,up) #Plane normal for each projector ray
        qpc = -t + qc #center of projection from camera persective
        # change to camera coordinate system:
        
        nc = self.R.dot(n)
        
        ## Intersection in terms of Camera Coord
        lambda_val = (nc.T).dot(qpc - qc) / nc.T.dot(rc) 
        
        intersection_point = qc + lambda_val * rc 
    
        return intersection_point

    def calculate_3D_points(self,decoded_image):
        """
        :param decoded_image: Matrix containing integer numbers of the column values from gray code decoding. 
        """
        height, width = decoded_image.shape
        points_3D = []  # Initialize as an empty list

        self.cam_inv = np.linalg.inv(self.K)
        self.proj_inv = np.linalg.inv(self.P_proj)
        
        # Iterate over each pixel in the decoded image
        for i in range(height):
            for j in range(width):
                proj_col = decoded_image[i, j]
                if np.isnan(proj_col) or proj_col < 0:
                    continue  # Skip invalid points

                # Image point in pixel coordinates
                u = np.array([j, i])  # 2D pixel coordinates

                # Calculate the intersection point
                intersection_point = self.ray_to_plane_intersection(u, proj_col)
                points_3D.append(intersection_point)
        
        # Convert points_3D to a numpy array
        self.points_3D = np.array(points_3D)
        
    def save(self,view=0):
        """
        :param view: Int value of view number.
        """
        filename = self.directory + "/" + "view_" + str(view) + ".csv"
        np.savetxt(filename, self.points_3D, delimiter=",", header="x,y,z", comments="")
        
        
# ## EXAMPLE USAGE FOR ONE VIEW:
directory = "grayCodePics"
cam_resolution = (1920,1080)
proj_resolution = (1920,1080)
scanner = SLscan(cam_resolution,proj_resolution,directory)
# scanner.generate_gray_code_patterns()
## PROJECT AND TAKE PICTURES HERE ##

captured_patterns = scanner.load_images()
decoded = scanner.preprocess(captured_patterns,threshold=70)
scanner.calculate_3D_points(decoded)
scanner.save(view=0)
