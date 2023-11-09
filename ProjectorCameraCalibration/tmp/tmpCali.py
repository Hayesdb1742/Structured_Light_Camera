#! /usr/bin/env python3
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np
import json



# Constants
INPUT_DIR = "testCalib"
OUTPUT_DIR = "decoded"
CHESSBOARD_SIZE = (9, 6)

def detect_corners(image_path, w_file):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE, None)
    if not ret:
        print(f"{image_path} not read")
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
        cv2.imshow('img', img)
        fileName=str(w_file)
        print(cv2.imwrite(f"drawnCorners/{fileName}", img))
        cv2.waitKey(500)
        cv2.destroyAllWindows()
        return corners
    else:
        return None

def convert_to_projector_coordinates(camera_corners, decoded_x, decoded_y):
    projector_corners = []
    height, width = decoded_x.shape
    for corner in camera_corners:
        x, y = corner.ravel()
        if 0 <= x < width and 0 <= y < height:  # Check bounds
            proj_x = decoded_x[int(y), int(x)]
            proj_y = decoded_y[int(y), int(x)]
            projector_corners.append([[proj_x, proj_y]])
    projector_corners = np.array(projector_corners, dtype=np.float32)
    return projector_corners

def compute_intrinsics_and_distortion(objpoints, imgpoints, img_shape):
    ret, matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_shape, None, None)
    print("ret:", ret)
    print("mtx:", matrix)
    print("dist:", dist_coeffs)
    print("rvecs:", rvecs)
    print("tvecs:", tvecs)
    return matrix, dist_coeffs, rvecs, tvecs

def perform_stereo_calibration(objpoints, camera_corners, projector_corners, cam_matrix, cam_dist, proj_matrix, proj_dist):
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, camera_corners, projector_corners,
        cam_matrix, cam_dist,
        proj_matrix, proj_dist,
        None
    )
    return R, T

if __name__ == "__main__":
    files = os.listdir(INPUT_DIR)
    print(files)
    white_files = files
    print(white_files)
    all_camera_corners = []
    all_projector_corners = []
    objpoints = []

    for w_file in white_files:
        camera_image_path = os.path.join(INPUT_DIR, w_file)
        print(camera_image_path)
        camera_corners = detect_corners(camera_image_path, w_file)
        if camera_corners is not None:
            # decoded_x_path = os.path.join(OUTPUT_DIR, os.path.splitext(w_file)[0] + '_x.exr')
            # decoded_y_path = os.path.join(OUTPUT_DIR, os.path.splitext(w_file)[0] + '_y.exr')

            # decoded_x = cv2.imread(decoded_x_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            # decoded_y = cv2.imread(decoded_y_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            projector_corners = convert_to_projector_coordinates(camera_corners, decoded_x, decoded_y)

            # if len(camera_corners) != len(projector_corners):
                # print(f"Skipping {w_file} due to inconsistent corner detection.")
                # continue

            all_camera_corners.append(camera_corners)
            all_projector_corners.append(projector_corners)

            objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
            objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
            objpoints.append(objp)
            
        else:
            print("cannot find corners")

    img_shape = cv2.imread(os.path.join(INPUT_DIR, white_files[0]), cv2.IMREAD_GRAYSCALE).shape[::-1]
    cv2.destroyAllWindows()
    print("Camera intrinsic parameters:")
    cam_matrix, cam_dist, rvecs, tvecs = compute_intrinsics_and_distortion(objpoints, all_camera_corners, img_shape)
    print("\nProjector intrinsic parameters:")
    proj_matrix, proj_dist = compute_intrinsics_and_distortion(objpoints, all_projector_corners, img_shape)

    #R, T = perform_stereo_calibration(objpoints, all_camera_corners, all_projector_corners, cam_matrix, cam_dist, proj_matrix, proj_dist)

    #data = {"cameraMatrix": cam_matrix, "cameraDistortion": cam_dist, "projectionMatrix": proj_matrix, "rotationMatrix": R, "translationMatrix": T}
    #np.savez('data.npz', array1=cam_matrix, array2=cam_dist, array3=proj_matrix, array4=proj_dist, array5=R, array6=T)
    


    print("\nSummary:")
    print(f"Camera Matrix:\n{cam_matrix}")
    print(f"Camera Distortion Coefficients:\n{cam_dist}")
    
    #print(f"Projector Matrix:\n{proj_matrix}")
    #print(f"Projector Distortion Coefficients:\n{proj_dist}")
    #print(f"Rotation Matrix (R):\n{R}")
    #print(f"Translation Vector (T):\n{T}")


    print("Gathering new image")
    img =cv2.imread("/home/lightwork/scripts/Structured_Light_Camera/ProjectorCameraCalibration/tempCalib/testCalib/test_image.jpg")
    h, w = img.shape[:2]
    newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(cam_matrix, cam_dist, (w,h), 1,(w,h))

    dst= cv2.undistort(img, cam_matrix, cam_dist, None, newcameramatrix)
    
    x,y,w,h =roi
    dst = dst[y:y+h,x:x+w]
    
    cv2.imwrite('calibresult.jpg', dst)
    
    
    mean_error=0
    
    for i in range(len(objpoints)):
        imgpoints2,_ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], cam_matrix, cam_dist)
        error = cv2.norm(all_camera_corners[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error+= error
    
    print("total_error:{}".format(mean_error/len(objpoints)))
    
    



