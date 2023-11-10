import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np


# Constants
INPUT_DIR = "testCalib"
OUTPUT_DIR = "decoded"
CHESSBOARD_SIZE = (9, 6)

def detect_corners(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
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
    files = sorted(os.listdir(INPUT_DIR), key=lambda x: os.path.getctime(os.path.join(INPUT_DIR, x)))
    print(files)
    white_files = files[::3]

    all_camera_corners = []
    all_projector_corners = []
    objpoints = []

    for w_file in white_files:
        camera_image_path = os.path.join(INPUT_DIR, w_file)
        camera_corners = detect_corners(camera_image_path)
        if camera_corners is not None:
            decoded_x_path = os.path.join(OUTPUT_DIR, os.path.splitext(w_file)[0] + '_x.exr')
            decoded_y_path = os.path.join(OUTPUT_DIR, os.path.splitext(w_file)[0] + '_y.exr')

            decoded_x = cv2.imread(decoded_x_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            decoded_y = cv2.imread(decoded_y_path, cv2.IMREAD_UNCHANGED).astype(np.float32)

            projector_corners = convert_to_projector_coordinates(camera_corners, decoded_x, decoded_y)

            if len(camera_corners) != len(projector_corners):
                print(f"Skipping {w_file} due to inconsistent corner detection.")
                continue

            all_camera_corners.append(camera_corners)
            all_projector_corners.append(projector_corners)

            objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
            objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
            objpoints.append(objp)

        else:
            print("Chessboard corners not found")

    img_shape = cv2.imread(os.path.join(INPUT_DIR, white_files[0]), cv2.IMREAD_GRAYSCALE).shape[::-1]
    print("Camera intrinsic parameters:")
    print(all_camera_corners)
    cam_matrix, cam_dist, c_rvecs, c_tvecs = compute_intrinsics_and_distortion(objpoints, all_camera_corners, img_shape)
    print("\nProjector intrinsic parameters:")
    print(all_projector_corners)
    proj_matrix, proj_dist, p_rvecs, p_tvecs = compute_intrinsics_and_distortion(objpoints, all_projector_corners, img_shape)

    R, T = perform_stereo_calibration(objpoints, all_camera_corners, all_projector_corners, cam_matrix, cam_dist, proj_matrix, proj_dist)

    print("\nSummary:")
    print(f"Camera Matrix:\n{cam_matrix}")
    print(f"Camera Distortion Coefficients:\n{cam_dist}")
    print(f"Projector Matrix:\n{proj_matrix}")
    print(f"Projector Distortion Coefficients:\n{proj_dist}")
    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (T):\n{T}")

    
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], p_rvecs[i], p_tvecs[i], proj_matrix, proj_dist)
        error = cv2.norm(all_projector_corners[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total projection error: {}".format(mean_error/len(objpoints)) )

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], c_rvecs[i], c_tvecs[i], cam_matrix, cam_dist)
        error = cv2.norm(all_camera_corners[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        mean_error += error
    print( "total camera error: {}".format(mean_error/len(objpoints)) )
