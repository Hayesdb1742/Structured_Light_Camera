import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
import numpy as np


# Constants
INPUT_DIR = "TestImages"
OUTPUT_DIR = "decoded"
CHESSBOARD_SIZE = (8, 5)

# 1. Detect corner points on the camera image
def detect_corners(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    ret, corners = cv2.findChessboardCorners(img, CHESSBOARD_SIZE, None)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
        return corners
    else:
        return None

# 2. Convert to projector image coordinates
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


# 3. Stereo calibration
def perform_stereo_calibration(objpoints, camera_corners, projector_corners):
    # Initial guesses for intrinsic matrices and distortion coefficients
    cam_matrix_init = np.eye(3)
    cam_dist_init = np.zeros((5,))
    proj_matrix_init = np.eye(3)
    proj_dist_init = np.zeros((5,))

    ret, camera_matrix, dist_coeffs, projector_matrix, proj_dist_coeffs, R, T, E, F = cv2.stereoCalibrate(
        objpoints, camera_corners, projector_corners,
        cam_matrix_init, cam_dist_init,
        proj_matrix_init, proj_dist_init,
        None, flags=cv2.CALIB_FIX_INTRINSIC
    )
    return camera_matrix, dist_coeffs, projector_matrix, proj_dist_coeffs, R, T


# Main
if __name__ == "__main__":
    files = os.listdir(INPUT_DIR)
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

            # Check for consistent number of corners
            if len(camera_corners) != len(projector_corners):
                print(f"Skipping {w_file} due to inconsistent corner detection.")
                continue

            all_camera_corners.append(camera_corners)
            all_projector_corners.append(projector_corners)

            objp = np.zeros((np.prod(CHESSBOARD_SIZE), 3), np.float32)
            objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
            objpoints.append(objp)

    cam_matrix, cam_dist, proj_matrix, proj_dist, R, T = perform_stereo_calibration(objpoints, all_camera_corners,
                                                                                    all_projector_corners)

    print(f"Camera Matrix:\n{cam_matrix}")
    print(f"Camera Distortion Coefficients:\n{cam_dist}")
    print(f"Projector Matrix:\n{proj_matrix}")
    print(f"Projector Distortion Coefficients:\n{proj_dist}")
    print(f"Rotation Matrix (R):\n{R}")
    print(f"Translation Vector (T):\n{T}")