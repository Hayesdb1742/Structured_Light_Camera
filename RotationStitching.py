import open3d as o3d
import numpy as np

def load_csv(csv_file):
    # Load points from CSV file
    points = np.loadtxt(csv_file, delimiter=",", skiprows=1)  

    # Filter out NaN and infinity values
    points = points[~np.isnan(points).any(axis=1)]
    points = points[~np.isinf(points).any(axis=1)]

    # Create a point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    return point_cloud

def stitch_pcds(directory, rotation_axis='y', degrees_per_capture=30):
    """
    Stitch together point clouds of an object rotating in front of a stationary camera.
    :param directory: Directory where the point cloud files are stored.
    :param rotation_axis: Axis of rotation ('x', 'y', or 'z').
    :param degrees_per_capture: MUST BE CLOCK WISE FOR Y. Degrees rotated between each capture. 
    :return: Stitched point cloud.
    """
    axis_dict = {'x': 0, 'y': 1, 'z': 2}
    
    if rotation_axis not in axis_dict:
        raise ValueError("rotation_axis must be 'x', 'y', or 'z'")
    
    num_files = int(360 / degrees_per_capture)
    
    # Load point cloud files
    o3d_point_clouds = []
    for i in range(num_files):
        filename = directory + "/view_" + str(i)
        point_cloud = load_csv(filename)
        o3d_point_clouds.append(point_cloud)
    
    # Stitch point clouds
    stitched_pc = o3d_point_clouds[0]
    for i in range(1, len(o3d_point_clouds)):
        # Define the rotation matrix
        angle = np.radians((i-1)*degrees_per_capture) # Makes view 1 Primary Viewing angle
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(angle * np.eye(3)[:, axis_dict[rotation_axis]])
        # Apply rotation
        o3d_point_clouds[i].transform(rotation_matrix)
        # Merge
        stitched_pc += o3d_point_clouds[i]
    
    # Cleaning
    cl, ind = stitched_pc.remove_radius_outlier(nb_points=16, radius=0.05)
    cleaned_pc = stitched_pc.select_by_index(ind)
    
    return cleaned_pc



