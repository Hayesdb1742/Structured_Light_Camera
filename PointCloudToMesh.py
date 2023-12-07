import open3d as o3d

def calculate_average_distance(pcd_file):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Calculate and return the average distance between points
    distances = pcd.compute_nearest_neighbor_distance()
    return sum(distances) / len(distances)

def convert_and_visualize_pcd_to_stl(pcd_file, stl_file, avg_distance):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(pcd_file)

    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # Determine radii based on average distance
    radii = [avg_distance * factor for factor in [0.8, 1, 1.2, 1.4, 2]]

    # Apply Ball Pivoting algorithm
    bpa_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))

    # Save mesh as STL
    o3d.io.write_triangle_mesh(stl_file, bpa_mesh)

    # Visualize the mesh
    o3d.visualization.draw_geometries([bpa_mesh])

# File paths
pcd_file_path = 'bunny.pcd'
stl_file_path = 'output_mesh.stl'

# Calculate average distance and use it for mesh generation
avg_distance = calculate_average_distance(pcd_file_path)
convert_and_visualize_pcd_to_stl(pcd_file_path, stl_file_path, avg_distance)
