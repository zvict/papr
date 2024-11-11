import numpy as np
import open3d as o3d

# Load point cloud from numpy array
# src_pc_path = "/NAS/spa176/papr-retarget/point_clouds/butterfly/points_0.npy"
src_pc_path = "/NAS/spa176/papr-retarget/point_clouds/hummingbird/points_0.npy"
points = np.load(src_pc_path)

# Create an Open3D point cloud object
point_cloud = o3d.geometry.PointCloud()
point_cloud.points = o3d.utility.Vector3dVector(points)

# Save the point cloud to a PLY file
o3d.io.write_point_cloud("hummingbird_pc_0.ply", point_cloud)
