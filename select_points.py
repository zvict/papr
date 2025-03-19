# import open3d as o3d
# import numpy as np

# # Load point cloud from numpy array
# src_pc_path = "/NAS/spa176/papr-retarget/point_clouds/butterfly/points_0.npy"
# points = np.load(src_pc_path)
# # points = np.random.rand(1000, 3)  # Example point cloud
# point_cloud = o3d.geometry.PointCloud()
# point_cloud.points = o3d.utility.Vector3dVector(points)

# # Visualize the point cloud
# vis = o3d.visualization.VisualizerWithEditing()
# vis.create_window()
# vis.add_geometry(point_cloud)
# vis.run()  # User selects points in the GUI
# vis.destroy_window()

# # Get the selected point indices
# selected_indices = vis.get_picked_points()
# print("Selected point indices:", selected_indices)

# # Save the selected indices to a file
# np.save("selected_indices.npy", selected_indices)

import numpy as np
from scipy.spatial import KDTree

# Load the original point cloud from the numpy array
src_pc_path = "/NAS/spa176/papr-retarget/point_clouds/butterfly/points_0.npy"
# src_pc_path = "/NAS/spa176/papr-retarget/point_clouds/hummingbird/points_0.npy"
original_points = np.load(src_pc_path)

# Load the subset point cloud from the XYZ file
def load_xyz(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y, z = map(float, line.split())
            points.append([x, y, z])
    return np.array(points)

# original_points = load_xyz("/NAS/spa176/papr-retarget/but_pc_0_wings.xyz")
original_points = load_xyz("/NAS/spa176/papr-retarget/hummingbird_wing_pc.xyz")

# subset_pc_path = "/NAS/spa176/papr-retarget/but_pc_0_wings_right.xyz"
subset_pc_path = "/NAS/spa176/papr-retarget/hummingbird_wing_pc_right.xyz"
# subset_pc_path = "/NAS/spa176/papr-retarget/but_pc_0_wings.xyz"
# subset_pc_path = "/NAS/spa176/papr-retarget/hummingbird_wing_pc.xyz"
subset_points = load_xyz(subset_pc_path)

# Find the indices of the subset points in the original point cloud
kdtree = KDTree(original_points)
_, indices = kdtree.query(subset_points)


print(f"Original point length: {len(original_points)}")
print(f"Length of indices of subset points in the original point cloud: {len(indices)}")

# Save the indices to a file
# np.save("but_wing_indices_right.npy", indices)
np.save("hummingbird_wing_indices_right.npy", indices)
# np.save("but_wing_indices.npy", indices)

# np.save("hummingbird_wing_indices.npy", indices)
