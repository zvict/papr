import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import mahalanobis
import os
import tqdm


def load_point_clouds(file_paths):
    """
    Load a series of point clouds from file paths.

    :param file_paths: List of file paths to the point clouds.
    :return: List of numpy arrays, each representing a point cloud (N, 3).
    """
    return [np.load(file_path) for file_path in file_paths]

def calculate_max_travel_distance(point_clouds):
    """
    Calculate the maximum Euclidean travel distance for each point.

    :param point_clouds: List of point clouds (N, 3) for each frame.
    :return: Array of maximum travel distances for each point (N,).
    """
    first_frame = point_clouds[0]
    max_distances = np.zeros(first_frame.shape[0])
    for frame in point_clouds:
        distances = np.linalg.norm(frame - first_frame, axis=1)
        max_distances = np.maximum(max_distances, distances)
    return max_distances

def calculate_mahalanobis_distance(point_clouds, semantic_indices=None):
    """
    Calculate the Mahalanobis distance for each point in the first frame.

    :param point_clouds: List of point clouds (N, 3) for each frame.
    :param semantic_indices: Dictionary mapping semantic part names to indices in the point cloud.
                             If None, the covariance is calculated for the whole point cloud.
    :return: Dictionary of Mahalanobis distances for each semantic part or the whole point cloud.
    """
    first_frame = point_clouds[0]
    distances = {}

    if semantic_indices is None:
        # Calculate covariance for the whole point cloud
        covariance = np.cov(first_frame, rowvar=False)
        inv_covariance = np.linalg.inv(covariance)
        distances["whole"] = np.array([
            mahalanobis(point, np.mean(first_frame, axis=0), inv_covariance)
            for point in first_frame
        ])
    else:
        # Calculate covariance for each semantic part
        for part_name, indices in semantic_indices.items():
            part_points = first_frame[indices]
            covariance = np.cov(part_points, rowvar=False)
            inv_covariance = np.linalg.inv(covariance)
            distances[part_name] = np.array([
                mahalanobis(point, np.mean(part_points, axis=0), inv_covariance)
                for point in part_points
            ])

    return distances

def plot_point_cloud_with_colors(point_cloud, colors, title):
    """
    Plot a point cloud with colors using Plotly.

    :param point_cloud: Numpy array of shape (N, 3) representing the point cloud.
    :param colors: Array of shape (N,) representing the color values for each point.
    :param title: Title of the plot.
    """
    fig = go.Figure(data=[
        go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                colorscale='Viridis',
                colorbar=dict(title="Distance"),
                opacity=0.8
            )
        )
    ])
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        )
    )
    fig.show()

# Example usage
if __name__ == "__main__":
    
    start = 0
    end = 30001
    interval = 200
    scale = 10.0
    src_pc_dir = "/NAS/spa176/papr-retarget/point_clouds/butterfly/"

    point_clouds = []
    for idx in tqdm.tqdm(range(start, end, interval)):
        src_pc_path = os.path.join(src_pc_dir, f"points_{idx}.npy")
        cur_src_pc = np.load(src_pc_path)
        cur_src_pc = cur_src_pc / scale
        point_clouds.append(cur_src_pc)
    

    # Calculate max travel distance
    max_travel_distances = calculate_max_travel_distance(point_clouds)

    # Plot max travel distance
    plot_point_cloud_with_colors(
        point_clouds[0],
        max_travel_distances,
        title="Max Travel Distance (Euclidean)"
    )

    # # Define semantic part indices (example)
    # semantic_indices = {
    #     "part_1": np.array([0, 1, 2, 3]),  # Indices of points in part 1
    #     "part_2": np.array([4, 5, 6, 7]),  # Indices of points in part 2
    #     # Add more parts as needed
    # }

    # # Calculate Mahalanobis distances
    # mahalanobis_distances = calculate_mahalanobis_distance(point_clouds, semantic_indices)

    # # Plot Mahalanobis distance for each semantic part
    # for part_name, distances in mahalanobis_distances.items():
    #     part_points = point_clouds[0][semantic_indices[part_name]]
    #     plot_point_cloud_with_colors(
    #         part_points,
    #         distances,
    #         title=f"Mahalanobis Distance ({part_name})"
    #     )