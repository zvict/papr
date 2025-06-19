import numpy as np
import trimesh

# import argparse # Keep commented out as per your original code
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import open3d as o3d


def align_meshes_icp(
    source_mesh_tm,
    target_mesh_tm,
    max_correspondence_distance=0.2,
    initial_transform=np.identity(4),
):
    """
    Aligns a source mesh to a target mesh using ICP.

    Args:
        source_mesh: An open3d.geometry.TriangleMesh object (the mesh to be transformed).
        target_mesh: An open3d.geometry.TriangleMesh object (the reference mesh).
        max_correspondence_distance: The maximum distance between points to be considered correspondences.
        initial_transform: An optional initial transformation matrix (4x4 numpy array).

    Returns:
        A tuple containing the aligned source mesh and the final transformation matrix.
    """
    source_pcd_o3d = o3d.geometry.PointCloud()
    source_pcd_o3d.points = o3d.utility.Vector3dVector(source_mesh_tm.vertices)

    target_pcd_o3d = o3d.geometry.PointCloud()
    target_pcd_o3d.points = o3d.utility.Vector3dVector(target_mesh_tm.vertices)

    # --- Step 4: Run ICP using Open3D ---
    # You can add other ICP parameters here if needed (e.g., max_iterations)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd_o3d,
        target_pcd_o3d,
        max_correspondence_distance,
        initial_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )  # Or TransformationEstimationPointToPlane()

    # --- Step 5: Get the final transformation matrix ---
    final_transform = reg_p2p.transformation

    # --- Step 6: Apply the transformation back to the Trimesh object ---
    # Create a copy to avoid modifying the original input mesh
    aligned_source_mesh_tm = source_mesh_tm.copy()

    # Apply the transformation matrix to the vertices of the trimesh object
    aligned_source_mesh_tm.vertices = trimesh.transform_points(
        aligned_source_mesh_tm.vertices, final_transform
    )

    return aligned_source_mesh_tm, final_transform


def get_per_vertex_distances(file_path1, file_path2, align=False):
    """
    Loads two meshes and calculates the point-wise Euclidean distances
    between corresponding vertices.

    Args:
        file_path1 (str): Path to the first OBJ mesh file.
        file_path2 (str): Path to the second OBJ mesh file.

    Returns:
        np.ndarray: A NumPy array of per-vertex distances, or None if meshes
                    cannot be compared.
    """
    try:
        # Load meshes using trimesh
        mesh1 = trimesh.load(file_path1, process=False)
        mesh2 = trimesh.load(file_path2, process=False)

        # Ensure they are Trimesh objects
        if not isinstance(mesh1, trimesh.Trimesh) or not isinstance(
            mesh2, trimesh.Trimesh
        ):
            print(f"Error: Could not load meshes from {file_path1} or {file_path2}")
            return None

        if align:
            mesh1 = align_meshes_icp(mesh1, mesh2)[0]
            print("Meshes aligned using ICP.")
            # save the aligned mesh1 if needed
            mesh1.export("aligned_mesh1.obj")

        vertices1 = mesh1.vertices
        vertices2 = mesh2.vertices

        # Check if the number of vertices matches
        if vertices1.shape != vertices2.shape:
            print(
                f"Error: Meshes have different number of vertices ({vertices1.shape[0]} vs {vertices2.shape[0]})"
            )
            return None

        # Calculate point-wise Euclidean distances
        distances = np.linalg.norm(vertices1 - vertices2, axis=1)

        return distances

    except Exception as e:
        print(f"An error occurred while getting distances: {e}")
        return None


def color_mesh_by_distances(mesh, distances, colormap="viridis"):
    """
    Colors a mesh's vertices based on corresponding distance values.

    Args:
        mesh (trimesh.Trimesh): The mesh object to color. This object
                                will be modified in-place by setting
                                its vertex_colors attribute.
        distances (np.ndarray): A NumPy array of distances for each vertex.
                                Must have the same length as the number of vertices.
        colormap (str): The matplotlib colormap name to use. 'viridis' (blue=low, yellow=high)
                        or 'plasma'/'inferno' (dark=low, yellow/red=high) are good choices.

    Returns:
        bool: True if coloring was successful, False otherwise.
    """
    if mesh.vertices.shape[0] != distances.shape[0]:
        print(
            "Error: Number of vertices in mesh does not match number of distances for coloring."
        )
        return False

    # Normalize distances to the range [0, 1]
    # Add a small epsilon to avoid division by zero if all distances are identical
    min_dist = np.min(distances)
    max_dist = np.max(distances)

    print(f"Coloring based on distances ranging from {min_dist:.4f} to {max_dist:.4f}")

    if max_dist == min_dist:
        print("All distances are the same. Coloring uniformly (mid-range color).")
        normalized_distances = (
            np.zeros_like(distances) + 0.5
        )  # Assign mid-color if range is 0
    else:
        normalized_distances = (distances - min_dist) / (max_dist - min_dist + 1e-8)

    # Choose a colormap
    try:
        cmap = cm.get_cmap(colormap)
    except ValueError:
        print(f"Warning: Colormap '{colormap}' not found. Using 'viridis'.")
        cmap = cm.get_cmap("viridis")

    # Map normalized distances to colors (RGBA floats [0, 1])
    colors_rgba = cmap(normalized_distances)

    # Convert colors to uint8 [0, 255] RGBA format expected by trimesh
    colors_uint8 = (colors_rgba * 255).astype(np.uint8)

    # Assign colors to the mesh vertices
    mesh.visual.vertex_colors = colors_uint8

    print(f"Mesh colored successfully using '{colormap}' colormap.")
    return True


if __name__ == "__main__":
    # --- File Paths ---
    exp_id = "exp_11"  # Example experiment ID
    align = True  # Set to True if you want to align the meshes using ICP
    # Use your hardcoded paths for demonstration
    mesh1_path = (
        f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/smpl/{exp_id}/deformed_mesh.obj"
    )
    mesh2_path = "/NAS/spa176/skeleton-free-pose-transfer/demo/gt_results_smpl/gt_1.obj"

    # Define the output path for the colored mesh
    output_colored_mesh_path = (
        f"ours_{exp_id}_mesh2_pmd_colored.ply"  # Recommended format for vertex colors
    )

    if align:
        output_colored_mesh_path = output_colored_mesh_path[-4:] + "_aligned.ply"

    # --- Calculate Distances ---
    per_vertex_distances = get_per_vertex_distances(mesh1_path, mesh2_path, align=align)

    if per_vertex_distances is not None:
        # --- Calculate and Print Stats ---
        mean_distance = np.mean(per_vertex_distances)
        max_distance = np.max(per_vertex_distances)
        min_distance = np.min(per_vertex_distances)

        print(f"\n--- PMD Statistics ---")
        print(f"  Mean PMD: {mean_distance:.4f}")
        print(f"  Max distance: {max_distance:.4f}")
        print(f"  Min distance: {min_distance:.4f}")
        print(f"----------------------\n")

        # --- Load Mesh2 for Coloring and Saving ---
        # We load mesh2 again specifically to apply colors for output.
        # This ensures the original mesh object (if used elsewhere) isn't modified
        # and provides a clean object for saving.
        try:
            mesh2_for_output = trimesh.load(mesh2_path, process=False)
        except Exception as e:
            print(f"Error loading mesh2 for coloring/saving: {e}")
            mesh2_for_output = None

        if mesh2_for_output is not None:
            # --- Color the Mesh ---
            # Use 'plasma' or 'inferno' colormap if you want higher values to lean towards red/yellow
            coloring_successful = color_mesh_by_distances(
                mesh2_for_output, per_vertex_distances, colormap="plasma"
            )  # Choose your colormap

            if coloring_successful:
                # --- Save the Colored Mesh ---
                try:
                    # Use .export() method. 'ply' is highly recommended for vertex colors.
                    # 'glb' (binary glTF) or 'gltf' (text glTF) are also good options.
                    # 'obj' support for vertex colors can vary.
                    mesh2_for_output.export(output_colored_mesh_path, file_type="ply")
                    print(
                        f"Colored mesh saved successfully to: {output_colored_mesh_path}"
                    )

                    # --- Optional: Visualize the Colored Mesh ---
                    # Uncomment the line below if you still want to see the mesh popup
                    # mesh2_for_output.show()

                except Exception as e:
                    print(
                        f"Error saving colored mesh to {output_colored_mesh_path}: {e}"
                    )
                    print(
                        "Please ensure the output directory exists and you have write permissions."
                    )
            else:
                print("Coloring failed, skipping save.")
        else:
            print("Mesh2 could not be loaded for output.")

    else:
        print("Could not calculate per-vertex distances. Cannot color or save mesh.")
