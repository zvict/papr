import numpy as np
import trimesh
import argparse


def calculate_pmd(file_path1, file_path2):
    """
    Calculates the mean Point-wise Mesh Euclidean Distance (PMD)
    between two OBJ meshes with matched vertices.

    Args:
        file_path1 (str): Path to the first OBJ mesh file.
        file_path2 (str): Path to the second OBJ mesh file.

    Returns:
        float: The mean PMD, or None if meshes cannot be compared.
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

        vertices1 = mesh1.vertices
        vertices2 = mesh2.vertices

        # print(f"Mesh 1 vertices: {vertices1.shape}")
        # Check if the number of vertices matches
        if vertices1.shape != vertices2.shape:
            print(
                f"Error: Meshes have different number of vertices ({vertices1.shape[0]} vs {vertices2.shape[0]})"
            )
            return None

        # Calculate point-wise Euclidean distances
        distances = np.linalg.norm(vertices1 - vertices2, axis=1)

        # Calculate the mean distance
        mean_pmd = np.mean(distances)

        # # print the max distance
        # max_distance = np.max(distances)
        # print(f"Max distance: {max_distance}")
        # # print the min distance
        # min_distance = np.min(distances)
        # print(f"Min distance: {min_distance}")

        return mean_pmd

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(
    #     description="Calculate Mean Point-wise Mesh Euclidean Distance (PMD) between two OBJ files."
    # )
    # parser.add_argument("mesh1", help="Path to the first OBJ mesh file.")
    # parser.add_argument("mesh2", help="Path to the second OBJ mesh file.")

    # args = parser.parse_args()

    # mean_distance = calculate_pmd(args.mesh1, args.mesh2)

    mesh1 = "/NAS/spa176/skeleton-free-pose-transfer/demo/results_smpl/dst1_1.obj"
    mesh2 = "/NAS/spa176/skeleton-free-pose-transfer/demo/gt_results_smpl/gt_1.obj"
    mean_distance = calculate_pmd(mesh1, mesh2)

    if mean_distance is not None:
        print(f"Mean Point-wise Mesh Euclidean Distance (PMD): {mean_distance}")
