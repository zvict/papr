import numpy as np

# import open3d as o3d # No longer needed for visualization
from pylgcpd import DeformableRegistration
import argparse
import os

# import sys # Not used
import plotly.graph_objects as go


def load_xyz(filepath):
    """Loads a point cloud from an .xyz file."""
    try:
        points = np.loadtxt(filepath)
        if points.ndim == 1:  # Handle case with only one point
            points = points.reshape(1, -1)
        if points.shape[1] != 3:
            raise ValueError(
                f"XYZ file {filepath} must contain 3 columns (X Y Z). Found {points.shape[1]}."
            )
        return points
    except Exception as e:
        print(f"Error loading XYZ file {filepath}: {e}")
        return None


def visualize_registration_plotly(X_target, Y_source_original, TY_source_registered):
    """
    Visualizes the target, original source, and registered source point clouds using Plotly.
    Draws lines from original source points to their registered positions.
    """
    fig = go.Figure()

    # Target points (X) - Blue
    fig.add_trace(
        go.Scatter3d(
            x=X_target[:, 0],
            y=X_target[:, 1],
            z=X_target[:, 2],
            mode="markers",
            marker=dict(size=3, color="blue", opacity=0.8),
            name="Target (X)",
        )
    )

    # Original source points (Y) - Orange
    fig.add_trace(
        go.Scatter3d(
            x=Y_source_original[:, 0],
            y=Y_source_original[:, 1],
            z=Y_source_original[:, 2],
            mode="markers",
            marker=dict(size=3, color="orange", opacity=0.8),
            name="Original Source (Y)",
        )
    )

    # Registered source points (TY) - Green
    fig.add_trace(
        go.Scatter3d(
            x=TY_source_registered[:, 0],
            y=TY_source_registered[:, 1],
            z=TY_source_registered[:, 2],
            mode="markers",
            marker=dict(size=3, color="green", opacity=0.8),
            name="Registered Source (TY)",
        )
    )

    # Lines for correspondences (from Y_original to TY_registered) - Gray
    num_points = Y_source_original.shape[0]
    for i in range(num_points):
        fig.add_trace(
            go.Scatter3d(
                x=[Y_source_original[i, 0], TY_source_registered[i, 0]],
                y=[Y_source_original[i, 1], TY_source_registered[i, 1]],
                z=[Y_source_original[i, 2], TY_source_registered[i, 2]],
                mode="lines",
                line=dict(color="gray", width=2),
                name=(
                    f"Correspondence {i}" if num_points < 20 else None
                ),  # Avoid too many legend items
                showlegend=(
                    i == 0 and num_points >= 20
                ),  # Show legend only for the first line if many
            )
        )

    if num_points >= 20 and len(fig.data) > 3:  # If lines were added and there are many
        fig.data[-1].name = "Correspondences"  # Group legend for lines
        fig.data[-1].showlegend = True

    fig.update_layout(
        title="Deformable Registration Result",
        scene=dict(
            xaxis_title="X Axis",
            yaxis_title="Y Axis",
            zaxis_title="Z Axis",
            aspectmode="data",  # Ensures aspect ratio is preserved
        ),
        margin=dict(l=0, r=0, b=0, t=40),  # Adjust margins
    )

    print("Visualizing registration with Plotly:")
    print("- Blue: Target points (X)")
    print("- Orange: Original source points (Y)")
    print("- Green: Registered source points (TY)")
    print("- Gray lines: Movement of source points (Y -> TY)")

    fig.show()


def main():
    parser = argparse.ArgumentParser(
        description="Register two sets of keypoints using pylgcpd."
    )
    # parser.add_argument("target_xyz_file", type=str, help="Path to the target .xyz keypoint file (fixed points X).")
    # parser.add_argument("source_xyz_file", type=str, help="Path to the source .xyz keypoint file (moving points Y).")

    # Add common pylgcpd parameters as arguments if needed
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Represents the trade-off between the goodness of maximum likelihood fit and regularization.",
    )  # Default from a common CPD setup
    parser.add_argument(
        "--beta",
        type=float,
        default=2.0,
        help="Represents the width of Gaussian kernel.",
    )  # Default from a common CPD setup
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Maximum number of registration iterations.",
    )
    parser.add_argument(
        "--tolerance", type=float, default=1e-5, help="Tolerance for convergence."
    )

    args = parser.parse_args()

    source_xyz_file = "/NAS/spa176/papr-retarget/but_kps_448.xyz"
    target_xyz_file = "/NAS/spa176/papr-retarget/bird_kps_448.xyz"

    print(f"Loading target keypoints from: {target_xyz_file}")
    X_target = load_xyz(target_xyz_file)
    if X_target is None:
        return

    print(f"Loading source keypoints from: {source_xyz_file}")
    Y_source = load_xyz(source_xyz_file)
    if Y_source is None:
        return

    print(f"\nTarget (X) shape: {X_target.shape}")
    print(f"Source (Y) shape: {Y_source.shape}")

    if X_target.shape[0] == 0 or Y_source.shape[0] == 0:
        print("Error: One or both point clouds are empty.")
        return

    print(
        f"\nInitializing DeformableRegistration with alpha={args.alpha}, beta={args.beta}, max_iter={args.max_iterations}..."
    )
    try:
        reg = DeformableRegistration(
            X=X_target,
            Y=Y_source,
            X_landmarks=np.array(
                [
                    [0.731024, -3.449986, -0.761838],  # id 4
                    [0.803824, 3.546330, -0.753426],  # id 1
                    [3.356918, 0.000732, -1.498761],  # id 16
                    [-2.630342, 0.085102, -2.069363],  # id 2
                    [0.644358, 0.519017, -1.054321],  # id 383
                    [0.441624, -0.466698, -1.057566],  # id 354
                    [-0.432543, -0.449939, -1.126825],  # id 135
                    [-0.421909, 0.572028, -1.125596],  # id 76
                    [0.596840, -1.267992, -0.977735],  # left wing bottom tip, id 18
                    [
                        -0.096449,
                        -2.102003,
                        -0.947039,
                    ],  # left wing top mid point, id 301
                    [0.624270, 1.368125, -0.974566],  # right wing bottom tip, id 393
                    [
                        -0.051515,
                        2.216305,
                        -0.941847,
                    ],  # right wing top mid point, id 300
                ],
                dtype=np.float32,
            ),
            Y_landmarks=np.array(
                [
                    [-1.166227, -3.440091, -0.447035],  # id 217
                    [-1.125638, 3.556031, -0.292649],  # id 1
                    [1.628076, 0.090335, -1.562362],  # id 2
                    [-0.745807, 0.067880, -1.311804],  # id 16
                    [0.580139, 0.361881, -1.361487],  # id 196
                    [0.209989, -0.133029, -1.205299],  # id 440
                    [-0.338329, -0.287892, -1.463169],  # id 42
                    [-0.432801, 0.522794, -1.110381],  # id 56
                    [1.664210, -1.262751, -1.512159],  # left wing bottom tip, id 11
                    [
                        -1.181260,
                        -1.939071,
                        -0.816759,
                    ],  # left wing top mid point, id 110
                    [1.702331, 1.714027, -1.348796],  # right wing bottom tip, id 148
                    [-1.063435, 1.876169, -0.762544],  # right wing top mid point, id 402
                ],
                dtype=np.float32,
            ),
            alpha=args.alpha,  # These are now default in the class if not overridden
            beta=args.beta,
            max_iterations=args.max_iterations,  # These might be passed to register() or set after init
            # tolerance=args.tolerance,
        )
        # If max_iterations and tolerance are not constructor args for your pylgcpd version:
        # reg.max_iterations = args.max_iterations
        # reg.tolerance = args.tolerance

        print("Starting registration...")
        TY, params = reg.register()

        print("Registration finished.")
        print(f"Registered source (TY) shape: {TY.shape}")

    except AttributeError as e:
        print(f"AttributeError during pylgcpd usage: {e}")
        print("This might be due to a mismatch in the expected pylgcpd API.")
        print(
            "Please check the pylgcpd version and its documentation for correct class/method names and parameters."
        )
        return
    except Exception as e:
        print(f"An error occurred during pylgcpd registration: {e}")
        return

    # Visualize the results using Plotly
    visualize_registration_plotly(X_target, Y_source, TY)


if __name__ == "__main__":
    # Create dummy xyz files for testing if they don't exist
    # (Your existing dummy file creation logic can remain here if desired)
    main()
