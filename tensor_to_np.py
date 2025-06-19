import torch
import numpy as np
import os
import argparse


def smooth_point_cloud(point_clouds, window_size):
    # Create an empty list to store the smoothed point clouds
    smoothed_point_clouds = []

    # Pad the list of point clouds with copies of the first and last point clouds
    padded_point_clouds = (
        ([point_clouds[0]] * (window_size // 2))
        + point_clouds
        + ([point_clouds[-1]] * (window_size // 2))
    )

    # Apply the moving average filter
    for i in range(window_size // 2, len(padded_point_clouds) - window_size // 2):
        # Get the window of point clouds for this iteration
        window = padded_point_clouds[i - window_size // 2 : i + window_size // 2 + 1]

        # check data type and separate numpy and pytorch tensors
        # if isinstance(window[0], torch.Tensor):
        #     average_point_cloud = torch.mean(window, dim=0)
        # else:
        average_point_cloud = np.mean(window, axis=0)
        # # Calculate the average point cloud for this window
        # average_point_cloud = np.mean(window, axis=0)

        # Add the average point cloud to the list of smoothed point clouds
        smoothed_point_clouds.append(average_point_cloud)

    # Set the first and last point clouds to be the same as the original ones
    smoothed_point_clouds[0] = point_clouds[0]
    smoothed_point_clouds[-1] = point_clouds[-1]

    return smoothed_point_clouds


def smooth_point_cloud_torch(point_clouds, window_size):
    """
    Apply temporal smoothing to a sequence of point clouds.

    :param point_clouds: PyTorch tensor of shape (T, N, 3), where T is the number of time steps,
                         N is the number of points, and 3 represents the spatial dimensions.
    :param window_size: Size of the moving average window (must be an odd number).
    :return: Smoothed point clouds as a PyTorch tensor of shape (T, N, 3).
    """
    assert window_size % 2 == 1, "Window size must be an odd number."

    print(f"Input tensor shape: {point_clouds.shape}")
    # Pad the point clouds along the temporal dimension
    padding = window_size // 2
    padded_point_clouds = torch.cat(
        [
            point_clouds[:1].repeat(padding, 1, 1),  # Repeat the first frame
            point_clouds,
            point_clouds[-1:].repeat(padding, 1, 1),
        ],  # Repeat the last frame
        dim=0,
    )

    # Apply the moving average filter
    smoothed_point_clouds = []
    for t in range(padding, len(padded_point_clouds) - padding):
        # Get the window of point clouds for this time step
        window = padded_point_clouds[t - padding : t + padding + 1]
        # Calculate the average point cloud for this window
        average_point_cloud = window.mean(dim=0)
        smoothed_point_clouds.append(average_point_cloud)

    # Stack the smoothed point clouds along the temporal dimension
    smoothed_point_clouds = torch.stack(smoothed_point_clouds, dim=0)
    print(f"Output tensor shape: {smoothed_point_clouds.shape}")

    return smoothed_point_clouds


def main():
    # parser = argparse.ArgumentParser(description="Smooth a sequence of point clouds")
    # parser.add_argument(
    #     "--input", type=str, required=True, help="Path to input tensor file"
    # )
    # parser.add_argument(
    #     "--output", type=str, required=True, help="Path to save output numpy file"
    # )
    # parser.add_argument(
    #     "--window", type=int, default=5, help="Window size for smoothing (must be odd)"
    # )
    # parser.add_argument(
    #     "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    # )

    # args = parser.parse_args()

    # # Ensure window size is odd
    # if args.window % 2 == 0:
    #     args.window += 1
    #     print(f"Window size adjusted to {args.window} to ensure it's odd")

    # Create output directory if it doesn't exist
    # os.makedirs(os.path.dirname(args.output), exist_ok=True)
    tensor_path = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/blender_mp_wingL520_wingR520_body1134/exp_0/transferred_tgt_kps_smooth.pth"

    # Load the tensor
    print(f"Loading tensor from {tensor_path}")
    point_clouds = torch.load(tensor_path).cpu().numpy()

    scale = 4.5905112356900775
    if scale != 1.0:
        print(f"Scaling point clouds by {scale}")
        point_clouds = point_clouds * scale

    skip_num = 5
    total_num_steps = point_clouds.shape[0]
    total_deformed_pcs = [point_clouds[i] for i in range(total_num_steps-1, -1, -1)]
    if skip_num > 1:
        total_deformed_pcs = total_deformed_pcs[::skip_num]

    print(f"Saving Total number of point clouds: {len(total_deformed_pcs)}")
    np.save(
        os.path.join(os.path.dirname(tensor_path), "deformed_pts.npy"),
        np.stack(total_deformed_pcs, axis=0),
    )
    window_size = 7
    # Apply smoothing
    print(f"Applying smoothing with window size {window_size}")
    smoothed_point_clouds = smooth_point_cloud(total_deformed_pcs, window_size)

    np.save(
        os.path.join(os.path.dirname(tensor_path), "deformed_pts_smooth.npy"),
        np.stack(smoothed_point_clouds, axis=0),
    )

    print("Done!")


if __name__ == "__main__":
    main()
