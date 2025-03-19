import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import tqdm


nn_body = 24
kpnn_body = 24
body_kp = 256
# reg_D = True
reg_D = False
inp_case = 0
# Load the tensor from the .pth file
# tensor_path = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/multi_mlp_icp_shift_pe8_pointnet_densify_concat/L1_fix_wing_kp_192_body_kp_256_kpnn_50_bodykpnn_256_cdw1000.0_rigidw1.0_nnwing5_nnbody250_kp_only_regD/total_deformed_kps_p1.pth"
# tensor_path = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/multi_mlp_icp_shift_pe8_pointnet_densify_concat/L1_fix_wing_kp_192_body_kp_96_kpnn_50_bodykpnn_96_cdw1000.0_rigidw1.0_nnwing5_nnbody90_kp_only_regD/total_deformed_kps_p1.pth"
if reg_D:
    exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body{body_kp}/L1_fix_wing_kp_192_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_regD_p3dNorm/"
else:
    if inp_case > 0:
        exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body{body_kp}/L1_fix_wing_kp_192_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm_inp{inp_case}/"
    else:
        exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body{body_kp}/L1_fix_wing_kp_192_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm/"
tensor_path = os.path.join(exp_path, "total_deformed_kps_p1.pth")

# tensor_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body96/L1_fix_wing_kp_192_body_kp_96_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_regD_p3dNorm/total_deformed_kps_p1.pth"
point_clouds = torch.load(tensor_path).cpu().numpy()  # Shape: (T, N, 3)

# Create a directory to save the frames
# output_dir = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/multi_mlp_icp_shift_pe8_pointnet_densify_concat/L1_fix_wing_kp_192_body_kp_96_kpnn_50_bodykpnn_96_cdw1000.0_rigidw1.0_nnwing5_nnbody90_kp_only_regD/kp_frames"
# get the parent directory of the tensor path and concatenate with the folder name "kp_frames"
output_dir = os.path.join(os.path.dirname(tensor_path), "kp_frames")
os.makedirs(output_dir, exist_ok=True)

# use_filtered = False
use_filtered = True

if use_filtered:
    # Apply the mask to exclude points
    # mask = ~(
    #     (point_clouds[0, :, 1] >= -0.3)
    #     & (point_clouds[0, :, 1] <= 0.3)
    #     & (point_clouds[0, :, 2] >= -0.5)
    #     & (point_clouds[0, :, 2] <= -0.25)
    # )
    mask = ~(
        (point_clouds[0, :, 1] >= -0.3)
        & (point_clouds[0, :, 1] <= 0.3)
        & (point_clouds[0, :, 2] >= -0.7)
        & (point_clouds[0, :, 2] <= -0.25)
    )
    filtered_indices = np.arange(point_clouds.shape[1])[mask]
else:
    filtered_indices = np.arange(point_clouds.shape[1])

# Function to plot the point cloud from different perspectives
def plot_point_clouds(point_cloud, frame_idx):
    fig = plt.figure(figsize=(15, 5))

    if use_filtered:
        filtered_points = point_cloud[mask]

        # # Set axis limits
        # xlim = (-0.3, 0.3)
        # ylim = (-0.7, -0.1)
        # zlim = (-0.2, 0.1)
        # Set axis limits
        xlim = (-0.7, -0.0)
        ylim = (-0.25, 0.1)
        zlim = (-0.2, 0.15)
    else:
        filtered_points = point_cloud

        # Set axis limits
        xlim = (-0.7, 0.7)
        ylim = (-0.7, 0.7)
        zlim = (-0.5, 0.0)

    # Front view
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2])
    for i, txt in enumerate(filtered_indices):
        ax1.text(
            filtered_points[i, 0],
            filtered_points[i, 1],
            filtered_points[i, 2],
            str(txt),
            size=8,
            zorder=1,
            color="k",
        )
    ax1.view_init(elev=0, azim=0)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.set_zlim(zlim)
    ax1.set_title('Front View')

    ax2 = fig.add_subplot(132, projection="3d")
    ax2.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2])
    for i, txt in enumerate(filtered_indices):
        ax2.text(
            filtered_points[i, 0],
            filtered_points[i, 1],
            filtered_points[i, 2],
            str(txt),
            size=8,
            zorder=1,
            color="k",
        )
    ax2.view_init(elev=0, azim=90)
    ax2.set_xlim(xlim)
    ax2.set_ylim(ylim)
    ax2.set_zlim(zlim)
    ax2.set_title("Left Side View")

    # Right side view
    ax3 = fig.add_subplot(133, projection="3d")
    ax3.scatter(filtered_points[:, 0], filtered_points[:, 1], filtered_points[:, 2])
    for i, txt in enumerate(filtered_indices):
        ax3.text(
            filtered_points[i, 0],
            filtered_points[i, 1],
            filtered_points[i, 2],
            str(txt),
            size=8,
            zorder=1,
            color="k",
        )
    ax3.view_init(elev=0, azim=-90)
    ax3.set_xlim(xlim)
    ax3.set_ylim(ylim)
    ax3.set_zlim(zlim)
    ax3.set_title("Right Side View")

    # # Side view
    # ax2 = fig.add_subplot(132, projection='3d')
    # ax2.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    # ax2.view_init(elev=0, azim=90)
    # ax2.set_title('Side View')

    # # Back view
    # ax3 = fig.add_subplot(133, projection='3d')
    # ax3.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2])
    # ax3.view_init(elev=0, azim=180)
    # ax3.set_title('Back View')

    plt.savefig(f'{output_dir}/frame_{frame_idx:04d}.png')
    plt.close()

# Generate frames for each time step
# use tqdm to show progress bar
for t in tqdm.tqdm(range(point_clouds.shape[0])):
    plot_point_clouds(point_clouds[t], t)

out_name = "kp_motion_filtered" if use_filtered else "kp_motion"
# Create a video from the frames
with imageio.get_writer(os.path.join(output_dir, f"{out_name}.mp4"), fps=10) as writer:
    for t in range(point_clouds.shape[0]):
        frame_path = f'{output_dir}/frame_{t:04d}.png'
        image = imageio.imread(frame_path)
        writer.append_data(image)

print("Video saved as point_cloud_motion.mp4")
