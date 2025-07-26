import numpy as np
import plotly.graph_objects as go
import plotly.colors
from pytorch3d.ops import sample_farthest_points as sfp
import torch

# Load the point cloud

# tgt_pc_path = "/NAS/spa176/papr-retarget/point_clouds/hummingbird/points_0.npy"
# tgt_pc = torch.from_numpy(np.load(tgt_pc_path)).float()
# bird_wing_indices = np.load("hummingbird_wing_indices.npy")
# bird_body_indices = np.setdiff1d(np.arange(len(tgt_pc)), bird_wing_indices)
# downsampled_points = sfp(tgt_pc[bird_body_indices].unsqueeze(0), K=96)[0].squeeze(0)
downsampled_points = torch.load(
    "/NAS/spa176/papr-retarget/fit_pointcloud_logs/smpl/exp_13/transferred_tgt_kps.pth"
)
print(f"Downsampled points shape: {downsampled_points.keys()}")

# Example: data_dict = {'label1': tensor1, ...}
color_list = plotly.colors.qualitative.Plotly  # 10 colors by default
# For more colors, you can use other color sets or repeat the list
all_colors = plotly.colors.qualitative.Alphabet  # 26 distinct colors

fig = go.Figure()
for i, (key, tensor) in enumerate(downsampled_points.items()):
    xyz = tensor[0][0].cpu().numpy()
    print(f"Key: {key}, Shape: {xyz.shape}")
    fig.add_trace(go.Scatter3d(
        x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
        mode='markers',
        marker=dict(size=3, color=all_colors[i % len(all_colors)]),
        name=key
    ))

fig.update_layout(
    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
    title="Point Cloud",
)
fig.show()

# for k, v in downsampled_points.items():


# # Create a Plotly scatter plot
# fig = go.Figure(
#     data=[
#         go.Scatter3d(
#             x=downsampled_points[:, 0],
#             y=downsampled_points[:, 1],
#             z=downsampled_points[:, 2],
#             mode="markers",
#             marker=dict(
#                 size=2,
#                 color=downsampled_points[:, 2],  # Color by z-coordinate
#                 colorscale="Viridis",
#                 opacity=0.8,
#             ),
#         )
#     ]
# )

# # Set plot layout
# fig.update_layout(
#     scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
#     title="Downsampled Point Cloud",
# )

# # Show the plot
# fig.show()
