import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import subprocess
import pytorch3d
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from pytorch3d.ops import iterative_closest_point as icp
from pytorch3d.ops import sample_farthest_points as sfp
from pytorch3d.transforms import quaternion_to_matrix
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import tqdm
import imageio
import io
from PIL import Image
import random
from models.mlp import MLP
from models.utils import PoseEnc
from pointnet2_utils import (
    PointNetSetAbstractionMsg,
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)
from pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer
from sklearn.neighbors import NearestNeighbors


class pointnet(nn.Module):
    def __init__(self, part_num=50, normal_channel=False):
        super(pointnet, self).__init__()
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.part_num = part_num
        self.stn = STN3d(channel)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128)
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, part_num, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud):
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat = self.fstn(out3)
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        return out5


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
    pc = pc / m
    return pc


def estimate_surface_normals(point_cloud, num_nn):
    """
    Estimate the surface normals for a point cloud.

    Args:
    - point_cloud (torch.Tensor): The point cloud (N, 3).
    - num_nn (int): The number of nearest neighbors to use for normal estimation.

    Returns:
    - normals (torch.Tensor): The estimated surface normals (N, 3).
    """
    num_pts = point_cloud.size(0)

    # Convert point cloud to numpy for NearestNeighbors
    point_cloud_np = point_cloud.cpu().numpy()

    # Find the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=num_nn + 1, algorithm="auto").fit(
        point_cloud_np
    )
    distances, indices = nbrs.kneighbors(point_cloud_np)

    # Initialize tensor for normals
    normals = torch.zeros((num_pts, 3), device=point_cloud.device)

    for pt_idx in range(num_pts):
        # Extract the nearest neighbors
        neighbors = point_cloud[indices[pt_idx, 1:], :]

        # Center the neighbors by subtracting the mean
        neighbors_centered = neighbors - neighbors.mean(dim=0, keepdim=True)

        # Compute the covariance matrix
        covariance_matrix = neighbors_centered.t().mm(neighbors_centered) / (num_nn - 1)

        # Perform eigen decomposition
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)

        # The eigenvector corresponding to the smallest eigenvalue is the normal
        normal = eigenvectors[:, eigenvalues[:, 0].argmin()]

        # Store the normal
        normals[pt_idx, :] = normal

    return normals


class DeformNet(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden_dim=256, num_layers=3, L=0):
        super().__init__()

        self.pose_enc = PoseEnc()
        self.L = L
        in_dim = in_dim + in_dim * 2 * L
        self.mlp = MLP(
            in_dim,
            num_layers,
            hidden_dim,
            out_dim,
            use_wn=False,
            act_type="relu",
            last_act_type="none",
        )

    def forward(self, x):
        x = self.pose_enc(x, self.L)
        return self.mlp(x)


# src_pc_path = "/NAS/spa176/papr-retarget/point_clouds/butterfly/points_0.npy"
# src_pc = np.load(src_pc_path)
# src_pc = src_pc / 10.0
# # calculate normals
# src_pc = torch.tensor(src_pc)
# src_normals = estimate_surface_normals(src_pc, 100)
# src_normals = src_normals / torch.norm(src_normals, dim=1, keepdim=True)
# src_normals = src_normals.numpy()
# # subsample the point cloud and normals
# # src_pc = src_pc.numpy()
# # src_pc = src_pc[::100]
# # src_normals = src_normals[::100]
# # plot the surface normals on the point cloud
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection="3d")
# # ax.scatter(src_pc[:, 0], src_pc[:, 1], src_pc[:, 2], c="b", marker="o")
# # ax.quiver(
# #     src_pc[:, 0],
# #     src_pc[:, 1],
# #     src_pc[:, 2],
# #     src_normals[:, 0],
# #     src_normals[:, 1],
# #     src_normals[:, 2],
# #     length=0.1,
# #     color="r",
# # )
# # ax.set_xlabel("X")
# # ax.set_ylabel("Y")
# # ax.set_zlabel("Z")
# # plt.show()
device = torch.device("cuda:0")
# tgt_pc_path = "/NAS/spa176/papr-retarget/point_clouds/hummingbird/points_0.npy"
# tgt_pc = np.load(tgt_pc_path)
# src_pc = torch.tensor(src_pc).float()
# tgt_pc = torch.tensor(tgt_pc).float()
# scale = 10.0
num_layers = 3
hidden_dim = 256
L = 0
USE_POINTNET = True

log_dir = "fit_pointcloud_logs"
if USE_POINTNET:
    exp_dir = f"learn_mlp_icp_shift_pe{L}_pointnet"
else:
    exp_dir = f"learn_mlp_icp_shift_pe{L}"
# exp_dir = f"learn_mlp_icp_shift_pe{L}"
# exp_dir = f'learn_mlp_icp_shift_pe{L}_pointnet'
log_dir = os.path.join(log_dir, exp_dir)

# tgt_pc = tgt_pc / scale

# converged, rmse, Xt, RTs, t_history = icp(tgt_pc.unsqueeze(0), src_pc.unsqueeze(0))
# print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
# tgt_pc = Xt.squeeze(0)

src_pc_dir = "/NAS/yza629/codes/papr-retarget/point_clouds/butterfly"
start = 0
end = 30001
interval = 1000
src_pcs = []
deformed_pcs = []
for idx in tqdm.tqdm(range(start, end, interval)):
    src_pc_path = os.path.join(src_pc_dir, f"points_{idx}.npy")
    src_pc = np.load(src_pc_path)

    # src_pc = torch.tensor(src_pc).float().to(device)
    src_pc = torch.tensor(src_pc).float()

    scale = 10.0
    src_pc = src_pc / scale

    src_pcs.append(src_pc)
src_pcs = torch.stack(src_pcs, dim=0)

init_pc = src_pcs[0]
init_normals = estimate_surface_normals(init_pc, 100)
init_normals = init_normals / torch.norm(init_normals, dim=1, keepdim=True)
init_normals = init_normals.numpy()
# subsample the point cloud and normals
init_pc = init_pc.numpy()
init_pc = init_pc[::100]
init_normals = init_normals[::100]

end_pc = src_pcs[-1]
end_normals = estimate_surface_normals(end_pc, 100)
end_normals = end_normals / torch.norm(end_normals, dim=1, keepdim=True)
end_normals = end_normals.numpy()
# subsample the point cloud and normals
end_pc = end_pc.numpy()
end_pc = end_pc[::100]
end_normals = end_normals[::100]

# calculate the dot product between the initial and final normals, and filter out the ones with negative dot product
dot_products = np.sum(init_normals * end_normals, axis=1)
init_pc = init_pc[dot_products <= 0]
init_normals = init_normals[dot_products <= 0]
end_pc = end_pc[dot_products <= 0]
end_normals = end_normals[dot_products <= 0]

num_points = init_pc.shape[0]
colors = cm.rainbow(np.linspace(0, 1, num_points))

# plot the surface normals on the point cloud and mark the initial and final point clouds
# make the points at the same index the same color for better visualization
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

for i in range(num_points):
    ax.scatter(init_pc[i, 0], init_pc[i, 1], init_pc[i, 2], color=colors[i], marker="o")
    ax.scatter(end_pc[i, 0], end_pc[i, 1], end_pc[i, 2], color=colors[i], marker="o")

# Plot initial and final normals with corresponding colors
for i in range(num_points):
    ax.quiver(
        init_pc[i, 0],
        init_pc[i, 1],
        init_pc[i, 2],
        init_normals[i, 0],
        init_normals[i, 1],
        init_normals[i, 2],
        length=0.1,
        color=colors[i],
    )
    ax.quiver(
        end_pc[i, 0],
        end_pc[i, 1],
        end_pc[i, 2],
        end_normals[i, 0],
        end_normals[i, 1],
        end_normals[i, 2],
        length=0.1,
        color=colors[i],
    )

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# ax.scatter(init_pc[:, 0], init_pc[:, 1], init_pc[:, 2], c="b", marker="o")
# ax.scatter(end_pc[:, 0], end_pc[:, 1], end_pc[:, 2], c="g", marker="o")
# ax.quiver(
#     init_pc[:, 0],
#     init_pc[:, 1],
#     init_pc[:, 2],
#     init_normals[:, 0],
#     init_normals[:, 1],
#     init_normals[:, 2],
#     length=0.1,
#     color="r",
# )
# ax.quiver(
#     end_pc[:, 0],
#     end_pc[:, 1],
#     end_pc[:, 2],
#     end_normals[:, 0],
#     end_normals[:, 1],
#     end_normals[:, 2],
#     length=0.1,
#     color="y",
# )
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()


# src_pc = src_pcs[-1]
# src_normals = estimate_surface_normals(src_pc, 100)
# src_normals = src_normals / torch.norm(src_normals, dim=1, keepdim=True)
# src_normals = src_normals.numpy()
# # subsample the point cloud and normals
# src_pc = src_pc.numpy()
# src_pc = src_pc[::100]
# src_normals = src_normals[::100]
# # plot the surface normals on the point cloud
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(src_pc[:, 0], src_pc[:, 1], src_pc[:, 2], c="b", marker="o")
# ax.quiver(
#     src_pc[:, 0],
#     src_pc[:, 1],
#     src_pc[:, 2],
#     src_normals[:, 0],
#     src_normals[:, 1],
#     src_normals[:, 2],
#     length=0.1,
#     color="r",
# )
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()


# if USE_POINTNET:
#     reg_net = DeformNet(2048, 3, hidden_dim=hidden_dim, num_layers=num_layers, L=L).to(
#         device
#     )
#     pointnet = pointnet(normal_channel=False).to(device)
#     checkpoint = torch.load("best_model.pth")
#     pointnet.load_state_dict(checkpoint["model_state_dict"])
#     pointnet.eval()
# else:
#     reg_net = DeformNet(3, 3, hidden_dim=hidden_dim, num_layers=num_layers, L=L).to(device)
# reg_net.load_state_dict(torch.load(os.path.join(log_dir, "deform_net.pth")))

# with torch.no_grad():
#     if USE_POINTNET:
#         point_feat = pointnet(
#             pc_normalize(src_pcs[0].to(device)).unsqueeze(0).transpose(2, 1)
#         ).transpose(2, 1)[0]
#         init_displacement = reg_net(point_feat)
#         deformed_src_pc = src_pcs[0] + init_displacement.detach().cpu().clone()
#     else:
#         init_displacement = reg_net(src_pcs[0:1].to(device))
#         deformed_src_pc = src_pcs[0] + init_displacement[0].detach().cpu().clone()

# del reg_net
# del pointnet

# sub_sample = 100
# # subsample init_displacement and plot the deformed point cloud
# if USE_POINTNET:
#     init_displacement_cpu = init_displacement.cpu().numpy()
# else:
#     init_displacement_cpu = init_displacement.cpu().numpy()[0]
# init_displacement_cpu = init_displacement_cpu[::sub_sample]
# src_pc = src_pcs[0].cpu().numpy()[::sub_sample]
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# # plot the src_pc and draw the displacement vectors init_displacement
# # plot the deformed point cloud and draw the displacement vectors init_displacement in between
# deformed_src_pc_cpu = deformed_src_pc.cpu().numpy()[::sub_sample]
# ax.scatter(src_pc[:, 0], src_pc[:, 1], src_pc[:, 2], c="b", marker="o")
# ax.scatter(
#     deformed_src_pc_cpu[:, 0],
#     deformed_src_pc_cpu[:, 1],
#     deformed_src_pc_cpu[:, 2],
#     c="g",
#     marker="o",
# )
# ax.quiver(
#     src_pc[:, 0],
#     src_pc[:, 1],
#     src_pc[:, 2],
#     init_displacement_cpu[:, 0],
#     init_displacement_cpu[:, 1],
#     init_displacement_cpu[:, 2],
#     # length=0.2,
#     color="r",
# )
# ax.set_xlabel("X")
# ax.set_ylabel("Y")
# ax.set_zlabel("Z")
# plt.show()
