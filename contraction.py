import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from matplotlib.colors import Normalize
from PIL import Image
from models.mlp import MLP
from models.utils import PoseEnc
import os
device = torch.device("cuda:0")


class DeformNet(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden_dim=256, num_layers=3, L=0):
        super().__init__()
        self.pose_enc = PoseEnc()
        self.L = L
        in_dim = 3 + 3 * 2 * L
        self.mlp = MLP(in_dim, num_layers, hidden_dim, out_dim, use_wn=False, act_type="relu", last_act_type="none")

    def forward(self, x):
        x = self.pose_enc(x, self.L)
        return self.mlp(x)

def load_mlp_weights(mlp, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    # print(checkpoint.keys())
    # mlp.load_state_dict(checkpoint['model_state_dict'])
    mlp.load_state_dict(checkpoint)


def partition_unit_cube(num_partitions, radius=0.7):
    step = (2 * radius) / num_partitions
    partitions = []
    for i in range(num_partitions):
        for j in range(num_partitions):
            for k in range(num_partitions):
                center = (
                    -radius + (i + 0.5) * step,
                    -radius + (j + 0.5) * step,
                    -radius + (k + 0.5) * step,
                )
                partitions.append((center, step))
    return partitions


def sample_points_in_ball(center, radius, num_points):
    points = []
    while len(points) < num_points:
        point = np.random.uniform(-radius, radius, 3)
        if np.linalg.norm(point) <= radius:
            points.append(center + point)
    return np.array(points)

def estimate_volume_contraction(mlp, points):
    with torch.no_grad():
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        transformed_points = mlp(points_tensor).cpu().numpy()
    original_volume = np.linalg.det(np.cov(points.T))
    transformed_volume = np.linalg.det(np.cov(transformed_points.T))
    contraction_factor = transformed_volume / original_volume
    return contraction_factor


# def color_cube(ax, partitions, contraction_factors):
#     # Normalize contraction factors to be between 0 and 1
#     min_factor = min(contraction_factors)
#     max_factor = max(contraction_factors)
#     normalized_factors = [
#         (cf - min_factor) / (max_factor - min_factor) for cf in contraction_factors
#     ]

#     for (center, step), norm_factor in zip(partitions, normalized_factors):
#         color = (
#             1 - norm_factor,
#             0,
#             norm_factor,
#             0.1,
#         )  # Faint color based on normalized contraction factor with smaller alpha
#         r = [center[0], center[0] + step]
#         g = [center[1], center[1] + step]
#         b = [center[2], center[2] + step]
#         vertices = [
#             [r[0], g[0], b[0]],
#             [r[1], g[0], b[0]],
#             [r[1], g[1], b[0]],
#             [r[0], g[1], b[0]],
#             [r[0], g[0], b[1]],
#             [r[1], g[0], b[1]],
#             [r[1], g[1], b[1]],
#             [r[0], g[1], b[1]],
#         ]
#         faces = [
#             [vertices[j] for j in [0, 1, 2, 3]],
#             [vertices[j] for j in [4, 5, 6, 7]],
#             [vertices[j] for j in [0, 1, 5, 4]],
#             [vertices[j] for j in [2, 3, 7, 6]],
#             [vertices[j] for j in [0, 3, 7, 4]],
#             [vertices[j] for j in [1, 2, 6, 5]],
#         ]
#         ax.add_collection3d(
#             Poly3DCollection(
#                 faces, facecolors=color, linewidths=1, edgecolors="r", alpha=0.25
#             )
#         )


def color_cube(ax, partitions, contraction_factors, cmap, norm):
    for (center, step), contraction_factor in zip(partitions, contraction_factors):
        color = cmap(norm(contraction_factor))
        r = [center[0] - step / 2, center[0] + step / 2]
        g = [center[1] - step / 2, center[1] + step / 2]
        b = [center[2] - step / 2, center[2] + step / 2]
        vertices = [
            [r[0], g[0], b[0]],
            [r[1], g[0], b[0]],
            [r[1], g[1], b[0]],
            [r[0], g[1], b[0]],
            [r[0], g[0], b[1]],
            [r[1], g[0], b[1]],
            [r[1], g[1], b[1]],
            [r[0], g[1], b[1]],
        ]
        faces = [
            [vertices[j] for j in [0, 1, 2, 3]],
            [vertices[j] for j in [4, 5, 6, 7]],
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [2, 3, 7, 6]],
            [vertices[j] for j in [0, 3, 7, 4]],
            [vertices[j] for j in [1, 2, 6, 5]],
        ]
        ax.add_collection3d(
            Poly3DCollection(
                faces, facecolors=color, linewidths=0.1, edgecolors="r", alpha=0.3
            )
        )


def main():
    num_partitions = 10
    num_points_per_cube = 100
    radius = 0.7
    checkpoint_path = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/learn_mlp_icp_shift_pe0/test_deformed_pc_avg_displacement_nn300/reg_deform_net.pth"

    num_layers = 3
    hidden_dim = 256
    L = 0
    # Initialize MLP
    mlp = DeformNet(3, 3, hidden_dim=hidden_dim, num_layers=num_layers, L=L).to(device)

    load_mlp_weights(mlp, checkpoint_path)
    mlp.eval()

    # Partition the unit cube
    partitions = partition_unit_cube(num_partitions, radius - 0.3)

    # Sample points and estimate volume contraction
    contraction_factors = []
    for center, step in partitions:
        points = sample_points_in_ball(np.array(center), step / 2, num_points_per_cube)
        contraction_factor = estimate_volume_contraction(mlp, points)
        # filter out the extreme values that are bigger than 100
        if contraction_factor > 2:
            print("##contraction_factor: ", contraction_factor)
            contraction_factor = 2
        contraction_factors.append(contraction_factor)

    # Normalize contraction factors
    min_factor = min(contraction_factors)
    max_factor = max(contraction_factors)
    print("!!min_factor: ", min_factor)
    print("!!max_factor: ", max_factor)
    # calculate mean contraction factor
    print("!!mean contraction_factors: ", np.array(contraction_factors).mean())
    norm = Normalize(vmin=min_factor, vmax=max_factor)
    cmap = cm.get_cmap("coolwarm")

    src_pc_path = "/NAS/spa176/papr-retarget/point_clouds/butterfly/points_0.npy"
    src_pc = np.load(src_pc_path)
    scale = 10.0
    src_pc = src_pc / scale

    # Plot the colored cube
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color_cube(ax, partitions, contraction_factors, cmap, norm)
    ax.set_xlim([-radius, radius])
    ax.set_ylim([-radius, radius])
    ax.set_zlim([-radius, radius])

    # Add color bar
    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(contraction_factors)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=5)
    cbar.set_label("Contraction Factor")

    ax.scatter(
        src_pc[:, 0],
        src_pc[:, 1],
        src_pc[:, 2],
        color="red",
        s=1,
        label="Source Point Cloud",
    )
    plt.legend()
    plt.show()
    # save figure to file
    fig.savefig("contraction.png")


if __name__ == "__main__":
    main()
