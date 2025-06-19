import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
import subprocess
import pytorch3d
from pytorch3d.ops import iterative_closest_point as icp
from pytorch3d.ops import sample_farthest_points as sfp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm
import imageio
import io
from PIL import Image
import random
from scipy.spatial import KDTree, cKDTree
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go
import open3d as o3d
from pytorch3d.ops import estimate_pointcloud_normals
from torch.nn import SmoothL1Loss
from scipy.spatial import ConvexHull
from abc import ABC, abstractmethod
from scipy.spatial.distance import cdist
from rbf_factory import RBFFactory
import shutil
import trimesh
from hierarchy import Hierarchy

# import tinycudann as tcnn

os.chdir("/NAS/spa176/papr-retarget")


device = torch.device("cuda:0")
# log_dir = "fit_pointcloud_logs"
# exp_dir = 'learn_mlp_icp_shift_pe0_pointnet'
# log_dir = os.path.join(log_dir, exp_dir)
# os.makedirs(log_dir, exist_ok=True)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

setup_seed(0)

def plot_pointcloud(points, save_dir, title="", extra_points=None, plot_match=True, scale=1.0):
    points = points.unsqueeze(0) if points.dim() == 2 else points
    num_pc = points.shape[0]

    fig = plt.figure(figsize=(10, 5 * num_pc))
    for i in range(num_pc):
        x, y, z = points[i].clone().detach().cpu().squeeze().unbind(1)
        colors = x

        ax = fig.add_subplot(num_pc, 2, i * 2 + 1, projection='3d')        
        ax.scatter3D(x, y, z, c=colors, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-0.6 * scale, 0.6 * scale)
        ax.set_ylim(-0.6 * scale, 0.6 * scale)
        ax.set_zlim(-0.6 * scale, 0.6 * scale)
        ax.set_title(title + f' pc {i} view 1')
        ax.view_init(9, -143)

        if extra_points is not None:
            x_e, y_e, z_e = extra_points.clone().detach().cpu().squeeze().unbind(1)
            cur_colors = x_e
            ax.scatter3D(x_e, y_e, z_e, c=cur_colors, cmap="magma", alpha=0.5)
            # plot a vector from each point to the corresponding extra point
            if plot_match:
                for j in range(len(x)):
                    ax.plot([x[j], x_e[j]], [y[j], y_e[j]], [z[j], z_e[j]], 'r-')

        ax = fig.add_subplot(num_pc, 2, i * 2 + 2, projection='3d')        
        ax.scatter3D(x, y, z, c=colors, cmap='viridis')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim(-0.6 * scale, 0.6 * scale)
        ax.set_ylim(-0.6 * scale, 0.6 * scale)
        ax.set_zlim(-0.6 * scale, 0.6 * scale)
        ax.set_title(title + f' pc {i} view 2')
        ax.view_init(79, 164)

        if extra_points is not None:
            ax.scatter3D(x_e, y_e, z_e, c=cur_colors, cmap="magma", alpha=0.5)
            # plot a vector from each point to the corresponding extra point
            if plot_match:
                for j in range(len(x)):
                    ax.plot([x[j], x_e[j]], [y[j], y_e[j]], [z[j], z_e[j]], 'r-')

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, title + ".png"))
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()
    return np.array(img)


def estimate_normals_pytorch3d(points, knn=20):
    reference_point = np.mean(points, axis=0)
    # reference_point=np.array([0, 0, 0])
    # Convert points to PyTorch tensor
    points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(
        0
    )  # Shape: (1, N, 3)

    # Estimate normals using PyTorch3D
    normals_tensor = estimate_pointcloud_normals(
        points_tensor, neighborhood_size=knn, disambiguate_directions=True
    )

    # Convert normals to numpy array
    normals = normals_tensor.squeeze(0).cpu().numpy()

    # Ensure normals are facing away from the reference point
    vectors_to_reference = points - reference_point
    dot_products = np.einsum("ij,ij->i", vectors_to_reference, normals)
    normals[dot_products < 0] *= -1

    return normals

def find_closest_points(source_points, target_points):
    """
    Find the closest points in the target point cloud for each point in the source point cloud.
    
    :param source_points: Source point cloud points (numpy array of shape (N, 3))
    :param target_points: Target point cloud points (numpy array of shape (M, 3))
    :return: Indices of the closest points in the target point cloud (numpy array of shape (N,))
    """
    source_points = np.expand_dims(source_points, axis=1)
    target_points = np.expand_dims(target_points, axis=0)
    distances = np.linalg.norm(source_points - target_points, axis=2)
    closest_indices = np.argmin(distances, axis=1)
    return closest_indices


# def positional_encoding(x, num_frequencies, mask_ratio):
#     # x: shape (N, 3)
#     # num_frequencies: max frequencies to use
#     # mask_ratio: fraction of frequencies to use (progressive)
#     frequencies = int(num_frequencies * mask_ratio)
#     # print("!!!frequencies", frequencies)
#     # pe = [x]
#     pe = []
#     for i in range(num_frequencies):
#         for fn in [torch.sin, torch.cos]:
#             pe.append(fn((2.0 ** i) * x) * (i < frequencies))
#     return torch.cat(pe, dim=-1)


# def positional_encoding(x):
#     pe = []
#     for fn in [torch.sin, torch.cos]:
#         pe.append(fn(x))
#     return torch.cat(pe, dim=-1)


def positional_encoding(x, num_frequencies, factor=2.0, with_self=False):
    if with_self:
        pe = [x]
    else:
        pe = []
    for i in range(num_frequencies):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn((factor**i) * x))
    return torch.cat(pe, dim=-1)


class PointwiseTransformNetFast(nn.Module):
    def __init__(self, base_channels=32, max_freq=1, temperature=50.0, out_dim=6):
        super().__init__()
        self.max_freq = max_freq
        self.temperature = temperature
        # self.net = nn.Sequential(
        #     # nn.Linear((1 + 2*max_freq) * 3, base_channels),
        #     nn.Linear((2 * max_freq) * 3, base_channels),
        #     nn.ReLU(),
        #     nn.Linear(base_channels, base_channels),
        #     nn.ReLU(),
        #     nn.Linear(base_channels, out_dim),
        # )
        config = {
            "encoding": {
                "otype": "Identity",  # No special encoding, raw input is used
            },
            "network": {
                "n_input_dims": (2 * max_freq) * 3,  # Input size
                "n_output_dims": out_dim,  # Output size
                "otype": "FullyFusedMLP",  # Fully connected MLP
                "activation": "ReLU",  # ReLU activation
                "output_activation": "None",  # No activation on the output layer
                "n_neurons": base_channels,  # Number of neurons in hidden layers
                "n_hidden_layers": 1,  # Number of hidden layers
            },
        }

        # Initialize the tinycudann model
        self.net = tcnn.NetworkWithInputEncoding(
            n_input_dims=(2 * max_freq) * 3,  # Input size
            n_output_dims=out_dim,  # Output size
            encoding_config=config["encoding"],
            network_config=config["network"],
        )

    def forward(self, x, mask_ratio=1.0):
        # normalize x so that it falls within [0, 1]
        x = (x - x.min()) / ((x.max() - x.min()) * self.temperature)
        # x: shape (N, 3)
        pe = positional_encoding(x)  # (N, enc_dim)
        out = self.net(pe)  # (N, out_dim)
        return out


class PointwiseTransformNet(nn.Module):
    def __init__(self, base_channels=64, max_freq=1, temperature=5.0, out_dim=6):
        super().__init__()
        self.max_freq = max_freq
        self.temperature = temperature
        self.net = nn.Sequential(
            # nn.Linear((1 + 2*max_freq) * 3, base_channels),
            nn.Linear((2*max_freq) * 3, base_channels),
            nn.ReLU(),
            nn.Linear(base_channels, base_channels),
            nn.ReLU(),
            nn.Linear(base_channels, out_dim)
        )

    def forward(self, x, mask_ratio=1.0):
        # normalize x so that it falls within [0, 1]
        x = (x - x.min()) / ((x.max() - x.min()) * self.temperature)
        # x: shape (N, 3)
        # pe = positional_encoding(x, self.max_freq, mask_ratio)  # (N, enc_dim)
        pe = positional_encoding(x, self.max_freq)  # (N, enc_dim)
        out = self.net(pe)  # (N, out_dim)
        return out


class PointwiseTransformPartNet(nn.Module):
    def __init__(self, base_channels=64, max_freq=1, temperature=50.0, out_dim=6, mask_freq=0, num_layers=1):
        super().__init__()
        self.max_freq = max_freq
        self.mask_freq = mask_freq
        self.temperature = temperature
        self.base_channels = base_channels
        self.num_layers = num_layers
        layers = [
            nn.Linear((2 * max_freq) * 3 + 2 * mask_freq, base_channels),
            nn.ReLU(),
        ]
        for _ in range(num_layers):
            layers.append(nn.Linear(base_channels, base_channels))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(base_channels, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask):
        # normalize x so that it falls within [0, 1]
        x = (x - x.min()) / ((x.max() - x.min()) * self.temperature)
        # x: shape (N, 3)
        # pe = positional_encoding(x, self.max_freq, mask_ratio)  # (N, enc_dim)
        pe = positional_encoding(x, self.max_freq, with_self=False)  # (N, enc_dim)
        mask = positional_encoding(mask, self.mask_freq, with_self=False)  # (N, mask_dim)
        out = self.net(torch.cat([pe, mask], dim=-1))  # (N, out_dim)
        return out


def axis_angle_to_rotation_matrix(axis_angle):
    # axis_angle: (N, 3)
    # Using Rodrigues' formula, or a PyTorch function if available
    # This is a minimal illustrative example
    angle = torch.norm(axis_angle, dim=-1, keepdim=True) + 1e-9
    axis = axis_angle / angle
    ca = torch.cos(angle)
    sa = torch.sin(angle)
    one_minus_ca = 1 - ca
    x, y, z = axis[...,0], axis[...,1], axis[...,2]
    R = torch.zeros(axis_angle.shape[0], 3, 3, device=axis_angle.device)
    R[:,0,0] = ca[:,0] + x*x*one_minus_ca[:,0]
    R[:,0,1] = x*y*one_minus_ca[:,0] - z*sa[:,0]
    R[:,0,2] = x*z*one_minus_ca[:,0] + y*sa[:,0]
    R[:,1,0] = y*x*one_minus_ca[:,0] + z*sa[:,0]
    R[:,1,1] = ca[:,0] + y*y*one_minus_ca[:,0]
    R[:,1,2] = y*z*one_minus_ca[:,0] - x*sa[:,0]
    R[:,2,0] = z*x*one_minus_ca[:,0] - y*sa[:,0]
    R[:,2,1] = z*y*one_minus_ca[:,0] + x*sa[:,0]
    R[:,2,2] = ca[:,0] + z*z*one_minus_ca[:,0]
    return R


def one_sided_chamfer_loss(src_transformed, nearest_tgt_points, loss_type="l2", nn_indices=None, c=0.2, reduce="mean"):
    # Gather the nearest target points using the nearest indices
    # nearest_tgt_points = tgt_points[nearest_indices]

    if loss_type == "l1":
        dists = torch.abs(src_transformed - nearest_tgt_points).sum(dim=1)
    elif loss_type == "l2":
        dists = torch.norm(src_transformed - nearest_tgt_points, dim=1) # shape (N,)
    elif loss_type == "smooth_l1":
        criterion = SmoothL1Loss()
        dists = criterion(src_transformed, nearest_tgt_points)
    elif loss_type == "hubert":
        criterion = nn.HuberLoss()
        dists = criterion(src_transformed, nearest_tgt_points)
    elif loss_type == "welsch":
        x = (src_transformed - nearest_tgt_points) / float(c)
        dists = (1.0 - torch.exp(x.pow(2) * -0.5)).sum(dim=1)
    elif loss_type == "geman":
        x = (src_transformed - nearest_tgt_points) / float(c)
        dists = (2 * x.pow(2) / (x.pow(2) + 4)).sum(dim=1)
    else:   
        raise ValueError("loss_type must be either 'l1', 'l2', or 'smooth_l1'")
    # Compute the L2 distances between each transformed source point and its nearest target point
    # dists = torch.norm(src_transformed - nearest_tgt_points, dim=1)

    # if use nn_indices, meaning we want to use a soft version of the distances where the distance of each pair is replaced by the average of its neighbours
    if nn_indices is not None:
        num_pts = src_transformed.shape[0]
        smooth_knn = nn_indices.shape[1]
        nn_dists = dists.view(num_pts, 1).expand(num_pts, num_pts).gather(
            0,
            nn_indices,
        )
        dists = nn_dists.mean(dim=1)

    # Compute the mean distance as the loss
    if reduce == "mean":
        loss = dists.mean()
    else:
        loss = dists.sum()
    return loss


def point_to_plane_loss(src_transformed, tgt_points, tgt_normals, nearest_indices=None):
    # Gather the nearest target points using the nearest indices
    # nearest_tgt_points = tgt_points[nearest_indices]
    # nearest_tgt_normals = tgt_normals[nearest_indices]

    # Compute the L2 distances between each transformed source point and its nearest target point
    dists = torch.abs(
       ((src_transformed - tgt_points) * tgt_normals).sum(dim=1)
    )

    # Compute the mean distance as the loss
    loss = dists.mean()

    return loss


class ARAPLoss(nn.Module):
    def __init__(self, base_pc, smooth_knn=10):
        super().__init__()
        self.num_pts = base_pc.shape[0]
        self.nn_init_positions = base_pc
        self.device = base_pc.device
        self.smooth_knn = smooth_knn
        self.nn_indices = torch.empty(self.num_pts, smooth_knn, dtype=torch.int64, device=self.device)
        self.nn_distances = torch.empty(self.num_pts, smooth_knn, dtype=torch.float32, device=self.device)
        for pt_idx in range(self.num_pts):
            # find the distance from the point at index i to all others points
            displacement_to_all_pts = (
                base_pc[pt_idx : pt_idx + 1, :].expand(self.num_pts, 3)
                - base_pc
            )
            dists, inds = torch.topk(
                displacement_to_all_pts.pow(2).sum(dim=1),
                smooth_knn + 1,
                largest=False,
                sorted=True,
            )
            self.nn_indices[pt_idx, :] = inds[1:].type(torch.int64)
            self.nn_distances[pt_idx, :] = dists[1:]
        
    def forward(self, pc_transformed, add_lda=False, lda_weight=1.0):
        nn_positions = pc_transformed.view(self.num_pts, 1, 3).expand(self.num_pts, self.num_pts, 3).gather(
                0,
                self.nn_indices[:, :self.smooth_knn]
                # .to(self.device)
                .unsqueeze(-1)
                .expand(self.num_pts, self.smooth_knn, 3),
            )
        total_displacement_after_update = (
            pc_transformed.view(self.num_pts, 1, 3).expand(
                self.num_pts, self.smooth_knn, 3
            )
            - nn_positions
        )
        loss = (total_displacement_after_update.pow(2).sum(dim=2)
                    - self.nn_distances[:, :self.smooth_knn]
                ).abs().sum() / (self.num_pts * self.smooth_knn)

        if add_lda:
            avg_displacement = (nn_positions - self.nn_init_positions.view(self.num_pts, 1, 3)
            .expand(self.num_pts, self.num_pts, 3)
            .gather(
                0,
                self.nn_indices
                # .to(device)
                .unsqueeze(-1)
                .expand(self.num_pts, self.smooth_knn, 3),
            )).mean(dim=1)
            loss += lda_weight * (
                    (
                        pc_transformed
                        - (avg_displacement + self.nn_init_positions)
                    )
                    .abs().sum()
                ) / (self.num_pts * 3)
        return loss


def LDA(all_pcs, smooth_knn=10):
    base_pc = all_pcs[0]
    num_pts = base_pc.shape[0]
    device = base_pc.device
    num_steps = all_pcs.shape[0]
    print(f"Running LDA Total num steps {num_steps} with total num_pts {num_pts}")
    nn_indices = torch.empty(num_pts, smooth_knn, dtype=torch.int64, device=device)
    for pt_idx in range(num_pts):
        # find the distance from the point at index i to all others points
        displacement_to_all_pts = (
            base_pc[pt_idx : pt_idx + 1, :].expand(num_pts, 3)
            - base_pc
        )
        _, inds = torch.topk(
            displacement_to_all_pts.pow(2).sum(dim=1),
            smooth_knn + 1,
            largest=False,
            sorted=True,
        )
        nn_indices[pt_idx, :] = inds[1:].type(torch.int64)

    for time_step in range(1, num_steps):
        avg_displacement = (
            all_pcs[time_step]
            .view(num_pts, 1, 3)
            .expand(num_pts, num_pts, 3)
            .gather(
                0,
                nn_indices
                .to(device)
                .unsqueeze(-1)
                .expand(num_pts, smooth_knn, 3),
            )
            - base_pc.view(num_pts, 1, 3)
            .expand(num_pts, num_pts, 3)
            .gather(
                0,
                nn_indices
                .to(device)
                .unsqueeze(-1)
                .expand(num_pts, smooth_knn, 3),
            )
        ).mean(dim=1)
        all_pcs[time_step] = avg_displacement + base_pc


def linear_sum_correspondence(src_trans_np, tgt_np, cur_tgt_normals, use_cos=False, alpha=0.75):
    # Estimate surface normals for source and target key points
    cur_src_normals = estimate_normals_pytorch3d(src_trans_np)
    # Compute the Euclidean distance matrix
    euclidean_cost_matrix = np.linalg.norm(
        src_trans_np[:, np.newaxis, :] - tgt_np[np.newaxis, :, :],
        axis=2,
    )

    # Compute the feature distance matrix (e.g., Euclidean distance between normals)
    if use_cos:
        src_norms = np.linalg.norm(cur_src_normals, axis=1)
        tgt_norms = np.linalg.norm(cur_tgt_normals, axis=1)
        dot_products = np.einsum('ik,jk->ij', cur_src_normals, cur_tgt_normals)
        norms_product = np.outer(src_norms, tgt_norms)
        cos_similarity = dot_products / (norms_product + 1e-8)
        feature_cost_matrix = 1.0 - cos_similarity

        cost_matrix = euclidean_cost_matrix * (1 + feature_cost_matrix)
    else:
        feature_cost_matrix = np.linalg.norm(
            cur_src_normals[:, np.newaxis, :] - cur_tgt_normals[np.newaxis, :, :],
            axis=2,
        )
        # Combine the two matrices to form the final cost matrix
        # alpha = 0.75  # Weighting factor for combining the distances
        # alpha = 0.5  # Weighting factor for combining the distances
        cost_matrix = alpha * euclidean_cost_matrix + (1 - alpha) * feature_cost_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = list(zip(row_ind, col_ind))

    new_tgt_kp = np.zeros_like(tgt_np)
    new_tgt_normals = np.zeros_like(cur_tgt_normals)
    tgt_indices = np.zeros(len(new_tgt_kp), dtype=np.int64)

    src_indices = np.array([src_idx for src_idx, _ in mapping])
    tgt_indices_arr = np.array([tgt_idx for _, tgt_idx in mapping])
    new_tgt_kp[src_indices] = tgt_np[tgt_indices_arr]
    new_tgt_normals[src_indices] = cur_tgt_normals[tgt_indices_arr]
    tgt_indices[src_indices] = tgt_indices_arr

    # # final_ori_tgt_kp = np.zeros_like(cur_src_kp)
    # for src_idx, tgt_idx in mapping:
    #     new_tgt_kp[src_idx] = tgt_np[tgt_idx]
    #     new_tgt_normals[src_idx] = cur_tgt_normals[tgt_idx]
    #     tgt_indices[src_idx] = tgt_idx
    #     # final_ori_tgt_kp[src_idx] = ori_tgt_kp[tgt_idx]
    return new_tgt_kp, new_tgt_normals, tgt_indices


def estimate_volume(point_cloud):
    """
    Estimate the volume of a convex shape represented by a point cloud.

    :param point_cloud: numpy array of shape (N, 3), where N is the number of points.
    :return: Volume of the convex hull.
    """
    hull = ConvexHull(point_cloud)
    return hull.volume


def find_wing_anchor_point_torch(child_pts, parent_pts, k=5):
    """
    Finds an 'anchor' point (joint location) in the wing point cloud
    by identifying which points lie closest to the body. Then returns
    the centroid of the top k closest wing points.

    Args:
        child_pts: (N, 3) torch.FloatTensor of 3D coordinates for the wing
        parent_pts: (M, 3) torch.FloatTensor of 3D coordinates for the body
        k: Number of nearest wing points (to the body) to average if you
           want a centroid.

    Returns:
        anchor_point: (3,) torch.FloatTensor representing the anchor location in the wing
    """

    # Ensure inputs are the correct shape
    # (N, 3) for child_pts, (M, 3) for parent_pts
    # Optionally, you can add checks or assertions here.

    # 1) Compute the distance of each wing point to every body point (naive approach).
    #    For large point clouds, consider more efficient methods (KD-Tree / nearest neighbor searches).
    #    distances will be shape (N, M).
    distances = torch.norm(
        child_pts.unsqueeze(1) - parent_pts.unsqueeze(0),
        dim=2
    )  # (N, M)

    # 2) For each wing point, find its minimal distance to any body point.
    #    This effectively measures how close that wing point is to the body.
    #    min_dist_to_body will be (N,) containing the minimal distance for each of the N wing points.
    min_dist_to_body, _ = torch.min(distances, dim=1)  # shape (N,)

    # 3) Identify the 'k' wing points that are closest to the body.
    #    We sort the distances to find the top k.
    #    The result of argsort is a (N,) array of indices.
    #    Then we take the first k to get the k closest.
    closest_indices = torch.argsort(min_dist_to_body)[:k]

    # The boundary region is the set of these k closest wing points.
    boundary_region = child_pts[closest_indices]

    # 4) The anchor point can be the centroid (average) of these boundary points.
    anchor_point = boundary_region.mean(dim=0)  # shape (3,)

    return anchor_point


class SVDAlignmentLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, keypoints_a, joint_a, keypoints_b, joint_b):
        """
        Calculates the SVD-based alignment loss.

        Args:
            keypoints_a (torch.Tensor): Tensor of shape (N, 3) for the first set of keypoints.
            joint_a (torch.Tensor): Tensor of shape (1, 3) or (3,) for the joint of the first set.
            keypoints_b (torch.Tensor): Tensor of shape (N, 3) for the second set of keypoints.
            joint_b (torch.Tensor): Tensor of shape (1, 3) or (3,) for the joint of the second set.

        Returns:
            torch.Tensor: Scalar loss value.
        """
        if keypoints_a.shape[0] == 0 or keypoints_b.shape[0] == 0:
            return torch.tensor(0.0, device=keypoints_a.device, dtype=keypoints_a.dtype)
        if keypoints_a.shape != keypoints_b.shape:
            raise ValueError("Keypoint sets A and B must have the same shape.")

        # Ensure joints are (1, 3) for broadcasting
        j_a = joint_a.view(1, 3)
        j_b = joint_b.view(1, 3)

        # 1. Calculate Relative Vectors
        vecs_a = keypoints_a - j_a
        vecs_b = keypoints_b - j_b

        # 2. Normalize Relative Vectors
        norm_vecs_a = F.normalize(vecs_a, p=2, dim=1, eps=self.epsilon)
        norm_vecs_b = F.normalize(vecs_b, p=2, dim=1, eps=self.epsilon)

        # 3. Compute the 3x3 covariance matrix H
        # H = sum(v_a_i * v_b_i^T)
        # v_a_i is (N, 3, 1), v_b_i^T is (N, 1, 3)
        # H = torch.matmul(norm_vecs_a.unsqueeze(2), norm_vecs_b.unsqueeze(1)).sum(dim=0)
        H = torch.matmul(norm_vecs_a.T, norm_vecs_b)  # (3, N) @ (N, 3) -> (3,3)

        # 4. Perform Singular Value Decomposition (SVD) on H
        try:
            U, S, Vh = torch.linalg.svd(H)  # Vh is V.T
            V = Vh.T
        except torch.linalg.LinAlgError as e:
            # Handle cases where SVD might fail (e.g., if H is degenerate)
            # This could happen if all points are collinear or coincident
            print(
                f"SVD failed: {e}. Returning high loss or zero loss based on context."
            )
            # Option 1: Return a high loss
            # return torch.tensor(float('inf'), device=H.device, dtype=H.dtype)
            # Option 2: Return sum of squared differences without rotation (essentially what happens if R is identity)
            diff = norm_vecs_a - norm_vecs_b
            loss = torch.sum(diff * diff)
            return loss

        # 5. Calculate the optimal rotation R
        R = torch.matmul(V, U.T)

        # 6. Ensure det(R) = +1 (reflection handling)
        if torch.linalg.det(R) < 0:
            # print("SVDAlignmentLoss: Correcting reflection in R.")
            V_corrected = V.clone()
            V_corrected[:, -1] *= -1  # Flip the sign of the last column of V
            R = torch.matmul(V_corrected, U.T)
            # Re-check determinant, though it should be positive now
            # if torch.linalg.det(R) < 0:
            #     print("SVDAlignmentLoss: Warning - determinant still negative after correction.")

        # 7. Calculate the loss
        # L_SVD = sum(||v_a_i - R @ v_b_i||^2)
        # R @ v_b_i means R.mm(v_b_i.T)).T
        rotated_norm_vecs_b = torch.matmul(
            R, norm_vecs_b.T
        ).T  # (3,3) @ (3,N) -> (3,N) -> (N,3)

        diff_after_rotation = norm_vecs_a - rotated_norm_vecs_b
        loss = torch.sum(
            diff_after_rotation * diff_after_rotation
        )  # Sum of squared errors

        return loss


def simple_joint_loss(keypoints_a, joint_a, keypoints_b, joint_b):
    j_a = joint_a.view(1, 3)
    j_b = joint_b.view(1, 3)

    # 1. Calculate Relative Vectors
    vecs_a = torch.mean(keypoints_a - j_a, dim=0, keepdim=True)
    vecs_b = torch.mean(keypoints_b - j_b, dim=0, keepdim=True)

    loss = torch.norm(vecs_a - vecs_b, p=2, dim=1).sum()
    return loss


def modified_icp_nn_new(
    reference_points,
    deforming_points,
    model_dict,
    log_dir,
    iterations=1e3,
    num_segments=1,
    # mask_ratio_schedule=None,
    loss_mode="chamfer",
    part_max_freqs=None,
    checkpoint=None,
    plot_match=False,
    loss_type="l2",
    correspondence="nn",
    lrt=1e-3,
    src_iter=0,
    iter_schedule=None,
    use_arap=False,
    smooth_knn=10,
    arap_w=1,
    joint_arap=False,
    avg_loss=False,
    robust_c=0.2,
    c_schedule=None,
    add_lda=False,
    lda_weight=1.0,
    early_stop=False,
    stop_volume_ratio=0.97,
    stop_volume_error_rates=[],
    init_volumes=[],
    # joint_pos=[],
    joint_k=1,
    min_iter=100,
    save_logs=True,
    time_step=0,
    save_img_dir=None,
    reference_normals=None,
    use_svd_loss=False,
    root_node_name="Torso",
    w_root=10.0,
    root_warmup_iters=100,
):
    if save_logs:
        save_path = os.path.join(
            log_dir,
            # f"model_nl_{model.num_layers}_nc_{model.base_channels}_mask_freq_{model.mask_freq}_num_parts{len(reference_points.keys())}_iter{iterations}_jointK{joint_k}/",
            f"num_parts{len(reference_points.nodes_by_name.keys())}_iter{iterations}_jointK{joint_k}/",
        )
        if use_arap:
            if joint_arap:
                save_path = save_path[:-1] + f"_joint_arap{arap_w}_k{smooth_knn}/"
            else:
                save_path = save_path[:-1] + f"_arap{arap_w}_k{smooth_knn}/"
        if use_svd_loss:
            save_path = save_path[:-1] + f"_svd_wRoot{w_root}_warmup{root_warmup_iters}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # plot_pointcloud(
        #     torch.cat(tgt, dim=0), save_path, title=f"Target Point Cloud", scale=VISIUAL_SCALE
        # )
        match_frames = []

    if use_arap and not joint_arap:
        arap_losses = {
            name: ARAPLoss(
                deforming_points.nodes_by_name[name].points, smooth_knn=smooth_knn
            )
            for name in reference_points.nodes_by_name.keys()
        }

    if use_svd_loss:
        svd_loss = SVDAlignmentLoss()
    # elif use_arap and joint_arap:
    #     arap_loss = ARAPLoss(torch.cat(src, dim=0), smooth_knn=smooth_knn)

    # if early_stop:
    #     init_volumes = [estimate_volume(src[i].cpu().numpy()) for i in range(len(src))] if len(base_volumes) == 0 else base_volumes

    total_params = []
    for model in model_dict.values():
        model.to(device)
        model.train()
        # optimizer = torch.optim.Adam(model.parameters(), lr=lrt)
        total_params += list(model.parameters())
    optimizer = torch.optim.Adam(total_params, lr=lrt)
    stop = False
    # new_tgt = None
    new_tgts = {}

    segment_length = iterations // num_segments
    cur_seg_idx = 0
    cur_c_seg_idx = 0
    cur_c = robust_c[cur_c_seg_idx] if isinstance(robust_c, list) else robust_c

    loss_history = []

    if use_svd_loss:
        total_num_pts = 0
        for name, c_src in reference_points.nodes_by_name.items():
            if name != root_node_name:
                total_num_pts += len(c_src.points) 
    else:
        total_num_pts = sum(
            [len(c_src.points) for c_src in reference_points.nodes_by_name.values()]
        )

    if correspondence == "linear_sum" and reference_normals is None:
        reference_normals = {
            name: estimate_normals_pytorch3d(node.points.cpu().numpy())
            for name, node in reference_points.nodes_by_name.items()
        }

    for it in tqdm.tqdm(range(iterations), desc="IMP Iterations"):
        if iter_schedule is not None:
            if len(iter_schedule) and it >= iter_schedule[0]:
                cur_seg_idx += 1
                # pop the first element of iter_schedule
                iter_schedule = iter_schedule[1:]
                tqdm.tqdm.write(f"Increasing network frequency to index {cur_seg_idx}")
            segment_index = cur_seg_idx
        else:
            segment_index = it // segment_length
        if c_schedule is not None:
            if len(c_schedule) and it >= c_schedule[0]:
                cur_c_seg_idx += 1
                # pop the first element of iter_schedule
                c_schedule = c_schedule[1:]
                tqdm.tqdm.write(f"Upating robustness constant to index {cur_c_seg_idx}")
            cur_c = robust_c[cur_c_seg_idx]
        src_transformed = []
        tgt_nn = []
        # tgt_normal_shuffled = []
        total_arap_loss = 0.
        loss = 0.
        # losses = []
        # root_R = None
        # root_t = None
        all_tgt_indices = {}

        output_params = {}
        input_dict = deforming_points.generate_model_input_dict()
        for name, cur_pts in input_dict.items():
            output_params[name] = model_dict[name](cur_pts)
        # params = model(input_pts, masks)  # (N, 6)
        deforming_points.store_model_output_dict(output_params)
        deforming_points.update_world_transforms()

        for name in sorted(reference_points.nodes_by_name.keys()):
            if it < root_warmup_iters and name != root_node_name:
                # skip the root node for the first few iterations
                continue
            transform_pts = deforming_points.get_world_points(name)
            src_transformed.append(transform_pts)
            transform_pts_np = transform_pts.detach().cpu().numpy()
            # calculate ARAP loss
            if use_arap and not joint_arap:
                total_arap_loss += arap_losses[name](
                    transform_pts, add_lda=add_lda, lda_weight=lda_weight
                )

            reference_kps = reference_points.nodes_by_name[name].points

            if correspondence == "linear_sum":
                new_tgt_kp, new_tgt_normal, tgt_indices = linear_sum_correspondence(
                    transform_pts_np,
                    reference_kps.cpu().numpy(),
                    reference_normals[name],
                    use_cos=False,
                    alpha=0.75,
                )
                tgt_nn.append(torch.from_numpy(new_tgt_kp).float().to(device))
                # tgt_normal_shuffled.append(torch.from_numpy(new_tgt_normal).float().to(device))
                all_tgt_indices[name] = tgt_indices
            else:
                raise ValueError(
                    "Correspondence method not supported. Use 'linear_sum' or 'nn'."
                )

            cur_ratio = (
                estimate_volume(transform_pts_np) / init_volumes[name]
                if init_volumes
                else 1.0
            )
            if (
                init_volumes
                and (
                    cur_ratio < stop_volume_ratio[name] - stop_volume_error_rates[name]
                )
                or (cur_ratio > stop_volume_ratio[name] + stop_volume_error_rates[name])
            ):
                if name not in new_tgts:
                    print(f"!! Changing target for part {name}")
                    new_tgts[name] = transform_pts.clone().detach()
                if use_svd_loss:
                    if name == root_node_name:
                        root_loss = one_sided_chamfer_loss(
                            transform_pts,
                            new_tgts[name],
                            loss_type=loss_type,
                            c=cur_c,
                            reduce="mean",
                        )
                    else:
                        # loss += svd_loss(
                        #     new_tgts[name],
                        #     reference_points.get_joint_positions_stored_in_parent(name),
                        #     transform_pts,
                        #     deforming_points.get_joint_positions_stored_in_parent(name),
                        # )
                        loss += simple_joint_loss(
                            new_tgts[name],
                            reference_points.get_joint_positions_stored_in_parent(name),
                            transform_pts,
                            deforming_points.get_joint_positions_stored_in_parent(name),
                        )
                else:
                    loss += one_sided_chamfer_loss(
                        transform_pts,
                        new_tgts[name],
                        loss_type=loss_type,
                        c=cur_c,
                        reduce="sum",
                    )
            else:
                if use_svd_loss:
                    if name == root_node_name:
                        root_loss = one_sided_chamfer_loss(
                            transform_pts,
                            tgt_nn[-1],
                            loss_type=loss_type,
                            c=cur_c,
                            reduce="mean",
                        )
                    else:
                        # loss += svd_loss(
                        #     tgt_nn[-1],
                        #     reference_points.get_joint_positions_stored_in_parent(name),
                        #     transform_pts,
                        #     deforming_points.get_joint_positions_stored_in_parent(name),
                        # )
                        loss += simple_joint_loss(
                            tgt_nn[-1],
                            reference_points.get_joint_positions_stored_in_parent(name),
                            transform_pts,
                            deforming_points.get_joint_positions_stored_in_parent(name),
                        )
                else:
                    loss += one_sided_chamfer_loss(
                        transform_pts,
                        tgt_nn[-1],
                        loss_type=loss_type,
                        c=cur_c,
                        reduce="sum",
                    )

        src_transformed = torch.cat(src_transformed, dim=0)
        tgt_nn = torch.cat(tgt_nn, dim=0)
        # if len(tgt_normal_shuffled) > 0:
        #     tgt_normal_shuffled = torch.cat(tgt_normal_shuffled, dim=0)

        # if use_arap and joint_arap:
        #     total_arap_loss = arap_loss(src_transformed, add_lda=add_lda, lda_weight=lda_weight)

        # if use_arap:
        loss = loss * 1. / total_num_pts + total_arap_loss * arap_w
        if use_svd_loss:
            loss += root_loss * w_root
            # if it >= root_warmup_iters:
            #     w_root = 1.0

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if save_logs:
            tqdm.tqdm.write(f"Iteration {it} Loss ({loss_mode}): {loss.item():.6f}, ARAP Loss: {total_arap_loss:.6f}")

        loss_history.append(loss.item())
        # check loss plateau
        if it > min_iter and len(loss_history) > min_iter:
            if np.mean(loss_history[-10:]) > np.mean(loss_history[-20:-10]):
                stop = True
            else:
                stop = False

        if save_logs:
            # Save the transformed point cloud
            if it % 10 == 0 or it == iterations - 1 or (stop and early_stop):
                plot_pointcloud(
                    src_transformed, save_path, title=f"Transformed Point Cloud Iter {it}", scale=VISIUAL_SCALE
                    # extra_points=tgt, plot_match=False
                )
                if plot_match:
                    # plot a vector that connects each point to the corresponding nearest neighbor
                    cur_frame = plot_pointcloud(
                        src_transformed,
                        None,
                        title=f"Match Iter {it}",
                        extra_points=tgt_nn,
                        plot_match=True,
                        scale=VISIUAL_SCALE,
                    )
                    match_frames.append(cur_frame)
                    if it % 100 == 0 or (stop and early_stop):
                        imageio.mimsave(os.path.join(save_path, "match_frames.mp4"), match_frames, fps=10)
                        # torch.save(model.state_dict(), os.path.join(save_path, f"deform_model_{it}.pth"))
                        # save the transformation parameters
                        for name, model in model_dict.items():
                            torch.save(model.state_dict(), os.path.join(save_path, f"{name}_model_{it}.pth"))

        if stop and early_stop:
            # save the src_transformed point cloud
            break
        # else:
        #     stop = True
    if plot_match and save_logs:
        imageio.mimsave(os.path.join(save_path, "match_frames.mp4"), match_frames, fps=10)
        # save model weights
        # torch.save(model.state_dict(), os.path.join(save_path, f"deform_model_final.pth"))
        # torch.save(transformations, os.path.join(save_path, f"transformation_final.pth"))
        # for m_idx, model in enumerate(models):
        for name, model in model_dict.items():
            torch.save(
                model.state_dict(), os.path.join(save_path, f"{name}_model_{it}.pth")
            )

    save_dir = save_img_dir if time_step % 10 == 0 else None
    cur_frame = plot_pointcloud(
        src_transformed,
        save_dir,
        title=f"Time Step {time_step}",
        extra_points=tgt_nn,
        plot_match=True,
        scale=VISIUAL_SCALE,
    )

    return model_dict, src_transformed.detach(), cur_frame, it, all_tgt_indices


def smooth_point_cloud_torch(point_clouds, window_size):
    """
    Apply temporal smoothing to a sequence of point clouds.

    :param point_clouds: PyTorch tensor of shape (T, N, 3), where T is the number of time steps,
                         N is the number of points, and 3 represents the spatial dimensions.
    :param window_size: Size of the moving average window (must be an odd number).
    :return: Smoothed point clouds as a PyTorch tensor of shape (T, N, 3).
    """
    assert window_size % 2 == 1, "Window size must be an odd number."

    print(point_clouds.shape)
    # Pad the point clouds along the temporal dimension
    padding = window_size // 2
    padded_point_clouds = torch.cat(
        [point_clouds[:1].repeat(padding, 1, 1),  # Repeat the first frame
         point_clouds,
         point_clouds[-1:].repeat(padding, 1, 1)],  # Repeat the last frame
        dim=0
    )

    # Apply the moving average filter
    smoothed_point_clouds = []
    for t in range(padding, len(padded_point_clouds) - padding):
        # Get the window of point clouds for this time step
        window = padded_point_clouds[t - padding:t + padding + 1]
        # Calculate the average point cloud for this window
        average_point_cloud = window.mean(dim=0)
        smoothed_point_clouds.append(average_point_cloud)

    # Stack the smoothed point clouds along the temporal dimension
    smoothed_point_clouds = torch.stack(smoothed_point_clouds, dim=0)

    return smoothed_point_clouds


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


class Deformation(ABC):
    """
    Abstract class for generic deformation.
    This class should be inherited for the development of new deformation
    techniques.
    """
 
    @abstractmethod
    def __init__(self, value):
        pass

    @abstractmethod
    def __call__(self, src):
        pass


class RBF(Deformation):
    """
    Class that handles the Radial Basis Functions interpolation on the mesh
    points.

    :param numpy.ndarray original_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the original
        interpolation control points before the deformation. The default is the
        vertices of the unit cube.
    :param numpy.ndarray deformed_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the
        interpolation control points after the deformation. The default is the
        vertices of the unit cube.
    :param func: the basis function to use in the transformation. Several basis
        function are already implemented and they are available through the
        :py:class:`~pygem.rbf.RBF` by passing the name of the right
        function (see class documentation for the updated list of basis
        function).  A callable object can be passed as basis function.
    :param float radius: the scaling parameter r that affects the shape of the
        basis functions.  For details see the class
        :class:`RBF`. The default value is 0.5.
    :param dict extra_parameter: the additional parameters that may be passed to
    	the kernel function. Default is None.
        
    :cvar numpy.ndarray weights: the matrix formed by the weights corresponding
        to the a-priori selected N control points, associated to the basis
        functions and c and Q terms that describe the polynomial of order one
        p(x) = c + Qx.  The shape is (*n_control_points+1+3*, *3*). It is
        computed internally.
    :cvar numpy.ndarray original_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the original
        interpolation control points before the deformation.
    :cvar numpy.ndarray deformed_control_points: it is an
        (*n_control_points*, *3*) array with the coordinates of the
        interpolation control points after the deformation.
    :cvar callable basis: the basis functions to use in the
        transformation.
    :cvar float radius: the scaling parameter that affects the shape of the
        basis functions.
    :cvar dict extra: the additional parameters that may be passed to the
        kernel function.
        
    :Example:

        >>> from pygem import RBF
        >>> import numpy as np
        >>> rbf = RBF(func='gaussian_spline')
        >>> xv = np.linspace(0, 1, 20)
        >>> yv = np.linspace(0, 1, 20)
        >>> zv = np.linspace(0, 1, 20)
        >>> z, y, x = np.meshgrid(zv, yv, xv)
        >>> mesh = np.array([x.ravel(), y.ravel(), z.ravel()])
        >>> deformed_mesh = rbf(mesh)
    """
    def __init__(self,
                 original_control_points=None,
                 deformed_control_points=None,
                 func='gaussian_spline',
                 radius=0.5,
                 extra_parameter=None):

        self.basis = func
        self.radius = radius

        if original_control_points is None:
            self.original_control_points = np.array([[0., 0., 0.], [0., 0., 1.],
                                                     [0., 1., 0.], [1., 0., 0.],
                                                     [0., 1., 1.], [1., 0., 1.],
                                                     [1., 1., 0.], [1., 1.,
                                                                    1.]])
        else:
            self.original_control_points = original_control_points

        if deformed_control_points is None:
            self.deformed_control_points = np.array([[0., 0., 0.], [0., 0., 1.],
                                                     [0., 1., 0.], [1., 0., 0.],
                                                     [0., 1., 1.], [1., 0., 1.],
                                                     [1., 1., 0.], [1., 1.,
                                                                    1.]])
        else:
            self.deformed_control_points = deformed_control_points

        self.extra = extra_parameter if extra_parameter else dict()

        self.weights = self._get_weights(self.original_control_points,
                                         self.deformed_control_points)

    @property
    def n_control_points(self):
        """
        Total number of control points.

        :rtype: int
        """
        return self.original_control_points.shape[0]

    @property
    def basis(self):
        """
        The kernel to use in the deformation.

        :getter: Returns the callable kernel
        :setter: Sets the kernel. It is possible to pass the name of the
            function (check the list of all implemented functions in the
            `pygem.rbf_factory.RBFFactory` class) or directly the callable
            function.
        :type: callable
        """
        return self.__basis

    @basis.setter
    def basis(self, func):
        if callable(func):
            self.__basis = func
        elif isinstance(func, str):
            self.__basis = RBFFactory(func)
        else:
            raise TypeError('`func` is not valid.')

    # def _get_weights(self, X, Y):
    #     """
    #     This private method, given the original control points and the deformed
    #     ones, returns the matrix with the weights and the polynomial terms, that
    #     is :math:`W`, :math:`c^T` and :math:`Q^T`. The shape is
    #     (*n_control_points+1+3*, *3*).

    #     :param numpy.ndarray X: it is an n_control_points-by-3 array with the
    #         coordinates of the original interpolation control points before the
    #         deformation.
    #     :param numpy.ndarray Y: it is an n_control_points-by-3 array with the
    #         coordinates of the interpolation control points after the
    #         deformation.

    #     :return: weights: the 2D array with the weights and the polynomial terms.
    #     :rtype: numpy.ndarray
    #     """
    #     npts, dim = X.shape
    #     H = np.zeros((npts + 3 + 1, npts + 3 + 1))
    #     H[:npts, :npts] = self.basis(cdist(X, X), self.radius, **self.extra)
    #     H[npts, :npts] = 1.0
    #     H[:npts, npts] = 1.0
    #     H[:npts, -3:] = X
    #     H[-3:, :npts] = X.T

    #     rhs = np.zeros((npts + 3 + 1, dim))
    #     rhs[:npts, :] = Y
    #     print(H.shape, rhs.shape, np.linalg.det(H))
    #     weights = np.linalg.solve(H, rhs)
    #     return weights

    def _get_weights(self, X, Y):
        """
        This private method, given the original control points and the deformed
        ones, returns the matrix with the weights and the polynomial terms, that
        is :math:`W`, :math:`c^T` and :math:`Q^T`. The shape is
        (*n_control_points+1+3*, *3*).

        :param numpy.ndarray X: it is an n_control_points-by-3 array with the
            coordinates of the original interpolation control points before the
            deformation.
        :param numpy.ndarray Y: it is an n_control_points-by-3 array with the
            coordinates of the interpolation control points after the
            deformation.

        :return: weights: the 2D array with the weights and the polynomial terms.
        :rtype: numpy.ndarray
        """
        npts, dim = X.shape
        H = np.zeros((npts, npts))
        H[:npts, :npts] = self.basis(cdist(X, X), self.radius, **self.extra)

        rhs = np.zeros((npts, dim))
        rhs[:npts, :] = Y
        # print(H.shape, rhs.shape, np.linalg.det(H))
        weights = np.linalg.solve(H, rhs)
        return weights

    def plot_points(self, filename=None):
        """
        Method to plot the control points. It is possible to save the resulting
        figure.

        :param str filename: if None the figure is shown, otherwise it is saved
            on the specified `filename`. Default is None.
        """
        fig = plt.figure(1)
        axes = fig.add_subplot(111, projection='3d')
        orig = axes.scatter(self.original_control_points[:, 0],
                            self.original_control_points[:, 1],
                            self.original_control_points[:, 2],
                            c='blue',
                            marker='o')
        defor = axes.scatter(self.deformed_control_points[:, 0],
                             self.deformed_control_points[:, 1],
                             self.deformed_control_points[:, 2],
                             c='red',
                             marker='x')

        axes.set_xlabel('X axis')
        axes.set_ylabel('Y axis')
        axes.set_zlabel('Z axis')

        plt.legend((orig, defor), ('Original', 'Deformed'),
                   scatterpoints=1,
                   loc='lower left',
                   ncol=2,
                   fontsize=10)

        # Show the plot to the screen
        if filename is None:
            plt.show()
        else:
            fig.savefig(filename)

    def compute_weights(self):
        """
        This method compute the weights according to the
        `original_control_points` and `deformed_control_points` arrays.
        """
        self.weights = self._get_weights(self.original_control_points,
                                         self.deformed_control_points)

    # def __call__(self, src_pts):
    #     """
    #     This method performs the deformation of the mesh points. After the
    #     execution it sets `self.modified_mesh_points`.
    #     """
    #     self.compute_weights()

    #     H = np.zeros((src_pts.shape[0], self.n_control_points + 3 + 1))
    #     H[:, :self.n_control_points] = self.basis(
    #         cdist(src_pts, self.original_control_points),
    #         self.radius,
    #         **self.extra)
    #     H[:, self.n_control_points] = 1.0
    #     H[:, -3:] = src_pts
    #     return np.asarray(np.dot(H, self.weights))

    def __call__(self, src_pts):
        """
        This method performs the deformation of the mesh points. After the
        execution it sets `self.modified_mesh_points`.
        """
        self.compute_weights()

        H = np.zeros((src_pts.shape[0], self.n_control_points))
        H[:, :self.n_control_points] = self.basis(
            cdist(src_pts, self.original_control_points), 
            self.radius,
            **self.extra)
        # H[:, self.n_control_points] = 1.0
        # H[:, -3:] = src_pts
        return np.asarray(np.dot(H, self.weights))


def scale_mesh_to_match_height(mesh_to_scale, target_mesh):
    """
    Scales mesh_to_scale to match the height of target_mesh.

    Args:
        mesh_to_scale: A trimesh.Mesh object to be scaled.
        target_mesh: A trimesh.Mesh object with the target height.

    Returns:
        A new trimesh.Mesh object scaled to the target height.
    """
    # 1. Determine heights
    bbox_to_scale = mesh_to_scale.bounding_box.bounds
    height_to_scale = bbox_to_scale[1, 1] - bbox_to_scale[0, 1]  # Max y - Min y

    bbox_target = target_mesh.bounding_box.bounds
    target_height = bbox_target[1, 1] - bbox_target[0, 1]  # Max y - Min y

    if height_to_scale == 0:
        print("Warning: Mesh to scale has zero height.")
        return mesh_to_scale

    # 2. Calculate scaling factor
    scaling_factor = target_height / height_to_scale

    # 3. Apply scaling (uniform scaling based on height)
    # Translate mesh_to_scale so its lowest point is at y=0 before scaling
    translation_vector = np.array([0, -bbox_to_scale[0, 1], 0])
    mesh_to_scale.vertices += translation_vector

    # Apply uniform scaling
    scaled_vertices = mesh_to_scale.vertices * scaling_factor

    # Translate the mesh back by the original lowest y-coordinate (scaled)
    scaled_mesh = trimesh.Trimesh(vertices=scaled_vertices, faces=mesh_to_scale.faces)
    final_translation_vector = np.array([0, bbox_to_scale[0, 1] * scaling_factor, 0])
    scaled_mesh.vertices += final_translation_vector

    return scaled_mesh


USE_ICP = True
# USE_FPS = False
# TRAIN_ICP = False
TRAIN_ICP = True
USE_FPS = True
DEFORM_KP_ONLY = True
TRAIN_DEFORM_NET = True
USE_LINEAR_SUM = True
# TRAIN_DEFORM_NET = False
TEST_DEFORM_NET = False
REG_DEFORM_NET = True
# REG_DEFORM_NET = False
# TEST_REG_DEFORM_NET = True
USE_POINTNET = True
# DENSIFY_BODY = True
DENSIFY_BODY = False
CONCAT_POS_DEFORM = True
# USE_COS = True
USE_COS = False
ADJUST_BOUNDARY = False
USE_KEYPOINTS = True
# USE_KEYPOINTS = False

# part_names = ["root", "head_neck", "left_arm", "right_arm", "left_leg", "right_leg"]
# part_names = [
#     "hips",
#     "spine",
#     "leftUpLeg",
#     "leftLeg",
#     "left_foot",
#     "rightUpLeg",
#     "rightLeg",
#     "right_foot",
#     "head",
#     "neck",
#     "spine1",
#     "spine2",
#     "leftShoulder",
#     "leftArm",
#     "leftForeArm",
#     "left_hand",
#     "rightShoulder",
#     "rightArm",
#     "rightForeArm",
#     "right_hand",
# ]
# part_names = [
#     "root",
#     "head_neck",
#     "left_upper_arm",
#     "right_upper_arm",
#     "left_lower_arm",
#     "right_lower_arm",
#     "left_upper_leg",
#     "right_upper_leg",
#     "left_lower_leg",
#     "right_lower_leg",
# ]
parent_lookup = {
    "Torso": None,
    "Head": "Torso",
    "LeftUpperArm": "Torso",
    "LeftLowerArm": "LeftUpperArm",
    "LeftHand": "LeftLowerArm",
    "RightUpperArm": "Torso",
    "RightLowerArm": "RightUpperArm",
    "RightHand": "RightLowerArm",
    "LeftUpLeg": "Torso",
    "LeftLeg": "LeftUpLeg",
    "LeftFoot": "LeftLeg",
    "RightUpLeg": "Torso",
    "RightLeg": "RightUpLeg",
    "RightFoot": "RightLeg",
}
part_names = list(parent_lookup.keys())

total_part_num = len(part_names)

reference_character = "Ch24_nonPBR"
reference_frame = "0046"
reference_obj_path = f"/NAS/spa176/data/Mixamo/Animated/{reference_character}/{reference_frame}.obj"
deforming_character = "Ch17_nonPBR"
deforming_part_indices = dict(
    np.load(
        f"/NAS/spa176/data/Mixamo/Segmentation/{deforming_character}_semantic_masks.npz"
    )
)
reference_part_indices = dict(
    np.load(
        f"/NAS/spa176/data/Mixamo/Segmentation/{reference_character}_semantic_masks.npz"
    )
)

# deforming_obj_mesh = trimesh.load(
#     f"/NAS/spa176/data/Mixamo/Segmentation/{deforming_character}_semantic_vis.ply",
#     process=False,
# )  # process=False preserves vertex order
# deforming_vertices = deforming_obj_mesh.vertices
# deforming_faces = deforming_obj_mesh.faces

deform_dict = dict(np.load(
    f"/NAS/spa176/data/Mixamo/Segmentation/Ch17_nonPBR_vertex_faces.npz"
))
deforming_vertices, deforming_faces = deform_dict.get("vertices"), deform_dict.get("faces")

# deforming_vertices = np.load(
#     f"/NAS/spa176/data/Mixamo/Segmentation/{deforming_character}_rest_verts_faces.npz"
# ).get("rest_verts")

# deforming_faces = np.load(
#     f"/NAS/spa176/data/Mixamo/Segmentation/{deforming_character}_rest_verts_faces.npz"
# ).get("faces")


print(f"Loaded {len(deforming_vertices)} deforming vertices.")
# deforming_vertex_normals = deforming_obj_mesh.vertex_normals
# print(f"Loaded {len(deforming_vertex_normals)} deforming vertex normals.")

ref_obj_mesh = trimesh.load(
    reference_obj_path, process=False
)  # process=False preserves vertex order

# ref_obj_mesh = scale_mesh_to_match_height(ref_obj_mesh, deforming_obj_mesh)
# 1. Get vertices in original order
# This is directly available as a numpy array
ref_vertices = ref_obj_mesh.vertices
print(f"Loaded {len(ref_vertices)} reference vertices.")

# 2. Get vertex normals
# trimesh computes these based on face normals if not directly available or calculates them on access.
# The order corresponds to the mesh.vertices order.
ref_vertex_normals = ref_obj_mesh.vertex_normals
print(f"Loaded {len(ref_vertex_normals)} reference vertex normals.")

# find the max index in the reference_part_indices and check if it is equal to the number of vertices
max_ref_index = max(
    np.max(indices) for indices in reference_part_indices.values() if indices.size > 0
)  # Added check for empty arrays
max_deform_index = max(
    np.max(indices) for indices in deforming_part_indices.values() if indices.size > 0
)  # Added check for empty arrays

print(f"Max reference index: {max_ref_index}")
print(f"Max deforming index: {max_deform_index}")

assert max_ref_index == (
    len(ref_vertices) - 1
), "Reference part indices do not match the number of vertices."
assert max_deform_index == (
    len(deforming_vertices) - 1
), "Deforming part indices do not match the number of vertices."


# fps_k = 1000
if USE_KEYPOINTS:
    # all_num_keypoints = [
    #     256, 256, 256, 256, 128, 128
    # ]
    # all_num_keypoints = {
    #     "head": 256,
    #     "neck": 32,
    #     "spine": 32,
    #     "spine1": 32,
    #     "spine2": 64,
    #     "hips": 64,
    #     "leftShoulder": 32,
    #     "leftArm": 32,
    #     "leftForeArm": 32,
    #     "left_hand": 64,
    #     "rightShoulder": 32,
    #     "rightArm": 32,
    #     "rightForeArm": 32,
    #     "right_hand": 64,
    #     "leftUpLeg": 32,
    #     "leftLeg": 32,
    #     "left_foot": 32,
    #     "rightUpLeg": 32,
    #     "rightLeg": 32,
    #     "right_foot": 32,
    # }
    # all_num_keypoints = {
    #     "root": 256,
    #     "head_neck": 256,
    #     "left_arm": 256,
    #     "right_arm": 256,
    #     "left_leg": 128,
    #     "right_leg": 128,
    # }
    # all_num_keypoints = {
    #     "root": 256,
    #     "head_neck": 256,
    #     "left_upper_arm": 64,
    #     "right_upper_arm": 64,
    #     "left_lower_arm": 64,
    #     "right_lower_arm": 64,
    #     "left_upper_leg": 32,
    #     "right_upper_leg": 32,
    #     "left_lower_leg": 64,
    #     "right_lower_leg": 64,
    # }
    all_num_keypoints = {
        "Torso": 256,
        "Head": 256,
        "LeftUpperArm": 64,
        "LeftLowerArm": 64,
        "LeftHand": 64,
        "RightUpperArm": 64,
        "RightLowerArm": 64,
        "RightHand": 64,
        "LeftUpLeg": 64,
        "LeftLeg": 64,
        "LeftFoot": 64,
        "RightUpLeg": 64,
        "RightLeg": 64,
        "RightFoot": 64,
    }
    # num_keypoint_percentage = 0.5
# else:
#     # all_num_keypoints = [len(part_indices[i]) for i in range(len(part_indices))]
#     all_num_keypoints = {name: len(indices) for name, indices in part_indices.items()}

log_dir = "fit_pointcloud_logs"
exp_dir = f"mixamo"

exp_id = 5

exp_sub_dir = f"{deforming_character}_to_{reference_character}_{exp_id}"
log_dir = os.path.join(log_dir, exp_dir, exp_sub_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
# copy the current file to log_dir
shutil.copy(__file__, log_dir)

save_img_dir = os.path.join(log_dir, "images")
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir, exist_ok=True)

VISIUAL_SCALE = 1.2

src_pc = torch.tensor(ref_vertices).float().to(device)
tgt_pc = torch.tensor(deforming_vertices).float().to(device)

print("src_pc: ", src_pc.shape, src_pc.min(), src_pc.max())
print("tgt_pc: ", tgt_pc.shape, tgt_pc.min(), tgt_pc.max())

# Plot the point clouds
plot_pointcloud(src_pc, log_dir, title="Source Point Cloud", scale=VISIUAL_SCALE)
plot_pointcloud(tgt_pc, log_dir, title="Target Point Cloud", scale=VISIUAL_SCALE)


use_hierarchical_transformation = True
joint_k = 5

# icp_model = PointwiseTransformNet(base_channels=64, max_freq=0).to(device)
temperature = 50.0
# temperature = 25.0
max_freq = 1
mask_freq = 4

# imp_model = PointwiseTransformPartNet(
#     base_channels=64, max_freq=max_freq, temperature=temperature, mask_freq=mask_freq, num_layers=3
# ).to(device)

imp_models = {
    name: PointwiseTransformNet(
        base_channels=32, max_freq=max_freq, temperature=temperature
    ).to(device)
    for name in part_names
}

src_kps = {}
tgt_kps = {}
# tgt_normals = {}
# tgt_pcs = []
src_kp_indices = {}
src_normals = {}
src_volumes = {}

for name in part_names:
    deforming_indices = deforming_part_indices[name]
    ref_indices = reference_part_indices[name]
    if USE_KEYPOINTS:
        # print(f"Part {name}: Selecting {all_num_keypoints[name]} from {len(ref_indices)} reference points and {len(deforming_indices)} deforming points.")
        # print(f"Source pc shape: {src_pc.shape} max reference index: {ref_indices.max()}")
        # print(f"Target pc shape: {tgt_pc.shape} max deforming index: {deforming_indices.max()}")
        # exit(1)
        # FPS
        cur_src_kp, src_kp_idx = sfp(
            src_pc[ref_indices].unsqueeze(0),
            K=min(all_num_keypoints[name], len(ref_indices)),
        )
        cur_src_kp = cur_src_kp.squeeze(0)

        cur_tgt_kp, tgt_kp_idx = sfp(
            tgt_pc[deforming_indices].unsqueeze(0),
            K=min(all_num_keypoints[name], len(deforming_indices)),
        )
        cur_tgt_kp = cur_tgt_kp.squeeze(0)

    #     # plot the keypoints
    #     plot_pointcloud(
    #         cur_src_kp,
    #         log_dir,
    #         title=f"Reference Character Keypoints {name}",
    #         scale=VISIUAL_SCALE,
    #     )
    #     plot_pointcloud(
    #         cur_tgt_kp,
    #         log_dir,
    #         title=f"Deforming Character Keypoints {name}",
    #         scale=VISIUAL_SCALE,
    #     )
    # # else:
    #     cur_src_kp = src_pc[indices]
    #     cur_tgt_kp = tgt_pc[indices]

    src_kps[name] = cur_src_kp
    src_kp_indices[name] = src_kp_idx
    src_volumes[name] = estimate_volume(cur_src_kp.cpu().numpy())
    src_normals[name] = ref_vertex_normals[ref_indices][
        src_kp_idx.squeeze(0).cpu().numpy()
    ]
    tgt_kps[name] = cur_tgt_kp
    # tgt_normals[name] = deforming_vertex_normals[indices][tgt_kp_idx.squeeze(0)]


# parent_lookup = {
#     "hips": None,
#     "spine": "hips",
#     "leftUpLeg": "hips",
#     "leftLeg": "leftUpLeg",
#     "left_foot": "leftLeg",
#     "rightUpLeg": "hips",
#     "rightLeg": "rightUpLeg",
#     "right_foot": "rightLeg",
#     "spine1": "spine",
#     "spine2": "spine1",
#     "neck": "spine2",
#     "head": "neck",
#     "leftShoulder": "spine2",
#     "leftArm": "leftShoulder",
#     "leftForeArm": "leftArm",
#     "left_hand": "leftForeArm",
#     "rightShoulder": "spine2",
#     "rightArm": "rightShoulder",
#     "rightForeArm": "rightArm",
#     "right_hand": "rightForeArm"
# }
# parent_lookup = {
#     "root": None,
#     "head_neck": "root",
#     "left_arm": "root",
#     "right_arm": "root",
#     "left_leg": "root",
#     "right_leg": "root",
# }
# parent_lookup = {
#     "root": None,
#     "head_neck": "root",
#     "left_upper_arm": "root",
#     "right_upper_arm": "root",
#     "left_lower_arm": "left_upper_arm",
#     "right_lower_arm": "right_upper_arm",
#     "left_upper_leg": "root",
#     "right_upper_leg": "root",
#     "left_lower_leg": "left_upper_leg",
#     "right_lower_leg": "right_upper_leg",
# }

# 1. Setup the Hierarchy Structure
hierarchy = Hierarchy(device=device)
ref_hierarchy = Hierarchy(device=device)
num_joint_points = 5
for name, parent in parent_lookup.items():
    hierarchy.add_node(
        name,
        points=tgt_kps[name],
        # normals=tgt_normals[name],
        num_joint_points=num_joint_points,
        parent_name=parent,
    )
    ref_hierarchy.add_node(
        name,
        points=src_kps[name],
        # normals=tgt_normals[name],
        num_joint_points=num_joint_points,
        parent_name=parent,
    )


src_volume_ratios = {name: 1.0 for name in part_names}
part_error_rates = {name: 0.01 for name in part_names}
# part_error_rates


# if use_hierarchical_transformation:
#     # find the joint position

#     joint_pos = [None]
#     for part_idx in range(1, total_part_num, 1):
#         cur_joint_pos = find_wing_anchor_point_torch(
#             tgt_kps[part_idx], tgt_kps[0], k=joint_k
#         )
#         joint_pos.append(cur_joint_pos.unsqueeze(0))

# src_volume_ratios = [1. for _ in range(total_part_num)]

# # loop through src_keypoints and calculate the volume ratio at each time step
# src_volume_ratios = [[] for _ in range(total_part_num)]
# for part_idx in range(total_part_num):
#     for time_step in range(len(src_keypoints[part_idx])):
#         cur_src_kp = src_keypoints[part_idx][time_step]
#         cur_src_volume = estimate_volume(cur_src_kp.cpu().numpy())
#         src_volume_ratios[part_idx].append(cur_src_volume * 1. / src_volumes[part_idx])


total_time_steps = 1

transferred_tgt_kps = []
match_frames = []
# TRAIN_ICP_MODEL = False
TRAIN_ICP_MODEL = True
base_volumes = {}
if TRAIN_ICP_MODEL:
    # loop through each time step
    for time_step in tqdm.tqdm(range(total_time_steps)):

        # cur_iterations = 800 if time_step == 0 else 500
        cur_min_iter = 150 if time_step == 0 else 10
        save_log = True if time_step == 0 else False
        cur_src_kps = src_kps
        # cur_src_kps = [src_keypoint[time_step] for src_keypoint in src_keypoints]
        correspondence = "linear_sum"

        if time_step > 0:
            cur_src_kps = [cur_src_kp[initial_tgt_indices[part_idx]] for part_idx, cur_src_kp in enumerate(cur_src_kps)]
            correspondence = None

        # print(f"Step {time_step}, ratios: {[src_volume_ratio[time_step] for src_volume_ratio in src_volume_ratios]}")

        imp_models, src_transformed, match_frame, train_iters, all_tgt_indices = (
            modified_icp_nn_new(
                ref_hierarchy,
                hierarchy,
                imp_models,
                log_dir,
                # iterations=600,
                iterations=500,
                # num_segments=max_freq,
                # num_segments=1,
                loss_mode="chamfer",
                part_max_freqs=[max_freq for _ in range(total_part_num)],
                plot_match=True,
                # loss_type="hubert",
                # loss_type="geman",
                # loss_type="p2p"
                correspondence=correspondence,
                # lrt=2.5e-3,
                # src_iter=src_iter,
                use_arap=True,
                smooth_knn=10,
                arap_w=10,
                joint_arap=True,
                # iter_schedule=[100],
                # avg_loss=True,
                # robust_c=0.05,
                # robust_c=[1.0, 0.05, 0.01],
                # c_schedule=[300, 600],
                # add_lda=True,
                # lda_weight=0.025,
                early_stop=True,
                # stop_volume_ratio=[
                #     src_volume_ratio[time_step]
                #     for src_volume_ratio in src_volume_ratios
                # ],
                stop_volume_ratio=src_volume_ratios,
                init_volumes=base_volumes,
                stop_volume_error_rates=part_error_rates,
                # stop_volume_ratio=0.97,
                # loss_modes=["chamfer", "pt2plane", "pt2plane"],
                # hierarchical_transformation=use_hierarchical_transformation,
                # joint_pos=joint_pos,
                joint_k=num_joint_points,
                save_logs=save_log,
                min_iter=cur_min_iter,
                save_img_dir=save_img_dir,
                time_step=time_step,
                use_svd_loss=True,
                w_root=10.0,
            )
        )

        # if time_step == 0:
        #     initial_tgt_indices = all_tgt_indices
        #     # update base volumes by calculating the src_transformed for now, slice the src_transformed by the number of keypoints in each part
        #     for part_idx in range(total_part_num):
        #         start_idx = 0 if part_idx == 0 else start_idx + len(src_kp_indices[part_idx - 1][0])
        #         end_idx = start_idx + len(src_kp_indices[part_idx][0])
        #         base_volumes.append(estimate_volume(src_transformed[start_idx:end_idx].cpu().numpy()))

        #     exit(0)

        tqdm.tqdm.write(f"Time step {time_step} converged in {train_iters} iterations")

        match_frames.append(match_frame)
        if time_step % 10 == 0:
            imageio.mimsave(os.path.join(save_img_dir, "match_frames.mp4"), match_frames, fps=10)

        transferred_tgt_kps.append(src_transformed.unsqueeze(0))
        # torch.save(torch.cat(transferred_tgt_kps, dim=0) * scale, os.path.join(log_dir, f"transferred_tgt_kps.pth"))
        torch.save(
            torch.cat(transferred_tgt_kps, dim=0),
            os.path.join(log_dir, f"transferred_tgt_kps.pth"),
        )

    # save the match frames to save_img_dir as mp4
    if len(match_frames) > 0:
        imageio.mimsave(os.path.join(save_img_dir, "match_frames.mp4"), match_frames, fps=10)

    # transferred_tgt_kps = torch.cat(transferred_tgt_kps, dim=0) * scale
    transferred_tgt_kps = torch.cat(transferred_tgt_kps, dim=0)
    # save the transferred_tgt_kps to disk as torch tensor
    # torch.save(transferred_tgt_kps, os.path.join(log_dir, f"transferred_tgt_kps.pth"))

    # smoothed_total_deformed_pcs = smooth_point_cloud_torch(transferred_tgt_kps, smooth_window_size)
    # torch.save(smoothed_total_deformed_pcs, os.path.join(log_dir, f"transferred_tgt_kps_smooth.pth"))

    # save numpy version
    transferred_tgt_kps_np = transferred_tgt_kps.cpu().numpy()
    if USE_KEYPOINTS:
        save_pt_name = "kps"
    else:
        save_pt_name = "pts"
    np.save(
        os.path.join(log_dir, f"deformed_{save_pt_name}.npy"), transferred_tgt_kps_np
    )
    # smoothed_total_deformed_pcs = smoothed_total_deformed_pcs.cpu().numpy()
    # np.save(
    #     os.path.join(log_dir, f"deformed_{save_pt_name}_smooth.npy"),
    #     smoothed_total_deformed_pcs,
    # )

else:
    transferred_tgt_kps = torch.load(os.path.join(log_dir, f"transferred_tgt_kps.pth"))
    print("Loaded", transferred_tgt_kps.shape)

if USE_KEYPOINTS:
    rbf_kernel = "polyharmonic_spline"
    ext_params = {"k": 3} if rbf_kernel == "polyharmonic_spline" else {}
    rbf_radius = 10

    print("=" * 20)
    print("Motion transfer finished, start to recover the full point cloud")

    # NOTE: assume only one time step for now
    total_deformed_pcs = np.zeros_like(deforming_vertices)
    for cur_step in tqdm.tqdm(range(total_time_steps)):
        # cur_deformed_pcs = []
        # for part_idx in range(total_part_num):
        for name in sorted(src_kps.keys()):
            # start_idx = 0 if part_idx == 0 else start_idx + len(src_kp_indices[part_idx - 1][0])
            # end_idx = start_idx + len(src_kp_indices[part_idx][0])
            # if cur_step == 0:
            #     print(f"Slicing part_idx: {part_idx}, start_idx: {start_idx}, end_idx: {end_idx}")
            rbf = RBF(
                original_control_points=hierarchy.nodes_by_name[name]
                .points.detach()
                .cpu()
                .numpy(),
                deformed_control_points=hierarchy.get_world_points(name)
                .detach()
                .cpu()
                .numpy(),
                radius=rbf_radius,
                func=rbf_kernel,
                extra_parameter=ext_params,
            )
            # # Deform the surface points
            deformed_full_point_cloud = rbf(
                np.array(
                    deforming_vertices[deforming_part_indices[name]], dtype=np.float32
                )
            )
            total_deformed_pcs[deforming_part_indices[name]] = deformed_full_point_cloud
            # cur_deformed_pcs.append(
            #     torch.from_numpy(deformed_full_point_cloud)
            #     .float()
            #     .to(device)
            # )
        # total_deformed_pcs.append(torch.cat(cur_deformed_pcs, dim=0))

    # total_deformed_pcs = torch.stack(total_deformed_pcs, dim=0)
    # np.save(os.path.join(log_dir, f"deformed_pts.npy"), total_deformed_pcs.cpu().numpy())
    # smoothed_total_deformed_pcs = smooth_point_cloud_torch(
    #     total_deformed_pcs, smooth_window_size
    # )
    # np.save(
    #     os.path.join(log_dir, f"deformed_pts_smooth.npy"),
    #     smoothed_total_deformed_pcs.cpu().numpy(),
    # )
    # save a new mesh with the deformed vertices
    deformed_mesh = trimesh.Trimesh(
        vertices=total_deformed_pcs, faces=deforming_faces, process=False
    )
    deformed_mesh.export(os.path.join(log_dir, "deformed_mesh.obj"))
    print("Deformation finished, saved to disk")
