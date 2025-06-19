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


def point_to_plane_icp(source, target, init_transformation=np.eye(4), threshold=0.02, max_iterations=50, regularization_weight=0.1):
    """
    Perform point-to-plane ICP with regularization to align source to target point cloud.
    
    :param source: Source point cloud (open3d.geometry.PointCloud)
    :param target: Target point cloud (open3d.geometry.PointCloud)
    :param init_transformation: Initial transformation matrix (4x4 numpy array)
    :param threshold: Distance threshold for ICP
    :param max_iterations: Maximum number of ICP iterations
    :param regularization_weight: Weight for the regularization term
    :return: Transformation matrix
    """
    # Convert point clouds to numpy arrays
    source_points = source.cpu().numpy()
    target_points = target.cpu().numpy()
    
    pcd = o3d.geometry.PointCloud()
    # Assign the numpy array to the point cloud's points
    pcd.points = o3d.utility.Vector3dVector(target_points)
    # Compute target normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    target_normals = np.asarray(pcd.normals)
    
    # Initialize transformation
    transformation = init_transformation
    
    for iteration in range(max_iterations):
        # Apply current transformation to source points
        transformed_source_points = (transformation[:3, :3] @ source_points.T).T + transformation[:3, 3]
        
        # Find closest points in target for each point in transformed source
        closest_indices = find_closest_points(transformed_source_points, target_points)
        
        # Compute point-to-plane distances and Jacobians
        A = []
        b = []
        for i, closest_index in enumerate(closest_indices):
            p = transformed_source_points[i]
            q = target_points[closest_index]
            n = target_normals[closest_index]
            
            # Point-to-plane distance
            d = np.dot(p - q, n)
            
            # Jacobian of the point-to-plane distance
            J = np.zeros((1, 6))
            J[0, :3] = n
            J[0, 3:] = np.cross(p, n)
            
            A.append(J)
            b.append(d)
        
        A = np.vstack(A)
        b = np.vstack(b)
        
        # Add regularization term to penalize large rotations or deformations
        reg_term = regularization_weight * np.eye(6)
        A = np.vstack((A, reg_term))
        b = np.vstack((b, np.zeros((6, 1))))
        
        # Solve for the update
        delta = np.linalg.lstsq(A, b, rcond=None)[0].flatten()
        
        # Update transformation
        delta_transformation = np.eye(4)
        delta_transformation[:3, :3] = o3d.geometry.get_rotation_matrix_from_axis_angle(delta[3:])
        delta_transformation[:3, 3] = delta[:3]
        
        transformation = delta_transformation @ transformation
        
        # Check convergence (optional)
        if np.linalg.norm(delta) < threshold:
            break
    
    return transformation


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


def positional_encoding(x):
    pe = []
    for fn in [torch.sin, torch.cos]:
        pe.append(fn(x))
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
    def __init__(self, base_channels=64, max_freq=10, temperature=5.0, out_dim=6):
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
        pe = positional_encoding(x)  # (N, enc_dim)
        out = self.net(pe)  # (N, out_dim)
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


def modified_icp_nn_new(
    src,
    tgt,
    models,
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
    loss_modes=[],
    hierarchical_transformation=False,
    joint_pos=[],
    joint_k=1,
    min_iter=100,
    save_logs=True,
    time_step=0,
    save_img_dir=None,
):
    if save_logs:
        save_path = os.path.join(log_dir, f"rigid_cor_{correspondence}_iter{iterations}_jointK{joint_k}/")    
        if len(loss_modes):
            save_path = save_path[:-1] + "_"
            for lm in loss_modes:
                if lm == "pt2plane":
                    save_path += f"p_"
                elif lm == "chamfer":
                    save_path += f"c_"
        else:
            if loss_mode == "pt2plane":
                save_path = save_path[:-1] + "_plane/"
        if loss_type != "l2":
            if isinstance(robust_c, list):
                save_path = save_path[:-1] + f"_{loss_type}_c{robust_c[0]}_{robust_c[1]}/"
            elif robust_c != 0.2:
                save_path = save_path[:-1] + f"_{loss_type}_c{robust_c}/"
            else:
                save_path = save_path[:-1] + f"_{loss_type}/"
        if src_iter >= 0:
            save_path = save_path[:-1] + f"_srciter{src_iter}/"
        if use_arap:
            if joint_arap:
                save_path = save_path[:-1] + f"_joint_arap{arap_w}_k{smooth_knn}/"
            else:
                save_path = save_path[:-1] + f"_arap{arap_w}_k{smooth_knn}/"
        # if avg_loss:
        #     save_path = save_path[:-1] + f"_avgL/"
        # if add_lda:
        #     save_path = save_path[:-1] + f"_lda{lda_weight}/"
        # if hierarchical_transformation:
        #     save_path = save_path[:-1] + f"_hier/"
        os.makedirs(save_path, exist_ok=True)

        plot_pointcloud(
            tgt, save_path, title=f"Target Point Cloud", scale=VISIUAL_SCALE
        )
        match_frames = []

    if use_arap and not joint_arap:
        arap_losses = [ARAPLoss(src[i], smooth_knn=smooth_knn) for i in range(len(src))]
    elif use_arap and joint_arap:
        arap_loss = ARAPLoss(torch.cat(src, dim=0), smooth_knn=smooth_knn)

    # if early_stop:
    #     init_volumes = [estimate_volume(src[i].cpu().numpy()) for i in range(len(src))] if len(base_volumes) == 0 else base_volumes

    total_params = []
    for model in models:
        model.to(device)
        model.train()
        total_params += list(model.parameters())
    optimizer = torch.optim.Adam(total_params, lr=lrt)
    stop = False
    # new_tgt = None
    new_tgts = [None for _ in range(len(src))]

    segment_length = iterations // num_segments
    cur_seg_idx = 0
    cur_c_seg_idx = 0
    cur_c = robust_c[cur_c_seg_idx] if isinstance(robust_c, list) else robust_c

    loss_history = []
    all_tgt_indices = None

    tgt_np = tgt.detach().cpu().numpy()

    total_num_pts = sum([len(c_src) for c_src in src])

    if correspondence == "linear_sum":
        tgt_normals = estimate_normals_pytorch3d(tgt.cpu().numpy())
        # tgt_normals = []
        # for i in range(len(tgt)):
        #     tgt_normals.append(estimate_normals_pytorch3d(tgt[i].cpu().numpy()))

    for it in tqdm.tqdm(range(iterations), desc="ICP Iterations"):
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
        tgt_normal_shuffled = []
        total_arap_loss = 0.
        loss = 0.
        # losses = []
        root_R = None
        root_t = None
        # all_tgt_indices = []
        for i in range(len(src)):

            model = models[i]
            mask_ratio = min((segment_index + 1) * 1.0 / num_segments, part_max_freqs[i] * 1.0 / model.max_freq)

            if i == 0:
                inp = torch.cat([src[i]] + joint_pos[1:], dim=0)
            else:
                inp = src[i]
            params = model(inp, mask_ratio=mask_ratio)  # (N, 6)
            axis_angle = params[:, :3]
            translation = params[:, 3:]

            # Convert axis-angle to rotation matrix
            R = axis_angle_to_rotation_matrix(axis_angle)

            if i > 0:
                src_centered = src[i] - joint_pos[i]
                rotated = torch.bmm(R, src_centered.unsqueeze(-1)).squeeze(-1)
                src_transformed.append(
                    torch.bmm(
                        root_R[(i-1):i].expand(len(src[i]), 3, 3),
                        (rotated + joint_pos[i].expand(len(src[i]), 3)).unsqueeze(
                            -1
                        ),
                    ).squeeze(-1) + root_t[(i-1):i].expand(len(src[i]), 3)
                )
            else:
                root_R = R[len(src[i]):]
                root_t = translation[len(src[i]):]

                # rotated = torch.bmm(R, inp.unsqueeze(-1)).squeeze(-1)
                # cur_transformed = rotated + translation
                # src_transformed.append(cur_transformed[: len(src[i])])
                rotated = torch.bmm(R[: len(src[i])], src[i].unsqueeze(-1)).squeeze(-1)
                cur_transformed = rotated + translation[: len(src[i])]
                src_transformed.append(cur_transformed)

            if use_arap and not joint_arap:
                total_arap_loss += arap_losses[i](src_transformed[i], add_lda=add_lda, lda_weight=lda_weight)

            # Find closest points (placeholder for nearest neighbor search)
            # src_trans_np = src_transformed[i].detach().cpu().numpy()
            # if isinstance(tgt, list):
            #     tgt_np = tgt[i].detach().cpu().numpy()
            # else:
            #     tgt_np = tgt.detach().cpu().numpy()

            # if correspondence == "linear_sum":
            #     new_tgt_kp, new_tgt_normal, tgt_indices = linear_sum_correspondence(
            #         src_trans_np, tgt_np, tgt_normals[i], use_cos=False, alpha=0.75
            #     )
            #     tgt_nn.append(torch.from_numpy(new_tgt_kp).float().to(device))
            #     tgt_normal_shuffled.append(torch.from_numpy(new_tgt_normal).float().to(device))
            #     all_tgt_indices.append(tgt_indices)
            # else:
            #     tgt_nn.append(tgt[i])

            # cur_ratio = estimate_volume(src_trans_np) / init_volumes[i] if len(init_volumes) else 1.0
            # if len(init_volumes) and (cur_ratio < stop_volume_ratio[i] - stop_volume_error_rates[i]) or (cur_ratio > stop_volume_ratio[i] + stop_volume_error_rates[i]):
            #     if new_tgts[i] is None:
            #         print(f"!! Changing target for part {i}")
            #         new_tgts[i] = src_transformed[i].clone().detach()
            #     loss += one_sided_chamfer_loss(
            #         src_transformed[i],
            #         new_tgts[i],
            #         loss_type=loss_type,
            #         c=cur_c,
            #         reduce="sum",
            #     )
            # else:
            #     loss += one_sided_chamfer_loss(src_transformed[i], tgt_nn[i], loss_type=loss_type, c=cur_c, reduce="sum")

        src_transformed = torch.cat(src_transformed, dim=0)

        src_trans_np = src_transformed.detach().cpu().numpy()

        if correspondence == "linear_sum":
            new_tgt_kp, new_tgt_normal, tgt_indices = linear_sum_correspondence(
                src_trans_np, tgt_np, tgt_normals, use_cos=False, alpha=0.75
            )
            tgt_nn = torch.from_numpy(new_tgt_kp).float().to(device)
            # tgt_normal_shuffled.append(torch.from_numpy(new_tgt_normal).float().to(device))
            all_tgt_indices = tgt_indices
        elif correspondence == "nn":
            tree = KDTree(tgt_np)
            dists, cur_nearest_indices = tree.query(src_trans_np, k=1)
            cur_nearest_indices = cur_nearest_indices.flatten()
            tgt_nn = tgt[cur_nearest_indices]

        loss += one_sided_chamfer_loss(src_transformed, tgt_nn, loss_type=loss_type, c=cur_c, reduce="sum")

        # tgt_nn = torch.cat(tgt_nn, dim=0)
        # if len(tgt_normal_shuffled) > 0:
        #     tgt_normal_shuffled = torch.cat(tgt_normal_shuffled, dim=0)

        # if use_arap and joint_arap:
        #     total_arap_loss = arap_loss(src_transformed, add_lda=add_lda, lda_weight=lda_weight)

        if use_arap:
            loss = loss * 1. / total_num_pts + total_arap_loss * arap_w
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
                        for m_idx, model in enumerate(models):
                            torch.save(model.state_dict(), os.path.join(save_path, f"icp_model_{m_idx}_{it}.pth"))

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
        for m_idx, model in enumerate(models):
            torch.save(model.state_dict(), os.path.join(save_path, f"icp_model_{m_idx}_{it}.pth"))

    save_dir = save_img_dir if time_step % 10 == 0 else None
    cur_frame = plot_pointcloud(
        src_transformed,
        save_dir,
        title=f"Time Step {time_step}",
        extra_points=tgt_nn,
        plot_match=True,
        scale=VISIUAL_SCALE,
    )

    return models, src_transformed.detach(), cur_frame, it, all_tgt_indices


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

# root_part = np.load("/NAS/spa176/papr-retarget/bird_Body_vertices.npy")
# child_part_0 = np.load("/NAS/spa176/papr-retarget/bird_LeftWing_vertices.npy")
# child_part_1 = np.load("/NAS/spa176/papr-retarget/bird_RightWing_vertices.npy")

# root_part_indieces = np.load("/NAS/spa176/papr-retarget/bird_Body_vertex_indieces.npy")
# child_part_0_indieces = np.load(
#     "/NAS/spa176/papr-retarget/bird_LeftWing_vertex_indieces.npy"
# )
# child_part_1_indieces = np.load(
#     "/NAS/spa176/papr-retarget/bird_RightWing_vertex_indieces.npy"
# )
# # concatenate the parts to make the full point cloud
# tgt_pc_parts = [root_part, child_part_0, child_part_1]
# tgt_pc = np.concatenate(tgt_pc_parts, axis=0)

# # print the number of points in each part
# print("Number of points in each part:")
# print("Body: ", len(root_part))
# print("Left Wing: ", len(child_part_0))
# print("Right Wing: ", len(child_part_1))

# fps_k = 1000
if USE_KEYPOINTS:
    num_keypoints_left = 96
    num_keypoints_right = 96
    num_keypoints_body = 256
# else:
#     num_keypoints_left = len(child_part_0)
#     num_keypoints_right = len(child_part_1)
#     num_keypoints_body = len(root_part)
# num_keypoints_body = 128
# num_keypoints_body = 5600
all_num_keypoints = [num_keypoints_body, num_keypoints_left, num_keypoints_right]


log_dir = "fit_pointcloud_logs"
exp_dir = f"blender_mp_wingL{num_keypoints_left}_wingR{num_keypoints_right}_body{num_keypoints_body}"

# exp_id = 7
exp_id = 4

exp_sub_dir = f"exp_{exp_id}"
log_dir = os.path.join(log_dir, exp_dir, exp_sub_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
# copy the current file to log_dir
shutil.copy(__file__, log_dir)

save_img_dir = os.path.join(log_dir, "images")
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir, exist_ok=True)

src_iter = 30000
# src_iter = 0
# src_iter = 24000
src_pc_path = f"/NAS/spa176/papr-retarget/point_clouds/butterfly/points_{src_iter}.npy"
src_pc = np.load(src_pc_path)

# scale = 10.0
# src_pc = src_pc / scale
tgt_pc = np.load(
    "/NAS/spa176/papr-retarget/bird_vertices.npy"
)

source_ranges = np.max(src_pc, axis=0) - np.min(src_pc, axis=0)
target_ranges = np.max(tgt_pc, axis=0) - np.min(tgt_pc, axis=0)

# Calculate dimension-wise scale ratios
scale_ratios = target_ranges / source_ranges

# Use the maximum scale ratio for consistent scaling
scale = np.min(scale_ratios)
# scale
print("Scale: ", scale)

src_pc = src_pc * scale

VISIUAL_SCALE = 5.0

# tgt_pc = tgt_pc / scale

src_pc = torch.tensor(src_pc).float().to(device)
tgt_pc = torch.tensor(tgt_pc).float().to(device)


# tgt_pc[..., 0] += 0.1
# tgt_pc[..., 1] -= 0.2
# tgt_pc[..., 2] += 0.3
# # shuffle the point cloud
# tgt_pc = tgt_pc[torch.randperm(tgt_pc.shape[0])]
print("src_pc: ", src_pc.shape, src_pc.min(), src_pc.max())
print("tgt_pc: ", tgt_pc.shape, tgt_pc.min(), tgt_pc.max())

# Plot the point clouds
plot_pointcloud(src_pc, log_dir, title="Source Point Cloud", scale=VISIUAL_SCALE)
plot_pointcloud(tgt_pc, log_dir, title="Target Point Cloud", scale=VISIUAL_SCALE)

if USE_ICP:
    converged, rmse, Xt, RTs, t_history = icp(tgt_pc.unsqueeze(0), src_pc.unsqueeze(0), max_iterations=300)
    print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
    plot_pointcloud(Xt.squeeze(), log_dir, title="ICP Point Cloud", scale=VISIUAL_SCALE)
    tgt_pc = Xt.squeeze(0)

# src_kps, src_kp_indices = sfp(src_pc.unsqueeze(0), K=sum(all_num_keypoints))
# tgt_kps, tgt_kp_indices = sfp(tgt_pc.unsqueeze(0), K=sum(all_num_keypoints))
# save the keypoints and indices
# np.save(
#     f"but_kps_{sum(all_num_keypoints)}.npy",
#     src_kps.cpu().numpy(),
# )
# # save the keypoint as a xyz file
# np.savetxt(
#     f"but_kps_{sum(all_num_keypoints)}.xyz",
#     src_kps.squeeze(0).cpu().numpy(),
#     fmt="%f %f %f",
# )
# np.save(
#     f"but_kp_indices_{sum(all_num_keypoints)}.npy",
#     src_kp_indices.squeeze(0).cpu().numpy(),
# )
# # np.save(f"bird_kps_{sum(all_num_keypoints)}.npy", tgt_kps.cpu().numpy())
# np.savetxt(
#     f"bird_kps_{sum(all_num_keypoints)}.xyz",
#     tgt_kps.squeeze(0).cpu().numpy(),
#     fmt="%f %f %f",
# )
# np.save(
#     f"bird_kp_indices_{sum(all_num_keypoints)}.npy",
#     tgt_kp_indices.squeeze(0).cpu().numpy(),
# )

# exit(0)

root_part_indices = np.load("/NAS/spa176/papr-retarget/bird_Body_vertex_indices.npy")
child_part_0_indices = np.load(
    "/NAS/spa176/papr-retarget/bird_LeftWing_vertex_indices.npy"
)
child_part_1_indices = np.load(
    "/NAS/spa176/papr-retarget/bird_RightWing_vertex_indices.npy"
)
# concatenate the parts to make the full point cloud
tgt_pc_parts = [
    tgt_pc[root_part_indices],
    tgt_pc[child_part_0_indices],
    tgt_pc[child_part_1_indices],
]
# tgt_pc = np.concatenate(tgt_pc_parts, axis=0)

# print the number of points in each part
print("Number of points in each part:")
print("Body: ", len(root_part_indices))
print("Left Wing: ", len(child_part_0_indices))
print("Right Wing: ", len(child_part_1_indices))


# load wing indices
but_wing_indices = np.load("but_wing_indices.npy")
but_body_indices = np.setdiff1d(np.arange(len(src_pc)), but_wing_indices)

but_wing_indices_right = np.load("but_wing_indices_right.npy")
but_wing_indices_left = np.setdiff1d(
    np.arange(len(but_wing_indices)), but_wing_indices_right
)

# bird_wing_indices = np.load("hummingbird_wing_indices.npy")
# bird_body_indices = np.setdiff1d(np.arange(len(tgt_pc)), bird_wing_indices)

# bird_wing_indices_right = np.load("hummingbird_wing_indices_right.npy")
# bird_wing_indices_left = np.setdiff1d(
#     np.arange(len(bird_wing_indices)), bird_wing_indices_right
# )


# init_kps = []
# kp_indices = []
# tgt_kps = []
# tgt_pcs = []
# boundary_indices = []
# ori_tgt_kps = []

# src_part_indices = [but_wing_indices[but_wing_indices_left], but_wing_indices[but_wing_indices_right], but_body_indices]
# tgt_part_indices = [
#     bird_wing_indices[bird_wing_indices_left],
#     bird_wing_indices[bird_wing_indices_right],
#     bird_body_indices,
# ]
src_part_indices = [but_body_indices, but_wing_indices[but_wing_indices_left], but_wing_indices[but_wing_indices_right]]
# tgt_part_indices = [
#     bird_body_indices,
#     bird_wing_indices[bird_wing_indices_left],
#     bird_wing_indices[bird_wing_indices_right],
# ]
use_hierarchical_transformation = True
total_part_num = 3
joint_k = 1

# icp_model = PointwiseTransformNet(base_channels=64, max_freq=0).to(device)
temperature = 50.0
# temperature = 25.0
max_freq = 1
icp_models = [
    PointwiseTransformNet(
        base_channels=32, max_freq=max_freq, temperature=temperature
    ).to(device)
    for _ in range(total_part_num)
]

if USE_ICP:

    # src_kps = []
    tgt_kps = []
    tgt_pcs = []
    # src_kp_indices = []
    # src_volumes = []

    # FPS
    src_kps, src_kp_indices = sfp(
        src_pc.unsqueeze(0), K=sum(all_num_keypoints)
    )

    src_kp_indices = src_kp_indices.squeeze(0)

    for part_idx in range(total_part_num):

        # FPS
        cur_src_kp, src_kp_idx = sfp(
            src_pc[src_part_indices[part_idx]].unsqueeze(0), K=all_num_keypoints[part_idx]
        )
        # cur_src_kp = cur_src_kp.squeeze(0)
        if USE_KEYPOINTS:
            cur_tgt_kp, tgt_kp_idx = sfp(
                tgt_pc_parts[part_idx].unsqueeze(0),
                K=all_num_keypoints[part_idx],
            )
            cur_tgt_kp = cur_tgt_kp.squeeze(0)
        else:
            cur_tgt_kp = torch.from_numpy(tgt_pc_parts[part_idx]).float().to(device)
        # ori_tgt_kps.append(torch.from_numpy(final_ori_tgt_kp).float())

        # src_kps.append(cur_src_kp.squeeze(0))
        # src_kp_indices.append(src_kp_idx)
        # kp_indices.append(src_kp_idx)
        # tgt_kps.append(torch.from_numpy(tgt_pc_parts[part_idx] / scale).float().to(device))
        tgt_kps.append(cur_tgt_kp)
        # tgt_pcs.append(cur_tgt_pc.clone().cpu())
        # torch.save(
        #     tgt_pc[tgt_part_indices[part_idx]],
        #     os.path.join(log_dir, f"part{part_idx}_tgt_pc_ori.pth")
        # )
        tgt_pcs.append(tgt_pc_parts[part_idx])
        # src_volumes.append(estimate_volume(src_kps[part_idx].cpu().numpy()))

    if use_hierarchical_transformation:
        # find the joint position

        joint_pos = [None]
        for part_idx in range(1, 3, 1):
            cur_joint_pos = find_wing_anchor_point_torch(
                tgt_kps[part_idx], tgt_kps[0], k=joint_k
            )
            joint_pos.append(cur_joint_pos.unsqueeze(0))


start = 0
end = 30001
interval = 200
# start = 30000
# end = -1
# interval = -200
# scale = 10.0
smooth_window_size = 35

src_pc_dir = "/NAS/spa176/papr-retarget/point_clouds/butterfly/"

src_keypoints = []
for idx in tqdm.tqdm(range(start, end, interval)):
    src_pc_path = os.path.join(src_pc_dir, f"points_{idx}.npy")
    cur_src_pc = np.load(src_pc_path)
    # cur_src_pc = cur_src_pc / scale
    cur_src_pc = cur_src_pc * scale
    src_keypoints.append(
        torch.tensor(cur_src_pc[src_kp_indices.cpu().numpy()]).float().to(device)
    )
    # point_clouds.append(cur_src_pc)
    # for part_idx in range(total_part_num):
    #     cur_src_kp = cur_src_pc[src_part_indices[part_idx]][src_kp_indices[part_idx].cpu().numpy()]
    #     cur_src_kp = torch.tensor(cur_src_kp).float().to(device)
    #     src_keypoints[part_idx].append(cur_src_kp)
    # src_volume_ratios[part_idx].append(
    #     estimate_volume(cur_src_kp.cpu().numpy()) * 1.
    #     / src_volumes[part_idx]
    # )
# for part_idx in range(total_part_num):
#     src_keypoints[part_idx] = smooth_point_cloud_torch(torch.cat(src_keypoints[part_idx], dim=0), smooth_window_size)
src_keypoints = smooth_point_cloud_torch(
    torch.stack(src_keypoints, dim=0), smooth_window_size
)

# loop through src_keypoints and calculate the volume ratio at each time step
# src_volume_ratios = [[] for _ in range(total_part_num)]
# for part_idx in range(total_part_num):
#     for time_step in range(len(src_keypoints[part_idx])):
#         cur_src_kp = src_keypoints[part_idx][time_step]
#         cur_src_volume = estimate_volume(cur_src_kp.cpu().numpy())
#         src_volume_ratios[part_idx].append(cur_src_volume * 1. / src_volumes[part_idx])


total_time_steps = len(src_keypoints)
total_time_steps = 1

transferred_tgt_kps = []
match_frames = []
# TRAIN_ICP_MODEL = False
TRAIN_ICP_MODEL = True
base_volumes = [] 
if TRAIN_ICP_MODEL:
    # loop through each time step
    for time_step in tqdm.tqdm(range(total_time_steps)):

        # cur_iterations = 800 if time_step == 0 else 500
        cur_min_iter = 100 if time_step == 0 else 10
        save_log = True if time_step == 0 else False
        cur_src_kps = src_keypoints[time_step]
        correspondence = "linear_sum"
        # correspondence = "nn"

        if time_step > 0:
            cur_src_kps = cur_src_kp[initial_tgt_indices]
            # cur_src_kps = [cur_src_kp[initial_tgt_indices[part_idx]] for part_idx, cur_src_kp in enumerate(cur_src_kps)]
            correspondence = None if USE_LINEAR_SUM else "nn"

        # print(f"Step {time_step}, ratios: {[src_volume_ratio[time_step] for src_volume_ratio in src_volume_ratios]}")

        icp_models, src_transformed, match_frame, train_iters, all_tgt_indices = (
            modified_icp_nn_new(
                # [tgt_pc[bird_body_indices], tgt_pc[bird_wing_indices]],
                # [src_pc[but_body_indices], src_pc[but_wing_indices]],
                tgt_kps,
                cur_src_kps,
                icp_models,
                log_dir,
                # iterations=600,
                iterations=500,
                # num_segments=max_freq,
                # num_segments=1,
                loss_mode="chamfer",
                part_max_freqs=[max_freq, max_freq, max_freq],
                plot_match=True,
                # loss_type="hubert",
                # loss_type="geman",
                # loss_type="p2p"
                correspondence=correspondence,
                # lrt=2.5e-3,
                src_iter=src_iter,
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
                stop_volume_ratio=[],
                init_volumes=base_volumes,
                stop_volume_error_rates=[0.05, 10.0, 10.0],
                # stop_volume_ratio=0.97,
                # loss_modes=["chamfer", "pt2plane", "pt2plane"],
                # hierarchical_transformation=use_hierarchical_transformation,
                joint_pos=joint_pos,
                joint_k=joint_k,
                save_logs=save_log,
                min_iter=cur_min_iter,
                save_img_dir=save_img_dir,
                time_step=time_step,
            )
        )

        if time_step == 0:
            initial_tgt_indices = all_tgt_indices
            # update base volumes by calculating the src_transformed for now, slice the src_transformed by the number of keypoints in each part
            for part_idx in range(total_part_num):
                start_idx = (
                    0
                    if part_idx == 0
                    else start_idx + all_num_keypoints[part_idx - 1]
                )
                end_idx = start_idx + all_num_keypoints[part_idx]
                base_volumes.append(estimate_volume(src_transformed[start_idx:end_idx].cpu().numpy()))

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

    smoothed_total_deformed_pcs = smooth_point_cloud_torch(transferred_tgt_kps, smooth_window_size)
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
    smoothed_total_deformed_pcs = smoothed_total_deformed_pcs.cpu().numpy()
    np.save(
        os.path.join(log_dir, f"deformed_{save_pt_name}_smooth.npy"),
        smoothed_total_deformed_pcs,
    )

else:
    transferred_tgt_kps = torch.load(os.path.join(log_dir, f"transferred_tgt_kps.pth"))
    print("Loaded", transferred_tgt_kps.shape)

if USE_KEYPOINTS:
    rbf_kernel = "polyharmonic_spline"
    ext_params = {"k": 3} if rbf_kernel == "polyharmonic_spline" else {}
    rbf_radius = 10

    print("=" * 20)
    print("Motion transfer finished, start to recover the full point cloud")

    total_deformed_pcs = []
    for cur_step in tqdm.tqdm(range(total_time_steps)):
        cur_deformed_pcs = []
        for part_idx in range(total_part_num):
            start_idx = (
                0
                if part_idx == 0
                else start_idx + all_num_keypoints[part_idx - 1]
            )
            end_idx = start_idx + all_num_keypoints[part_idx]
            if cur_step == 0:
                print(f"Slicing part_idx: {part_idx}, start_idx: {start_idx}, end_idx: {end_idx}")
            rbf = RBF(
                original_control_points=tgt_kps[part_idx].cpu().numpy(),
                deformed_control_points=transferred_tgt_kps[
                    cur_step, start_idx:end_idx, :
                ].cpu().numpy(),
                radius=rbf_radius,
                func=rbf_kernel,
                extra_parameter=ext_params,
            )
            # # Deform the surface points
            deformed_full_point_cloud = rbf(tgt_pcs[part_idx])
            cur_deformed_pcs.append(
                torch.from_numpy(deformed_full_point_cloud)
                .float()
                .to(device)
            )
        total_deformed_pcs.append(torch.cat(cur_deformed_pcs, dim=0))

    total_deformed_pcs = torch.stack(total_deformed_pcs, dim=0)
    np.save(os.path.join(log_dir, f"deformed_pts.npy"), total_deformed_pcs.cpu().numpy())
    smoothed_total_deformed_pcs = smooth_point_cloud_torch(
        total_deformed_pcs, smooth_window_size
    )
    np.save(
        os.path.join(log_dir, f"deformed_pts_smooth.npy"),
        smoothed_total_deformed_pcs.cpu().numpy(),
    )
    print("Deformation finished, saved to disk")

# target_exp_index = 13
# save_model_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-{target_exp_index}-exp{exp_id}/"
# os.makedirs(save_model_path, exist_ok=True)

# state_dict = torch.load("/NAS/spa176/papr-retarget/experiments/hummingbird-start-1/model_no_rot_fix_name.pth")
# step = list(state_dict.keys())[0]
# state_dict = state_dict[step]

# new_pc_feats = [state_dict["pc_feats"][tgt_part_indices[part_idx]] for part_idx in range(total_part_num)]
# new_pc_feats = torch.cat(new_pc_feats, dim=0)
# state_dict["pc_feats"] = new_pc_feats

# new_points_influ_scores = [state_dict["points_influ_scores"][tgt_part_indices[part_idx]] for part_idx in range(total_part_num)]
# new_points_influ_scores = torch.cat(new_points_influ_scores, dim=0)
# state_dict["points_influ_scores"] = new_points_influ_scores

# # align the deformed_pc with original pc
# converged, rmse, Xt, RTs, t_history = icp(
#     total_deformed_pcs[0].unsqueeze(0), state_dict["points"].unsqueeze(0), max_iterations=300
# )
# print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
# # deformed_pc = Xt.squeeze(0)

# R = RTs.R
# T = RTs.T

# total_deformed_pcs = torch.bmm(total_deformed_pcs, R.expand(total_deformed_pcs.shape[0], 3, 3)) + T[:, None, :].expand(total_deformed_pcs.shape[0], -1, 3)

# save_pc_name = f"total_deformed_pc_rbf.pth"
# # save the transformed deformed_pc
# torch.save(total_deformed_pcs, save_model_path + save_pc_name)


# smoothed_total_deformed_pcs = smooth_point_cloud_torch(total_deformed_pcs, smooth_window_size)
# save_pc_name = f"total_deformed_pc_rbf_smooth.pth"
# torch.save(smoothed_total_deformed_pcs, save_model_path + save_pc_name)


# LDA(total_deformed_pcs, smooth_knn=100)
# save_pc_name = f"total_deformed_pc_rbf_lda.pth"
# torch.save(total_deformed_pcs, save_model_path + save_pc_name)

# LDA(smoothed_total_deformed_pcs, smooth_knn=100)
# save_pc_name = f"total_deformed_pc_rbf_lda_smooth.pth"
# torch.save(smoothed_total_deformed_pcs, save_model_path + save_pc_name)

# state_dict["points"] = total_deformed_pcs[0]
# save_sd = {step: state_dict}
# torch.save(save_sd, save_model_path + "model.pth")


# print("Model saved!")
