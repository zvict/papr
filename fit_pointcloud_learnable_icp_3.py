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
import tqdm
import imageio
import io
from PIL import Image
import random
from models.mlp import MLP
from models.utils import PoseEnc, activation_func
from sklearn.neighbors import NearestNeighbors
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

def plot_pointcloud(points, save_dir, title="", extra_points=None, plot_match=True):
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
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
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
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_zlim(-0.6, 0.6)
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


class AffineTransformationNet(nn.Module):
    def __init__(self, input_dim, L=0, non_pe_dim=0):
        super(AffineTransformationNet, self).__init__()
        self.pose_enc = PoseEnc()
        self.L = L
        in_dim = input_dim + input_dim * 2 * L + non_pe_dim
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)  # 1 for scale, 4 for quaternion

    def forward(self, x, non_pe=None):
        x = self.pose_enc(x, self.L)
        if non_pe is not None:
            x = torch.cat([x, non_pe], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if x.dim() == 2:
            # Shape is [batch_size, n]
            return x[:, 0], x[:, 1:]
        elif x.dim() == 3:
            # Shape is [num_steps, batch_size, n]
            return x[:, :, 0], x[:, :, 1:]


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a rotation matrix.
    """
    q = F.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    B = q.size(0)
    R = torch.zeros((B, 3, 3), device=q.device)
    R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x**2 + y**2)
    return R


def apply_transformation(displacement, scale, rotation_matrix, no_scale=False):
    """
    Apply scaling and rotation to the displacement vector.
    """
    if no_scale:
        scaled_displacement = displacement
    else:
        scaled_displacement = F.relu(scale.unsqueeze(-1) + 1.) * displacement
    # print("!!scaled_displacement: ", scaled_displacement.shape)
    # print("!!displacement: ", displacement.shape)
    # print("!!scale: ", scale.shape)
    # print(a)
    # print("!!rotation_matrix: ", rotation_matrix.shape)
    if scaled_displacement.dim() == 2:
        rotated_displacement = torch.bmm(
            rotation_matrix, scaled_displacement.unsqueeze(-1)
        ).squeeze(-1)
    else:
        rotated_displacement = torch.matmul(
            rotation_matrix, scaled_displacement.unsqueeze(-1)
        ).squeeze(-1)
    return rotated_displacement


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
        # eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
        # print("eigenvalues: ", eigenvalues.shape)
        # print("eigenvectors: ", eigenvectors.shape)
        # The eigenvector corresponding to the smallest eigenvalue is the normal
        # normal = eigenvectors[:, eigenvalues[:, 0].argmin()]
        normal = eigenvectors[:, eigenvalues.real.argmin()]

        # Store the normal
        normals[pt_idx, :] = normal

    # Normalize the normals
    normals = normals / normals.norm(dim=-1, keepdim=True)

    return normals


def compute_rotation_matrix(normal1, normal2):
    """
    Compute the rotation matrix that aligns normal1 to normal2.
    
    Args:
    - normal1 (torch.Tensor): The first normal vector (3,).
    - normal2 (torch.Tensor): The second normal vector (3,).
    
    Returns:
    - rotation_matrix (torch.Tensor): The rotation matrix (3, 3).
    """
    # Normalize the normals
    # normal1 = normal1 / normal1.norm(dim=-1, keepdim=True)
    # normal2 = normal2 / normal2.norm(dim=-1, keepdim=True)

    # Compute the axis of rotation (cross product)
    axis = torch.cross(normal1, normal2, dim=-1)
    axis = axis / axis.norm(dim=-1, keepdim=True)

    # Compute the angle of rotation (dot product)
    angle = torch.acos(torch.clamp(torch.sum(normal1 * normal2, dim=-1), -1.0, 1.0))

    # Ensure axis has the correct shape
    if axis.dim() == 1:
        axis = axis.unsqueeze(0)

    # Compute the components of the rotation matrix
    K = torch.zeros((axis.size(0), 3, 3), device=axis.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    I = torch.eye(3, device=normal1.device).unsqueeze(0).repeat(axis.size(0), 1, 1)

    angle = angle.unsqueeze(-1).unsqueeze(-1)
    sin_angle = torch.sin(angle)
    cos_angle = torch.cos(angle)

    rotation_matrix = I + sin_angle * K + (1 - cos_angle) * K.bmm(K)

    return rotation_matrix


def compute_rotation_matrices_for_batch(point_clouds, num_nn, flip_normal=False):
    """
    Compute the rotation matrices for a batch of point clouds.
    
    Args:
    - point_clouds (torch.Tensor): The input point clouds (T, N, 3).
    - num_nn (int): The number of nearest neighbors to use for normal estimation.
    
    Returns:
    - rotation_matrices (torch.Tensor): The rotation matrices (T, N, 3, 3).
    """
    T, N, _ = point_clouds.shape

    # Calculate the surface normals for each time step
    normals = torch.zeros((T, N, 3), device=point_clouds.device)
    for t in range(T):
        # normals[t] = estimate_surface_normals(point_clouds[t], num_nn)
        normals[t] = estimate_normals(point_clouds[t], num_nn)

    # Compute the rotation matrices
    rotation_matrices = torch.zeros((T, N, 3, 3), device=point_clouds.device)
    for t in range(T):
        if t == 0:
            # make it identity
            rotation_matrices[t] = torch.eye(3, device=point_clouds.device).unsqueeze(0).repeat(N, 1, 1)
        for n in range(N):
            normal1 = normals[0, n]
            normal2 = normals[t, n]
            # TODO: remove this when the normals are correct
            # calculate the dot product of each normals, and flip the normal index in normal 2 if the dot product is negative
            if flip_normal and (torch.dot(normal1, normal2) < 0):
                normal2 = -normal2
                normals[t, n] = normal2
            rotation_matrices[t, n] = compute_rotation_matrix(normal1, normal2)

    return rotation_matrices, normals


def parametertize_pc(points, keypoints, num_knn, step=50000):
    tree = KDTree(keypoints.cpu().numpy())
    nn_dists, nn_inds = tree.query(points.cpu().numpy(), k=num_knn)
    knn_sp = keypoints[nn_inds]
    act = activation_func("sigmoid")
    weights = torch.ones((points.shape[0], num_knn), device=points.device) / num_knn
    weights = nn.Parameter(weights, requires_grad=True)
    optimizer_w = torch.optim.Adam([weights], lr=1e-1)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_w, mode='min', factor=0.1, patience=1000, verbose=True)
    for i in range(step):
        pred_points = torch.einsum('ij,ijk->ik', act(weights), knn_sp)
        loss = torch.nn.functional.mse_loss(pred_points, points)
        optimizer_w.zero_grad()
        loss.backward()
        optimizer_w.step()
        lr_scheduler.step(loss)
        if i % 1000 == 0:
            print("iter", i, loss.item())

    weights = act(weights)
    print("weights", weights.shape, weights.min(), weights.max())
    pred_points = torch.einsum('ij,ijk->ik', weights, knn_sp)
    print((points - pred_points).abs().max())
    print("chamfer", chamfer_distance(pred_points[None], points[None])[0].item())

    return pred_points, weights.detach(), nn_inds


def estimate_normals(points, num_nn=30, flip_normal=False):

    # reference_point=np.array([0, 0, 0])
    # Convert points to PyTorch tensor
    points_tensor = points.unsqueeze(0)  # Shape: (1, N, 3)

    # Estimate normals using PyTorch3D
    normals_tensor = estimate_pointcloud_normals(
        points_tensor, neighborhood_size=num_nn, disambiguate_directions=True
    )

    # Convert normals to numpy array
    normals = normals_tensor.squeeze(0)

    if flip_normal:
        reference_point = torch.mean(points, axis=0)
        # Ensure normals are facing away from the reference point
        vectors_to_reference = points - reference_point
        dot_products = np.einsum("ij,ij->i", vectors_to_reference, normals)
        normals[dot_products < 0] *= -1

    return normals


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


def positional_encoding(x, num_frequencies, mask_ratio):
    # x: shape (N, 3)
    # num_frequencies: max frequencies to use
    # mask_ratio: fraction of frequencies to use (progressive)
    frequencies = int(num_frequencies * mask_ratio)
    # print("!!!frequencies", frequencies)
    # pe = [x]
    pe = []
    for i in range(num_frequencies):
        for fn in [torch.sin, torch.cos]:
            pe.append(fn((2.0 ** i) * x) * (i < frequencies))
    return torch.cat(pe, dim=-1)


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
        pe = positional_encoding(x, self.max_freq, mask_ratio)  # (N, enc_dim)
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


# def one_sided_chamfer_loss(src_transformed, tgt_points, nearest_indices):
#     # Gather the nearest target points using the nearest indices
#     nearest_tgt_points = tgt_points[nearest_indices]

#     # Compute the L2 distances between each transformed source point and its nearest target point
#     dists = torch.norm(src_transformed - nearest_tgt_points, dim=1)


#     # Compute the mean distance as the loss
#     loss = dists.mean()
#     return loss
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
    # final_ori_tgt_kp = np.zeros_like(cur_src_kp)
    for src_idx, tgt_idx in mapping:
        new_tgt_kp[src_idx] = tgt_np[tgt_idx]
        new_tgt_normals[src_idx] = cur_tgt_normals[tgt_idx]
        # final_ori_tgt_kp[src_idx] = ori_tgt_kp[tgt_idx]
    return new_tgt_kp, new_tgt_normals


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


def modified_icp_with_nn(
    src,
    tgt,
    model,
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
    stop_volume_ratio=0.8,
    loss_modes=[],
    hierarchical_transformation=False,
    joint_pos=[],
):
    if part_max_freqs is not None:
        save_path = os.path.join(log_dir, f"pe{model.max_freq}_p0f{part_max_freqs[0]}_p1f{part_max_freqs[1]}_temp{model.temperature}_seg{num_segments}_iter{iterations}/")
    else:
        save_path = os.path.join(log_dir, f"pe{model.max_freq}_temp{model.temperature}_seg{num_segments}_iter{iterations}/")
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
    if isinstance(tgt, list):
        save_path = save_path[:-1] + "_partmatch/"
    if loss_type != "l2":
        if isinstance(robust_c, list):
            save_path = save_path[:-1] + f"_{loss_type}_c{robust_c[0]}_{robust_c[1]}/"
        elif robust_c != 0.2:
            save_path = save_path[:-1] + f"_{loss_type}_c{robust_c}/"
        else:
            save_path = save_path[:-1] + f"_{loss_type}/"
    if USE_FPS:
        save_path = save_path[:-1] + "_kp/"
    if correspondence == "linear_sum":
        save_path = save_path[:-1] + "_LS/"
    if src_iter >= 0:
        save_path = save_path[:-1] + f"_srciter{src_iter}/"
    if use_arap:
        if joint_arap:
            save_path = save_path[:-1] + f"_joint_arap{arap_w}_k{smooth_knn}/"
        else:
            save_path = save_path[:-1] + f"_arap{arap_w}_k{smooth_knn}/"
    if avg_loss:
        save_path = save_path[:-1] + f"_avgL/"
    if add_lda:
        save_path = save_path[:-1] + f"_lda{lda_weight}/"
    if hierarchical_transformation:
        save_path = save_path[:-1] + f"_hier/"
    os.makedirs(save_path, exist_ok=True)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint))
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrt)
    # source_points, target_points: np arrays of shape (N,3) or (M,3)
    # src = torch.tensor(source_points, dtype=torch.float32, device=device)
    # tgt = torch.tensor(target_points, dtype=torch.float32, device=device)
    if isinstance(tgt, list):
        plot_pointcloud(
            torch.cat(tgt, dim=0), save_path, title=f"Target Point Cloud"
        )
        if use_arap and not joint_arap:
            arap_losses = [ARAPLoss(src[i], smooth_knn=smooth_knn) for i in range(len(src))]
        elif use_arap and joint_arap:
            arap_loss = ARAPLoss(torch.cat(src, dim=0), smooth_knn=smooth_knn)
    else:
        plot_pointcloud(
            tgt, save_path, title=f"Target Point Cloud"
        )

    match_frames = []

    if early_stop:
        init_volumes = [estimate_volume(src[i].cpu().numpy()) for i in range(len(src))]
        # cur_volumes = []
        stops = [False for i in range(len(src))]
        stop_tgt_normal_shuffled = [None for i in range(len(src))]

    stop_tgt_nn = [None for i in range(len(src))]
    stop = True

    segment_length = iterations // num_segments
    cur_seg_idx = 0
    cur_c_seg_idx = 0
    cur_c = robust_c[cur_c_seg_idx] if isinstance(robust_c, list) else robust_c

    if correspondence == "linear_sum":
        tgt_normals = []
        for i in range(len(tgt)):
            tgt_normals.append(estimate_normals_pytorch3d(tgt[i].cpu().numpy()))

    if loss_mode == "pt2plane" and correspondence == "nn":
        tgt_normals = torch.tensor(estimate_normals_pytorch3d(tgt.cpu().numpy(), knn=30), dtype=torch.float32, device=device)
    for it in tqdm.tqdm(range(iterations), desc="ICP Iterations"):
        # mask_ratio = 1.0 if mask_ratio_schedule is None else mask_ratio_schedule[it]
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
        if part_max_freqs is not None:
            src_transformed = []
            # nearest_indices = []
            tgt_nn = []
            tgt_normal_shuffled = []
            total_arap_loss = 0.
            loss = 0.
            root_R = None
            root_t = None
            for i in range(len(part_max_freqs)):

                mask_ratio = min((segment_index + 1) * 1.0 / num_segments, part_max_freqs[i] * 1.0 / model.max_freq)

                params = model(src[i], mask_ratio=mask_ratio)  # (N, 6)
                axis_angle = params[:, :3]
                translation = params[:, 3:]

                # Convert axis-angle to rotation matrix
                R = axis_angle_to_rotation_matrix(axis_angle)

                if hierarchical_transformation and i == 0:
                    # root_R would be the mean of the R
                    root_R = R.mean(dim=0)
                    root_t = translation.mean(dim=0)

                # Apply transformations
                cur_src_transformed = []
                for j in range(src[i].shape[0]):
                    if hierarchical_transformation and i > 0:
                        # normalize the point coordinate by the joint position
                        rotated = torch.matmul(R[j], src[i][j] - joint_pos[i])
                        cur_src_transformed.append(torch.matmul(root_R, rotated + translation[j] + joint_pos[i]) + root_t)
                    else:
                        rotated = torch.matmul(R[j], src[i][j])
                        cur_src_transformed.append(rotated + translation[j])
                src_transformed.append(torch.stack(cur_src_transformed, dim=0))

                if use_arap and not joint_arap:
                    total_arap_loss += arap_losses[i](src_transformed[i], add_lda=add_lda, lda_weight=lda_weight)

                # Find closest points (placeholder for nearest neighbor search)
                src_trans_np = src_transformed[i].detach().cpu().numpy()
                if isinstance(tgt, list):
                    tgt_np = tgt[i].detach().cpu().numpy()
                else:
                    tgt_np = tgt.detach().cpu().numpy()

                if correspondence == "nn":
                    tree = KDTree(tgt_np)
                    dists, cur_nearest_indices = tree.query(src_trans_np, k=1)
                    cur_nearest_indices = cur_nearest_indices.flatten()

                    if isinstance(tgt, list):
                        tgt_nn.append(tgt[i][cur_nearest_indices])
                    else:
                        tgt_nn.append(tgt[cur_nearest_indices])
                elif correspondence == "linear_sum":
                    new_tgt_kp, new_tgt_normal = linear_sum_correspondence(src_trans_np, tgt_np, tgt_normals[i], use_cos=False, alpha=0.75)
                    
                    if early_stop and not stops[i]:
                        cur_volume = estimate_volume(src_trans_np)
                        # NOTE: temporarily only check the first part
                        if i == 0 and cur_volume / init_volumes[i] < stop_volume_ratio:
                            stops[i] = True
                            stop_tgt_nn[i] = torch.from_numpy(new_tgt_kp).float().to(device)
                            stop_tgt_normal_shuffled[i] = torch.from_numpy(new_tgt_normal).float().to(device)
                        
                    if stop_tgt_nn[i] is not None:
                        tgt_nn.append(stop_tgt_nn[i])
                        tgt_normal_shuffled.append(stop_tgt_normal_shuffled[i])
                    else:
                        tgt_nn.append(torch.from_numpy(new_tgt_kp).float().to(device))
                        tgt_normal_shuffled.append(torch.from_numpy(new_tgt_normal).float().to(device))

                    if len(loss_modes):
                        if loss_modes[i] == "chamfer":
                            loss += one_sided_chamfer_loss(src_transformed[i], tgt_nn[i], loss_type=loss_type, c=cur_c)
                        elif loss_modes[i] == "pt2plane":
                            loss += point_to_plane_loss(
                                src_transformed[i], tgt_nn[i], tgt_normal_shuffled[i]
                            )

                    if avg_loss and not joint_arap and not stops[i]:
                        if loss_mode == "chamfer":
                            loss += one_sided_chamfer_loss(src_transformed[i], tgt_nn[i], loss_type=loss_type, nn_indices=arap_losses[i].nn_indices, c=cur_c)
                        elif loss_mode == "pt2plane":
                            loss += point_to_plane_loss(
                                src_transformed[i], tgt_nn[i], tgt_normal_shuffled[i]
                            )
                else:
                    raise ValueError("correspondence must be either 'nn' or 'linear_sum'")

                # if early_stop and i == 0:
                #     cur_volume = estimate_volume(src_trans_np)
                #     if cur_volume / init_volumes[i] < stop_volume_ratio:
                #         stop = True

            src_transformed = torch.cat(src_transformed, dim=0)
            tgt_nn = torch.cat(tgt_nn, dim=0)
            tgt_normal_shuffled = torch.cat(tgt_normal_shuffled, dim=0)
        else:
            mask_ratio = (segment_index + 1) * 1.0 / num_segments

            params = model(src, mask_ratio=mask_ratio)  # (N, 6)
            axis_angle = params[:, :3]
            translation = params[:, 3:]

            # Convert axis-angle to rotation matrix
            R = axis_angle_to_rotation_matrix(axis_angle)

            # Apply transformations
            src_transformed = []
            for i in range(src.shape[0]):
                rotated = torch.matmul(R[i], src[i])
                src_transformed.append(rotated + translation[i])
            src_transformed = torch.stack(src_transformed, dim=0)

            # Find closest points (placeholder for nearest neighbor search)
            # if segment_index < 2 or it % 10:
            src_trans_np = src_transformed.detach().cpu().numpy()
            tgt_np = tgt.detach().cpu().numpy()
            tree = KDTree(tgt_np)
            dists, nearest_indices = tree.query(src_trans_np, k=1)
            nearest_indices = nearest_indices.flatten()
            tgt_nn = tgt[nearest_indices]
        # Compute loss and backprop if doing a learnable approach
        if len(loss_modes) == 0: 
            if loss_mode == "chamfer" and not avg_loss:
                # loss = one_sided_chamfer_loss(src_transformed, tgt, nearest_indices)
                loss = one_sided_chamfer_loss(src_transformed, tgt_nn, loss_type=loss_type, c=cur_c)
            elif loss_mode == "pt2plane":
                loss = point_to_plane_loss(
                    src_transformed, tgt_nn, tgt_normal_shuffled
                )
        # else:
        #     raise ValueError("loss_mode must be either 'chamfer' or 'pt2plane'")

        if use_arap and joint_arap:
            if avg_loss:
                loss = one_sided_chamfer_loss(src_transformed, tgt_nn, loss_type=loss_type, nn_indices=arap_loss.nn_indices, c=cur_c)
            total_arap_loss = arap_loss(src_transformed, add_lda=add_lda, lda_weight=lda_weight)

        if use_arap:
            loss += total_arap_loss * arap_w
        # (Place your backprop code here using loss)
        # For example:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if early_stop:
            tqdm.tqdm.write(f"Iteration {it} Loss ({loss_mode}): {loss.item():.6f}, ARAP Loss: {total_arap_loss:.6f} Volume ratio: {(cur_volume / init_volumes[0]):.6f}")
        else:
            # Print loss value
            tqdm.tqdm.write(f"Iteration {it} Loss ({loss_mode}): {loss.item():.6f}, ARAP Loss: {total_arap_loss:.6f}")

        if early_stop:
            for cur_stop in stops:
                stop = stop and cur_stop

        # Save the transformed point cloud
        if it % 10 == 0 or it == iterations - 1 or (stop and early_stop):
            plot_pointcloud(
                src_transformed, save_path, title=f"Transformed Point Cloud Iter {it}",
                # extra_points=tgt, plot_match=False
            )
            if plot_match:
                # plot a vector that connects each point to the corresponding nearest neighbor
                cur_frame = plot_pointcloud(
                    src_transformed, None, title=f"Match Iter {it}",
                    extra_points=tgt_nn, plot_match=True
                )
                match_frames.append(cur_frame)
                if it % 100 == 0 or (stop and early_stop):
                    imageio.mimsave(os.path.join(save_path, "match_frames.mp4"), match_frames, fps=10)
                    torch.save(model.state_dict(), os.path.join(save_path, f"deform_model_{it}.pth"))
        if stop and early_stop:
            break
        else:
            stop = True
    if plot_match:
        imageio.mimsave(os.path.join(save_path, "match_frames.mp4"), match_frames, fps=10)
        # save model weights
        torch.save(model.state_dict(), os.path.join(save_path, f"deform_model_final.pth"))

    return model, src_transformed.detach().cpu().numpy()


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
            torch.cat(tgt, dim=0), save_path, title=f"Target Point Cloud"
        )
        match_frames = []


    if use_arap and not joint_arap:
        arap_losses = [ARAPLoss(src[i], smooth_knn=smooth_knn) for i in range(len(src))]
    elif use_arap and joint_arap:
        arap_loss = ARAPLoss(torch.cat(src, dim=0), smooth_knn=smooth_knn)

    if early_stop:
        init_volumes = [estimate_volume(src[i].cpu().numpy()) for i in range(len(src))]

    total_params = []
    for model in models:
        model.to(device)
        model.train()
        total_params += list(model.parameters())
    optimizer = torch.optim.Adam(total_params, lr=lrt)
    stop = False
    new_tgt = None

    segment_length = iterations // num_segments
    cur_seg_idx = 0
    cur_c_seg_idx = 0
    cur_c = robust_c[cur_c_seg_idx] if isinstance(robust_c, list) else robust_c

    loss_history = []

    total_num_pts = sum([len(c_src) for c_src in src])

    if correspondence == "linear_sum":
        tgt_normals = []
        for i in range(len(tgt)):
            tgt_normals.append(estimate_normals_pytorch3d(tgt[i].cpu().numpy()))

    if loss_mode == "pt2plane" and correspondence == "nn":
        tgt_normals = torch.tensor(estimate_normals_pytorch3d(tgt.cpu().numpy(), knn=30), dtype=torch.float32, device=device)
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
        # nearest_indices = []
        tgt_nn = []
        tgt_normal_shuffled = []
        total_arap_loss = 0.
        loss = 0.
        # losses = []
        root_R = None
        root_t = None
        for i in range(len(src)):

            # mask_ratio = min((segment_index + 1) * 1.0 / num_segments, part_max_freqs[i] * 1.0 / model.max_freq)

            # params = model(src[i], mask_ratio=mask_ratio)  # (N, 6)
            # params = transformations[i]
            # rot_param = params[:6]
            # R = pytorch3d.transforms.rotation_6d_to_matrix(rot_param.unsqueeze(0)).squeeze(0)
            # translation = params[6:]

            model = models[i]
            mask_ratio = min((segment_index + 1) * 1.0 / num_segments, part_max_freqs[i] * 1.0 / model.max_freq)

            params = model(src[i], mask_ratio=mask_ratio)  # (N, 6)
            axis_angle = params[:, :3]
            translation = params[:, 3:]

            # Convert axis-angle to rotation matrix
            R = axis_angle_to_rotation_matrix(axis_angle)

            cur_src_transformed = []
            for j in range(src[i].shape[0]):
                if i > 0:
                    # normalize the point coordinate by the joint position
                    rotated = torch.matmul(R[j], src[i][j] - joint_pos[i])
                    params = models[0](joint_pos[i].unsqueeze(0), mask_ratio=mask_ratio)  # (N, 6)
                    axis_angle = params[:, :3]
                    root_t = params[:, 3:]
                    root_R = axis_angle_to_rotation_matrix(axis_angle)

                    # cur_src_transformed.append(torch.matmul(root_R, rotated + translation[j] + joint_pos[i]) + root_t)
                    cur_src_transformed.append((torch.matmul(root_R, rotated + joint_pos[i]) + root_t).squeeze(0))
                else:
                    rotated = torch.matmul(R[j], src[i][j])
                    cur_src_transformed.append(rotated + translation[j])
                    # root_R = R.mean(dim=0)
                    # root_t = translation.mean(dim=0)
            
            src_transformed.append(torch.stack(cur_src_transformed, dim=0))

            if use_arap and not joint_arap:
                total_arap_loss += arap_losses[i](src_transformed[i], add_lda=add_lda, lda_weight=lda_weight)

            # Find closest points (placeholder for nearest neighbor search)
            src_trans_np = src_transformed[i].detach().cpu().numpy()
            if isinstance(tgt, list):
                tgt_np = tgt[i].detach().cpu().numpy()
            else:
                tgt_np = tgt.detach().cpu().numpy()

            if correspondence == "nn":
                tree = KDTree(tgt_np)
                dists, cur_nearest_indices = tree.query(src_trans_np, k=1)
                cur_nearest_indices = cur_nearest_indices.flatten()

                if isinstance(tgt, list):
                    tgt_nn.append(tgt[i][cur_nearest_indices])
                else:
                    tgt_nn.append(tgt[cur_nearest_indices])
            elif correspondence == "linear_sum":
                new_tgt_kp, new_tgt_normal = linear_sum_correspondence(src_trans_np, tgt_np, tgt_normals[i], use_cos=False, alpha=0.75)
                
                # # NOTE: temporarily only check the first part
                # if i == 0 and estimate_volume(src_trans_np) / init_volumes[i] < stop_volume_ratio:
                #     tgt_nn.append(src_transformed[i])
                # else:
                #     tgt_nn.append(torch.from_numpy(new_tgt_kp).float().to(device))
                # if stop_tgt_nn[i] is not None:
                #     tgt_nn.append(stop_tgt_nn[i])
                #     tgt_normal_shuffled.append(stop_tgt_normal_shuffled[i])
                # else:


                tgt_nn.append(torch.from_numpy(new_tgt_kp).float().to(device))
                tgt_normal_shuffled.append(torch.from_numpy(new_tgt_normal).float().to(device))

                # NOTE: temp commented out
                if loss_mode == "chamfer":
                    # NOTE: temporarily only check the first part
                    if i == 0 and (estimate_volume(src_trans_np) / init_volumes[i] < stop_volume_ratio):
                        if new_tgt is None:
                            print("!! Changing target")
                            new_tgt = src_transformed[i].clone().detach()
                        loss += one_sided_chamfer_loss(src_transformed[i], new_tgt, loss_type=loss_type, c=cur_c, reduce="sum")
                    else:
                        loss += one_sided_chamfer_loss(src_transformed[i], tgt_nn[i], loss_type=loss_type, c=cur_c, reduce="sum")
                elif loss_mode == "pt2plane":
                    loss += point_to_plane_loss(
                        src_transformed[i], tgt_nn[i], tgt_normal_shuffled[i]
                    )



                # if len(loss_modes):
                #     if loss_modes[i] == "chamfer":
                #         loss += one_sided_chamfer_loss(src_transformed[i], tgt_nn[i], loss_type=loss_type, c=cur_c)
                #     elif loss_modes[i] == "pt2plane":
                #         loss += point_to_plane_loss(
                #             src_transformed[i], tgt_nn[i], tgt_normal_shuffled[i]
                #         )

                # if avg_loss and not joint_arap and not stops[i]:
                #     if loss_mode == "chamfer":
                #         loss += one_sided_chamfer_loss(src_transformed[i], tgt_nn[i], loss_type=loss_type, nn_indices=arap_losses[i].nn_indices, c=cur_c)
                #     elif loss_mode == "pt2plane":
                #         loss += point_to_plane_loss(
                #             src_transformed[i], tgt_nn[i], tgt_normal_shuffled[i]
                #         )
            else:
                raise ValueError("correspondence must be either 'nn' or 'linear_sum'")

            # if early_stop and i == 0:
            #     cur_volume = estimate_volume(src_trans_np)
            #     if cur_volume / init_volumes[i] < stop_volume_ratio:
            #         stop = True

        src_transformed = torch.cat(src_transformed, dim=0)
        tgt_nn = torch.cat(tgt_nn, dim=0)
        if len(tgt_normal_shuffled) > 0:
            # tgt_normal_shuffled = torch.zeros_like(src_transformed)
            tgt_normal_shuffled = torch.cat(tgt_normal_shuffled, dim=0)
        
        # # Compute loss and backprop if doing a learnable approach
        # if len(loss_modes) == 0: 
        #     if loss_mode == "chamfer" and not avg_loss:
        #         # loss = one_sided_chamfer_loss(src_transformed, tgt, nearest_indices)
        #         loss = one_sided_chamfer_loss(src_transformed, tgt_nn, loss_type=loss_type, c=cur_c)
        #     elif loss_mode == "pt2plane":
        #         loss = point_to_plane_loss(
        #             src_transformed, tgt_nn, tgt_normal_shuffled
        #         )
        # else:
        #     raise ValueError("loss_mode must be either 'chamfer' or 'pt2plane'")

        if use_arap and joint_arap:
            # if avg_loss:
            #     loss = one_sided_chamfer_loss(src_transformed, tgt_nn, loss_type=loss_type, nn_indices=arap_loss.nn_indices, c=cur_c)
            total_arap_loss = arap_loss(src_transformed, add_lda=add_lda, lda_weight=lda_weight)

        if use_arap:
            loss = loss * 1. / total_num_pts + total_arap_loss * arap_w
        # For example:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if early_stop:
        #     tqdm.tqdm.write(f"Iteration {it} Loss ({loss_mode}): {loss.item():.6f}, ARAP Loss: {total_arap_loss:.6f} Volume ratio: {(cur_volume / init_volumes[0]):.6f}")
        # else:
            # Print loss value
        if save_logs:
            tqdm.tqdm.write(f"Iteration {it} Loss ({loss_mode}): {loss.item():.6f}, ARAP Loss: {total_arap_loss:.6f}")

        loss_history.append(loss.item())
        # check loss plateau
        if it > min_iter and len(loss_history) > min_iter:
            if np.mean(loss_history[-10:]) > np.mean(loss_history[-20:-10]):
                stop = True
            else:
                stop = False

        # if early_stop:
        #     for cur_stop in stops:
        #         stop = stop and cur_stop

        if save_logs:
            # Save the transformed point cloud
            if it % 10 == 0 or it == iterations - 1 or (stop and early_stop):
                plot_pointcloud(
                    src_transformed, save_path, title=f"Transformed Point Cloud Iter {it}",
                    # extra_points=tgt, plot_match=False
                )
                if plot_match:
                    # plot a vector that connects each point to the corresponding nearest neighbor
                    cur_frame = plot_pointcloud(
                        src_transformed, None, title=f"Match Iter {it}",
                        extra_points=tgt_nn, plot_match=True
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
                    src_transformed, save_dir, title=f"Time Step {time_step}",
                    extra_points=tgt_nn, plot_match=True
                )

    return models, src_transformed.detach(), cur_frame, it


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


# fps_k = 1000
num_keypoints_left = 96
num_keypoints_right = 96
num_keypoints_body = 256
# num_keypoints_body = 128
# num_keypoints_body = 5600
all_num_keypoints = [num_keypoints_body, num_keypoints_left, num_keypoints_right]
icp_instance = False

# num_keypoints_body = 192
scale = 10.0
num_layers = 3
# num_layers = 4
hidden_dim = 256
L = 8
# L = 4
# L = 10
# L = 0
log_dir = "fit_pointcloud_logs"
# if USE_POINTNET:
#     # if DENSIFY_BODY:
#     #     exp_dir = f"multi_mlp_icp_shift_pe{L}_pointnet_densify"
#     # elif CONCAT_POS_DEFORM:
#     #     exp_dir = f"multi_mlp_icp_shift_pe{L}_pointnet_concat"
#     # else:
#     exp_dir = f'multi_ot_icp_shift_pe{L}_pointnet'
#     if DENSIFY_BODY:
#         exp_dir += "_densify"
#     if CONCAT_POS_DEFORM:
#         exp_dir += "_concat"
#     if USE_FPS:
#         exp_dir += "_fps"
#     if num_layers != 3:
#         exp_dir += f"_layers{num_layers}"
#     if DEFORM_KP_ONLY:
#         exp_dir += "_kp"
# else:
#     exp_dir = f"multi_mlp_icp_shift_pe{L}"
# exp_dir = f'ot_wing{num_keypoints}_body{num_keypoints_body}'
exp_dir = f"learnable_icp_wingL{num_keypoints_left}_wingR{num_keypoints_right}_body{num_keypoints_body}"
if icp_instance:
    exp_dir += "_icp"
if USE_COS:
    exp_dir += "_cos"

# exp_id = 7
exp_id = 22

exp_sub_dir = f"exp_{exp_id}"
log_dir = os.path.join(log_dir, exp_dir, exp_sub_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
# copy the current file to log_dir
shutil.copy(__file__, log_dir)

save_img_dir = os.path.join(log_dir, "images")
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir, exist_ok=True)

# src_iter = 30000
src_iter = 0
# src_iter = 24000
src_pc_path = f"/NAS/spa176/papr-retarget/point_clouds/butterfly/points_{src_iter}.npy"
# src_pc_path = "/NAS/spa176/papr-retarget/point_clouds/butterfly/points_0.npy"
# src_pc_path = "/NAS/spa176/papr-retarget/point_clouds/butterfly/points_30000.npy"
# src_pc_path = f"/NAS/spa176/papr-retarget/point_clouds/butterfly/points_{num_keypoints_body}.npy"
tgt_pc_path = "/NAS/spa176/papr-retarget/point_clouds/hummingbird/points_0.npy"
# tgt_pc_path = "/NAS/spa176/papr-retarget/point_clouds/butterfly/points_0.npy"

src_pc = np.load(src_pc_path)
tgt_pc = np.load(tgt_pc_path)

src_pc = torch.tensor(src_pc).float().to(device)
tgt_pc = torch.tensor(tgt_pc).float().to(device)

src_pc = src_pc / scale
tgt_pc = tgt_pc / scale
# tgt_pc[..., 0] += 0.1
# tgt_pc[..., 1] -= 0.2
# tgt_pc[..., 2] += 0.3
# # shuffle the point cloud
# tgt_pc = tgt_pc[torch.randperm(tgt_pc.shape[0])]
print("src_pc: ", src_pc.shape, src_pc.min(), src_pc.max())
print("tgt_pc: ", tgt_pc.shape, tgt_pc.min(), tgt_pc.max())

# Plot the point clouds
plot_pointcloud(src_pc, log_dir, title="Source Point Cloud")
plot_pointcloud(tgt_pc, log_dir, title="Target Point Cloud")


# load wing indices
but_wing_indices = np.load("but_wing_indices.npy")
but_body_indices = np.setdiff1d(np.arange(len(src_pc)), but_wing_indices)

but_wing_indices_right = np.load("but_wing_indices_right.npy")
but_wing_indices_left = np.setdiff1d(
    np.arange(len(but_wing_indices)), but_wing_indices_right
)

if DENSIFY_BODY:
    # but_body_indices = np.setdiff1d(np.arange(len(src_pc)), but_wing_indices)
    # add points to the body of cur_src_pc by inserting one point between each pair of nearest neighbors
    cur_src_pc = src_pc[but_body_indices]
    cur_src_pc_np = cur_src_pc.cpu().numpy()
    tree = KDTree(cur_src_pc_np)
    nn_dists, nn_inds = tree.query(cur_src_pc_np, k=2)
    new_points = cur_src_pc[nn_inds].sum(1) / 2
    src_pc = torch.cat([src_pc, new_points], 0)
    but_body_indices_concat = np.setdiff1d(np.arange(len(src_pc)), but_wing_indices)
bird_wing_indices = np.load("hummingbird_wing_indices.npy")
bird_body_indices = np.setdiff1d(np.arange(len(tgt_pc)), bird_wing_indices)

bird_wing_indices_right = np.load("hummingbird_wing_indices_right.npy")
bird_wing_indices_left = np.setdiff1d(
    np.arange(len(bird_wing_indices)), bird_wing_indices_right
)


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
tgt_part_indices = [
    bird_body_indices,
    bird_wing_indices[bird_wing_indices_left],
    bird_wing_indices[bird_wing_indices_right],
]
use_hierarchical_transformation = True
total_part_num = 3
joint_k = 1

# icp_model = PointwiseTransformNet(base_channels=64, max_freq=0).to(device)
temperature = 50.0
# temperature = 25.0
max_freq = 1
icp_models = [PointwiseTransformNet(
    base_channels=32, max_freq=max_freq, temperature=temperature
).to(device) for _ in range(total_part_num)]

if USE_ICP:

    src_kps = []
    tgt_kps = []
    tgt_pcs = []
    src_kp_indices = []

    for part_idx in range(total_part_num):

        # FPS
        cur_src_kp, src_kp_idx = sfp(
            src_pc[src_part_indices[part_idx]].unsqueeze(0), K=all_num_keypoints[part_idx]
        )
        # cur_src_kp = cur_src_kp.squeeze(0)
        cur_tgt_kp, tgt_kp_idx = sfp(tgt_pc[tgt_part_indices[part_idx]].unsqueeze(0), K=all_num_keypoints[part_idx])

        # ori_tgt_kps.append(torch.from_numpy(final_ori_tgt_kp).float())

        src_kps.append(cur_src_kp.squeeze())
        src_kp_indices.append(src_kp_idx)
        # kp_indices.append(src_kp_idx)
        tgt_kps.append(cur_tgt_kp.squeeze())
        # tgt_pcs.append(cur_tgt_pc.clone().cpu())
        # torch.save(
        #     tgt_pc[tgt_part_indices[part_idx]],
        #     os.path.join(log_dir, f"part{part_idx}_tgt_pc_ori.pth")
        # )
        tgt_pcs.append(
            tgt_pc[tgt_part_indices[part_idx]]
        )

    if use_hierarchical_transformation:
        # find the joint position
        
        joint_pos = [None]
        for part_idx in range(1, 3, 1):
            cur_joint_pos = find_wing_anchor_point_torch(
                tgt_kps[part_idx], tgt_kps[0], k=joint_k
            )
            joint_pos.append(cur_joint_pos)
            # if part_idx == 1:
            #     # use plotly to visualize the joint position with tgt_kps[part_idx] and tgt_kps[0] also drawn in different colors
            #     fig = go.Figure()
            #     fig.add_trace(
            #         go.Scatter3d(
            #             x=tgt_kps[part_idx][:, 0].cpu().numpy(),
            #             y=tgt_kps[part_idx][:, 1].cpu().numpy(),
            #             z=tgt_kps[part_idx][:, 2].cpu().numpy(),
            #             mode="markers",
            #             marker=dict(
            #                 size=2,
            #                 color="red",
            #             ),
            #             name="Wing",
            #         )
            #     )
            #     fig.add_trace(
            #         go.Scatter3d(
            #             x=tgt_kps[0][:, 0].cpu().numpy(),
            #             y=tgt_kps[0][:, 1].cpu().numpy(),
            #             z=tgt_kps[0][:, 2].cpu().numpy(),
            #             mode="markers",
            #             marker=dict(
            #                 size=2,
            #                 color="blue",
            #             ),
            #             name="Body",
            #         )
            #     )
            #     fig.add_trace(
            #         go.Scatter3d(
            #             x=joint_pos[part_idx][0].cpu().numpy(),
            #             y=joint_pos[part_idx][1].cpu().numpy(),
            #             z=joint_pos[part_idx][2].cpu().numpy(),
            #             mode="markers",
            #             marker=dict(
            #                 size=5,
            #                 color="green",
            #             ),
            #             name="Joint",
            #         )
            #     )
            #     fig.update_layout(
            #         scene=dict(
            #             xaxis_title="X",
            #             yaxis_title="Y",
            #             zaxis_title="Z",
            #             aspectmode="cube",
            #         ),
            #         title="Joint Position",
            #         width=800,
            #         height=800,
            #         margin=dict(l=0, r=0, b=0, t=0),
            #     )
            #     fig.show()
            #     exit(0)


        # if TRAIN_ICP:
            # # train icp model
            # icp_model, src_transformed = modified_icp_with_nn(
            #     tgt_pc,
            #     src_pc,
            #     icp_model,
            #     log_dir,
            #     iterations=500,
            #     num_segments=max_freq,
            #     # num_segments=1,
            #     loss_mode="chamfer",
            #     # loss_mode="pt2plane",
            # )



            # # train icp model
            # icp_model, src_transformed = modified_icp_with_nn(
            #     # [tgt_pc[bird_body_indices], tgt_pc[bird_wing_indices]],
            #     # [src_pc[but_body_indices], src_pc[but_wing_indices]],
            #     tgt_kps,
            #     src_kps,
            #     icp_model,
            #     log_dir,
            #     iterations=500,
            #     # iterations=1000,
            #     num_segments=max_freq,
            #     # num_segments=1,
            #     loss_mode="chamfer",
            #     # loss_mode="pt2plane",
            #     # part_max_freqs=[2, 5],
            #     # part_max_freqs=[2, 2],
            #     # part_max_freqs=[2, 2, 2],
            #     # part_max_freqs=[5, 5, 5],
            #     # part_max_freqs=[4, 4, 4],
            #     # part_max_freqs=[2, max_freq, max_freq],
            #     part_max_freqs=[max_freq, max_freq, max_freq],
            #     # part_max_freqs=[1, 1],
            #     # checkpoint="/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body256/pe2_p0f2_p1f2_temp50.0_seg2_iter200_partmatch/deform_model.pth",
            #     # checkpoint="/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body3400/pe2_p0f2_p1f2_temp50.0_seg1_iter200_partmatch/deform_model.pth",
            #     # checkpoint="/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body0/pe2_p0f2_p1f2_temp25.0_seg2_iter200_partmatch/deform_model.pth",
            #     # checkpoint="/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body3400/pe2_p0f2_p1f2_temp25.0_seg1_iter200_partmatch/deform_model.pth",
            #     # checkpoint="/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body256/pe1_p0f1_p1f1_temp25.0_seg1_iter500_partmatch_kp_LS_joint_arap1_k90/deform_model.pth",
            #     # checkpoint="/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body256/pe1_p0f1_p1f1_temp25.0_seg1_iter200_partmatch_kp_LS_srciter12000_joint_arap1_k90/deform_model.pth",
            #     # checkpoint="/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body256/pe1_p0f1_p1f1_temp25.0_seg1_iter1000_partmatch_kp_LS_srciter0_joint_arap10_k10/deform_model.pth",
            #     # checkpoint="/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body256/pe1_p0f1_p1f1_temp25.0_seg1_iter1000_partmatch_kp_LS_srciter0_joint_arap30_k10/deform_model.pth",
            #     # checkpoint="/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body256/pe1_p0f1_p1f1_temp25.0_seg1_iter500_partmatch_kp_LS_srciter12000_joint_arap30_k10/deform_model.pth",
            #     plot_match=True,
            #     # loss_type="hubert",
            #     # loss_type="geman",
            #     # loss_type="p2p"
            #     correspondence="linear_sum",
            #     lrt=2.5e-3,
            #     src_iter=src_iter,
            #     use_arap=True,
            #     smooth_knn=10,
            #     arap_w=10,
            #     joint_arap=True,
            #     # iter_schedule=[100],
            #     # avg_loss=True,
            #     # robust_c=0.05,
            #     # robust_c=[1.0, 0.05, 0.01],
            #     # c_schedule=[300, 600],
            #     # add_lda=True,
            #     # lda_weight=0.025,
            #     # early_stop=True,
            #     stop_volume_ratio=0.97,
            #     # loss_modes=["chamfer", "pt2plane", "pt2plane"],
            #     hierarchical_transformation=use_hierarchical_transformation,
            #     joint_pos=joint_pos,
            # )

# initialize the transformation parameters, each with 9 dimentional vector initialized to 0
transformations = torch.nn.ParameterList(
    [
        torch.nn.Parameter(torch.tensor([1., 0., 0., 0., 1., 0., 0., 0., 0.], dtype=torch.float32).to(device))
        for _ in range(len(src_kps))
    ]
)



start = 0
end = 30001
interval = 200
scale = 10.0
smooth_window_size = 35

src_pc_dir = "/NAS/spa176/papr-retarget/point_clouds/butterfly/"

src_keypoints = [[] for _ in range(total_part_num)]
for idx in tqdm.tqdm(range(start, end, interval)):
    src_pc_path = os.path.join(src_pc_dir, f"points_{idx}.npy")
    cur_src_pc = np.load(src_pc_path)
    cur_src_pc = cur_src_pc / scale
    # point_clouds.append(cur_src_pc)
    for part_idx in range(total_part_num):
        cur_src_kp = cur_src_pc[src_part_indices[part_idx]][src_kp_indices[part_idx].cpu().numpy()]
        cur_src_kp = torch.tensor(cur_src_kp).float().to(device)
        src_keypoints[part_idx].append(cur_src_kp)
for part_idx in range(total_part_num):
    src_keypoints[part_idx] = smooth_point_cloud_torch(torch.cat(src_keypoints[part_idx], dim=0), smooth_window_size)

total_time_steps = len(src_keypoints[0])

transferred_tgt_kps = []
match_frames = []
# TRAIN_ICP_MODEL = False
TRAIN_ICP_MODEL = True
if TRAIN_ICP_MODEL:
    # loop through each time step
    for time_step in tqdm.tqdm(range(total_time_steps)):

        # cur_iterations = 800 if time_step == 0 else 500
        cur_min_iter = 100 if time_step == 0 else 10
        save_log = True if time_step == 0 else False
        cur_src_kps = [src_keypoint[time_step] for src_keypoint in src_keypoints]

        icp_models, src_transformed, match_frame, train_iters = modified_icp_nn_new(
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
            correspondence="linear_sum",
            # lrt=2.5e-3,
            src_iter=src_iter,
            use_arap=True,
            smooth_knn=20,
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
            # stop_volume_ratio=0.97,
            # loss_modes=["chamfer", "pt2plane", "pt2plane"],
            # hierarchical_transformation=use_hierarchical_transformation,
            joint_pos=joint_pos,
            joint_k=joint_k,
            save_logs=save_log,
            min_iter=cur_min_iter,
            save_img_dir=save_img_dir,
            time_step=time_step
        )

        # if time_step == 0:
        #     exit(0)

        tqdm.tqdm.write(f"Time step {time_step} converged in {train_iters} iterations")

        match_frames.append(match_frame)
        if time_step % 10 == 0:
            imageio.mimsave(os.path.join(save_img_dir, "match_frames.mp4"), match_frames, fps=10)

        transferred_tgt_kps.append(src_transformed.unsqueeze(0))
        torch.save(torch.cat(transferred_tgt_kps, dim=0), os.path.join(log_dir, f"transferred_tgt_kps.pth"))

    # save the match frames to save_img_dir as mp4
    if len(match_frames) > 0:
        imageio.mimsave(os.path.join(save_img_dir, "match_frames.mp4"), match_frames, fps=10)

    transferred_tgt_kps = torch.cat(transferred_tgt_kps, dim=0)
    # save the transferred_tgt_kps to disk as torch tensor
    torch.save(transferred_tgt_kps, os.path.join(log_dir, f"transferred_tgt_kps.pth"))
else:
    transferred_tgt_kps = torch.load(os.path.join(log_dir, f"transferred_tgt_kps.pth"))
    print("Loaded", transferred_tgt_kps.shape)

rbf_kernel = "polyharmonic_spline"
ext_params = {"k": 3} if rbf_kernel == "polyharmonic_spline" else {}
rbf_radius = 10

print("=" * 20)
print("Motion transfer finished, start to deform the point cloud")

total_deformed_pcs = []
for cur_step in tqdm.tqdm(range(total_time_steps)):
    cur_deformed_pcs = []
    for part_idx in range(total_part_num):
        start_idx = 0 if part_idx == 0 else start_idx + len(src_kp_indices[part_idx - 1][0])
        end_idx = start_idx + len(src_kp_indices[part_idx][0])
        if cur_step == 0:
            print(f"Slicing part_idx: {part_idx}, start_idx: {start_idx}, end_idx: {end_idx}")
        rbf = RBF(
            original_control_points=tgt_kps[part_idx].cpu().numpy() * scale,
            deformed_control_points=transferred_tgt_kps[cur_step, start_idx:end_idx, :].cpu().numpy() * scale,
            radius=rbf_radius,
            func=rbf_kernel,
            extra_parameter=ext_params,
        )
        # # Deform the surface points
        deformed_full_point_cloud = rbf(tgt_pcs[part_idx].cpu().numpy() * scale)
        cur_deformed_pcs.append(
            torch.from_numpy(deformed_full_point_cloud)
            .float()
            .to(device)
        )
    total_deformed_pcs.append(torch.cat(cur_deformed_pcs, dim=0))

total_deformed_pcs = torch.stack(total_deformed_pcs, dim=0)

target_exp_index = 13
save_model_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-{target_exp_index}-exp{exp_id}/"
os.makedirs(save_model_path, exist_ok=True)

state_dict = torch.load("/NAS/spa176/papr-retarget/experiments/hummingbird-start-1/model_no_rot_fix_name.pth")
step = list(state_dict.keys())[0]
state_dict = state_dict[step]

new_pc_feats = [state_dict["pc_feats"][tgt_part_indices[part_idx]] for part_idx in range(total_part_num)]
new_pc_feats = torch.cat(new_pc_feats, dim=0)
state_dict["pc_feats"] = new_pc_feats

new_points_influ_scores = [state_dict["points_influ_scores"][tgt_part_indices[part_idx]] for part_idx in range(total_part_num)]
new_points_influ_scores = torch.cat(new_points_influ_scores, dim=0)
state_dict["points_influ_scores"] = new_points_influ_scores

# align the deformed_pc with original pc
converged, rmse, Xt, RTs, t_history = icp(
    total_deformed_pcs[0].unsqueeze(0), state_dict["points"].unsqueeze(0), max_iterations=300
)
print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
# deformed_pc = Xt.squeeze(0)

R = RTs.R
T = RTs.T

total_deformed_pcs = torch.bmm(total_deformed_pcs, R.expand(total_deformed_pcs.shape[0], 3, 3)) + T[:, None, :].expand(total_deformed_pcs.shape[0], -1, 3)

save_pc_name = f"total_deformed_pc_rbf.pth"
# save the transformed deformed_pc
torch.save(total_deformed_pcs, save_model_path + save_pc_name)


smoothed_total_deformed_pcs = smooth_point_cloud_torch(total_deformed_pcs, smooth_window_size)
save_pc_name = f"total_deformed_pc_rbf_smooth.pth"
torch.save(smoothed_total_deformed_pcs, save_model_path + save_pc_name)


LDA(total_deformed_pcs, smooth_knn=100)
save_pc_name = f"total_deformed_pc_rbf_lda.pth"
torch.save(total_deformed_pcs, save_model_path + save_pc_name)

LDA(smoothed_total_deformed_pcs, smooth_knn=100)
save_pc_name = f"total_deformed_pc_rbf_lda_smooth.pth"
torch.save(smoothed_total_deformed_pcs, save_model_path + save_pc_name)

state_dict["points"] = total_deformed_pcs[0]
save_sd = {step: state_dict}
torch.save(save_sd, save_model_path + "model.pth")


# for part_idx in range(total_part_num):
#     start_idx = 0 if part_idx == 0 else start_idx + len(tgt_part_indices[part_idx - 1])
#     end_idx = start_idx + len(tgt_part_indices[part_idx])
#     print(f"Slicing part_idx: {part_idx}, start_idx: {start_idx}, end_idx: {end_idx}")
#     # find out the transferred points for the part
#     cur_transferred_tgt_kps = transferred_tgt_kps[:, start_idx:end_idx, :] * scale

print("Model saved!")




exit(0)


























#     else:
#         # checkpoint = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body256/pe2_p0f2_p1f2_temp50.0_seg2_iter200_partmatch/deform_model.pth"
#         # checkpoint = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body256/pe1_p0f1_p1f1_temp25.0_seg1_iter1000_partmatch_kp_LS_srciter0_joint_arap30_k10/deform_model.pth"
#         checkpoint = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/learnable_icp_wingL96_wingR96_body256/exp_1/pe5_p0f5_p1f5_temp50.0_seg5_iter500_partmatch_kp_LS_srciter0_joint_arap10_k40/deform_model_final.pth"
#         icp_model.load_state_dict(torch.load(checkpoint))
#         icp_model.eval()

#         mask_ratio = 1.0

#         params = icp_model(tgt_pc, mask_ratio=mask_ratio)  # (N, 6)
#         axis_angle = params[:, :3]
#         translation = params[:, 3:]

#         # Convert axis-angle to rotation matrix
#         R = axis_angle_to_rotation_matrix(axis_angle)

#         # Apply transformations
#         src_transformed = []
#         for i in range(tgt_pc.shape[0]):
#             rotated = torch.matmul(R[i], tgt_pc[i])
#             src_transformed.append(rotated + translation[i])
#         tgt_pc = torch.stack(src_transformed, dim=0).detach()

#         # save the rotation and translation
#         # torch.save(R, os.path.join(log_dir, "R.pth"))
#         # torch.save(translation, os.path.join(log_dir, "T.pth"))

#     # # ICP
#     # converged, rmse, Xt, RTs, t_history = icp(tgt_pc.unsqueeze(0), src_pc.unsqueeze(0))
#     # print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
#     # plot_pointcloud(Xt.squeeze(), log_dir, title="ICP Point Cloud")
#     # tgt_pc = Xt.squeeze(0)
#     # exit(0)

# # print("R :", RTs.R)
# # print("t :", RTs.T)
# # print("s :", RTs.s)
# # # save the RTs to disk as torch tensor
# # torch.save(RTs, "bird_to_but_RTs.pth")

# # # # save the tgt_pc to disk as torch tensor
# # # torch.save(tgt_pc, "rotated_bird_pc.pth")
# # print(a)
# init_kps = []
# kp_indices = []
# tgt_kps = []
# tgt_pcs = []
# boundary_indices = []
# ori_tgt_kps = []


# if TRAIN_DEFORM_NET:
#     total_result = []
#     total_kp_result = []

#     for part_idx in range(3):
#         # for part_idx in range(1, 2):
#         # for part_idx in range(2, 3):
#         # for part_idx in range(1):

#         cur_src_pc = src_pc[src_part_indices[part_idx]]
#         cur_tgt_pc = tgt_pc[tgt_part_indices[part_idx]]
#         ori_tgt_pc = cur_tgt_pc.clone()
#         np.save(
#             os.path.join(log_dir, f"part{part_idx}_tgt_pc_ori.npy"),
#             ori_tgt_pc.cpu().numpy(),
#         )
#         # if USE_FPS:
#         #     # FPS
#         #     cur_src_kp, src_kp_idx = sfp(cur_src_pc.unsqueeze(0), K=all_num_keypoints[part_idx])
#         #     # cur_src_kp = cur_src_kp.squeeze(0)
#         #     cur_tgt_kp = sfp(cur_tgt_pc.unsqueeze(0), K=all_num_keypoints[part_idx])[0]

#         # if part_idx < 2:
#         #     # align the part points
#         #     converged, rmse, Xt, RTs, t_history = icp(
#         #         cur_tgt_pc.unsqueeze(0), cur_src_pc.unsqueeze(0)
#         #     )
#         #     print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
#         #     cur_tgt_pc = Xt.squeeze(0)
#         #     # converged, rmse, Xt, RTs, t_history = icp(
#         #     #     cur_tgt_kp, cur_src_kp
#         #     # )
#         #     # print(
#         #     #     f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}"
#         #     # )
#         #     # cur_tgt_kp = Xt

#         # find the point in the current wing point cloud that is closest point index to the body point cloud
#         if part_idx < 2 and ADJUST_BOUNDARY:
#             # tree = cKDTree(src_pc[but_body_indices].cpu().numpy())
#             # # Query the KDTree with points from set A
#             # dists, indices = tree.query(cur_src_pc.cpu().numpy(), k=1)
#             # # Find the point in A with the minimum distance to any point in B
#             # min_dist_index = np.argmin(dists)
#             # src_min_pt = cur_src_pc[min_dist_index]

#             # tree = cKDTree(tgt_pc[bird_body_indices].cpu().numpy())
#             # # Query the KDTree with points from set A
#             # dists, indices = tree.query(cur_tgt_pc.cpu().numpy(), k=1)
#             # # Find the point in A with the minimum distance to any point in B
#             # min_dist_index = np.argmin(dists)
#             # tgt_min_pt = cur_tgt_pc[min_dist_index]

#             # Find the boundary points between the two parts
#             boundary_points1_indices, boundary_points2_indices = find_boundary_points(
#                 cur_src_pc, src_pc[but_body_indices], threshold=0.01
#             )
#             src_bound_pts = cur_src_pc[boundary_points1_indices]
#             src_min_pt = torch.mean(src_bound_pts)

#             boundary_points1_indices, boundary_points2_indices = find_boundary_points(
#                 cur_tgt_pc, tgt_pc[bird_body_indices], threshold=0.01
#             )
#             if icp_instance:
#                 # ========================================================
#                 # redo ICP to align with the src part
#                 converged, rmse, Xt, RTs, t_history = icp(
#                     cur_tgt_pc.unsqueeze(0), cur_src_pc.unsqueeze(0)
#                 )
#                 print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
#                 cur_tgt_pc = Xt.squeeze(0)
#                 # ========================================================

#             # def numpy_to_o3d_pcd(np_array):
#             #     # Create an Open3D point cloud object
#             #     pcd = o3d.geometry.PointCloud()

#             #     # Assign the numpy array to the point cloud's points
#             #     pcd.points = o3d.utility.Vector3dVector(np_array)

#             #     return pcd
#             # source_pcd = numpy_to_o3d_pcd(cur_tgt_pc.cpu().numpy())
#             # target_pcd = numpy_to_o3d_pcd(cur_src_pc.cpu().numpy())
#             # target_pcd.estimate_normals(
#             #     search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
#             # )

#             # init_transformation = np.eye(4)
#             # # Perform point-to-plane ICP
#             # reg_p2p = o3d.pipelines.registration.registration_icp(
#             #     source_pcd, target_pcd, 0.02, init_transformation,
#             #     o3d.pipelines.registration.TransformationEstimationPointToPlane(),
#             #     o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)
#             # )
#             # transformation = reg_p2p.transformation

#             # # Convert the source point cloud using the transformation matrix
#             # source_points = np.asarray(source_pcd.points)
#             # transformed_source_points = (transformation[:3, :3] @ source_points.T).T + transformation[:3, 3]
#             # cur_tgt_pc = torch.tensor(transformed_source_points, device=cur_tgt_pc.device, dtype=cur_tgt_pc.dtype)

#             # transformation = point_to_plane_icp(cur_tgt_pc, cur_src_pc, init_transformation)
#             # transformation = torch.tensor(transformation, device=cur_tgt_pc.device, dtype=cur_tgt_pc.dtype)
#             # cur_tgt_pc = (transformation[:3, :3] @ cur_tgt_pc.T).T + transformation[:3, 3]

#             tgt_bound_pts = cur_tgt_pc[boundary_points1_indices]
#             tgt_min_pt = torch.mean(tgt_bound_pts)

#             # # Create a Plotly scatter plot
#             # fig = go.Figure()

#             # cur_src_pc_plot = cur_src_pc.cpu().numpy()
#             # cur_tgt_pc_plot = cur_tgt_pc.cpu().numpy()
#             # src_bound_pts = src_bound_pts.cpu().numpy()
#             # tgt_bound_pts = tgt_bound_pts.cpu().numpy()
#             # Add source key points
#             # fig.add_trace(
#             #     go.Scatter3d(
#             #         x=cur_src_pc_plot[:, 0],
#             #         y=cur_src_pc_plot[:, 1],
#             #         z=cur_src_pc_plot[:, 2],
#             #         mode="markers",
#             #         marker=dict(size=5, color="blue", opacity=0.5),
#             #         name="Source Full Part",
#             #     )
#             # )

#             # # Add target key points
#             # fig.add_trace(go.Scatter3d(
#             #     x=src_bound_pts[:, 0],
#             #     y=src_bound_pts[:, 1],
#             #     z=src_bound_pts[:, 2],
#             #     mode='markers',
#             #     marker=dict(size=5, color='red', opacity=0.5),
#             #     name='Source Boundary pts'
#             # ))

#             # fig.add_trace(
#             #     go.Scatter3d(
#             #         x=cur_tgt_pc_plot[:, 0],
#             #         y=cur_tgt_pc_plot[:, 1],
#             #         z=cur_tgt_pc_plot[:, 2],
#             #         mode="markers",
#             #         marker=dict(size=5, color="blue", opacity=0.5),
#             #         name="Target Full Part",
#             #     )
#             # )

#             # # Add target key points
#             # fig.add_trace(
#             #     go.Scatter3d(
#             #         x=tgt_bound_pts[:, 0],
#             #         y=tgt_bound_pts[:, 1],
#             #         z=tgt_bound_pts[:, 2],
#             #         mode="markers",
#             #         marker=dict(size=5, color="red", opacity=0.5),
#             #         name="Target Boundary pts",
#             #     )
#             # )

#             # # Set plot layout
#             # fig.update_layout(
#             #     scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
#             #     title="Key Point Mapping",
#             # )

#             # # Show the plot
#             # fig.show()

#             # translate all points of the tgt point cloud by the difference between the two points
#             cur_tgt_pc = cur_tgt_pc - tgt_min_pt + src_min_pt

#         if USE_FPS:
#             # FPS
#             cur_src_kp, src_kp_idx = sfp(
#                 cur_src_pc.unsqueeze(0), K=all_num_keypoints[part_idx]
#             )
#             # cur_src_kp = cur_src_kp.squeeze(0)
#             cur_tgt_kp, tgt_kp_idx = sfp(cur_tgt_pc.unsqueeze(0), K=all_num_keypoints[part_idx])

#         ori_tgt_kp = ori_tgt_pc[tgt_kp_idx[0]]
#         plot_pointcloud(cur_src_kp.squeeze(), log_dir, title=f"Part{part_idx} Src KP")
#         plot_pointcloud(
#             cur_tgt_kp.squeeze(), log_dir, title=f"Part{part_idx} Target KP"
#         )
#         plot_pointcloud(
#             ori_tgt_kp.squeeze(), log_dir, title=f"Part{part_idx} Target KP Original"
#         )

#         PLOT_MATCHING = False
#         # PLOT_MATCHING = True
#         PLOT_NORMAL = False

#         PLOT_BOUNDARY = False

#         cur_src_kp = cur_src_kp.squeeze(0).cpu().numpy()  # Shape: (N, 3)
#         cur_tgt_kp = cur_tgt_kp.squeeze(0).cpu().numpy()
#         ori_tgt_kp = ori_tgt_kp.squeeze(0).cpu().numpy()

#         # Estimate surface normals for source and target key points
#         cur_src_normals = estimate_normals_pytorch3d(cur_src_kp)
#         cur_tgt_normals = estimate_normals_pytorch3d(cur_tgt_kp)

#         # Compute the Euclidean distance matrix
#         euclidean_cost_matrix = np.linalg.norm(
#             cur_src_kp[:, np.newaxis, :] - cur_tgt_kp[np.newaxis, :, :],
#             axis=2,
#         )

#         # Compute the feature distance matrix (e.g., Euclidean distance between normals)
#         if USE_COS:
#             src_norms = np.linalg.norm(cur_src_normals, axis=1)
#             tgt_norms = np.linalg.norm(cur_tgt_normals, axis=1)
#             dot_products = np.einsum('ik,jk->ij', cur_src_normals, cur_tgt_normals)
#             norms_product = np.outer(src_norms, tgt_norms)
#             cos_similarity = dot_products / (norms_product + 1e-8)
#             feature_cost_matrix = 1.0 - cos_similarity

#             cost_matrix = euclidean_cost_matrix * (1 + feature_cost_matrix)
#         else:
#             feature_cost_matrix = np.linalg.norm(
#                 cur_src_normals[:, np.newaxis, :] - cur_tgt_normals[np.newaxis, :, :],
#                 axis=2,
#             )

#             # Combine the two matrices to form the final cost matrix
#             alpha = 0.75  # Weighting factor for combining the distances
#             # alpha = 0.5  # Weighting factor for combining the distances
#             cost_matrix = alpha * euclidean_cost_matrix + (1 - alpha) * feature_cost_matrix

#         # cost_matrix = np.linalg.norm(
#         #     cur_src_kp[:, np.newaxis, :] - cur_tgt_kp[np.newaxis, :, :],
#         #     axis=2,
#         # )

#         # Solve the optimal transport problem using the Hungarian algorithm
#         row_ind, col_ind = linear_sum_assignment(cost_matrix)

#         # The row_ind and col_ind arrays give the optimal 1-to-1 mapping
#         # cur_src_kp[row_ind[i]] is mapped to cur_tgt_kp[col_ind[i]]
#         mapping = list(zip(row_ind, col_ind))

#         new_tgt_kp = np.zeros_like(cur_src_kp)
#         final_ori_tgt_kp = np.zeros_like(cur_src_kp)
#         for src_idx, tgt_idx in mapping:
#             new_tgt_kp[src_idx] = cur_tgt_kp[tgt_idx]
#             final_ori_tgt_kp[src_idx] = ori_tgt_kp[tgt_idx]

#         ori_tgt_kps.append(torch.from_numpy(final_ori_tgt_kp).float())

#         init_kps.append(torch.from_numpy(cur_src_kp).float())
#         kp_indices.append(src_kp_idx)
#         tgt_kps.append(torch.from_numpy(new_tgt_kp).float())
#         tgt_pcs.append(cur_tgt_pc.clone().cpu())

#         # save the key point to disk
#         np.save(os.path.join(log_dir, f"part{part_idx}_src_kp.npy"), cur_src_kp)
#         np.save(os.path.join(log_dir, f"part{part_idx}_tgt_kp.npy"), new_tgt_kp)
#         np.save(os.path.join(log_dir, f"part{part_idx}_tgt_pc.npy"), cur_tgt_pc.cpu().numpy())
#         np.save(
#             os.path.join(log_dir, f"part{part_idx}_tgt_kp_ori.npy"),
#             final_ori_tgt_kp,
#         )
#         # save src_kp_idx
#         # np.save(os.path.join(log_dir, f"part{part_idx}_src_kp_idx.npy"), src_kp_idx)
#         torch.save(src_kp_idx, os.path.join(log_dir, f"part{part_idx}_src_kp_idx.pth"))

#         if PLOT_MATCHING:

#             # Create a Plotly scatter plot
#             fig = go.Figure()

#             # Add source key points
#             fig.add_trace(go.Scatter3d(
#                 x=cur_src_kp[:, 0],
#                 y=cur_src_kp[:, 1],
#                 z=cur_src_kp[:, 2],
#                 mode='markers',
#                 marker=dict(size=5, color='blue'),
#                 name='Source Key Points'
#             ))

#             # Add target key points
#             fig.add_trace(go.Scatter3d(
#                 x=cur_tgt_kp[:, 0],
#                 y=cur_tgt_kp[:, 1],
#                 z=cur_tgt_kp[:, 2],
#                 mode='markers',
#                 marker=dict(size=5, color='red'),
#                 name='Target Key Points'
#             ))

#             # Add vectors showing the mapping
#             for i, j in mapping:
#                 fig.add_trace(go.Scatter3d(
#                     x=[cur_src_kp[i, 0], cur_tgt_kp[j, 0]],
#                     y=[cur_src_kp[i, 1], cur_tgt_kp[j, 1]],
#                     z=[cur_src_kp[i, 2], cur_tgt_kp[j, 2]],
#                     mode='lines',
#                     line=dict(color='green', width=2),
#                     name=f'Mapping {i}->{j}'
#                 ))

#             if PLOT_NORMAL:
#                 # Add surface normals for source key points
#                 normal_length = 0.1  # Adjust the length of the normals as needed
#                 for i in range(cur_src_kp.shape[0]):
#                     fig.add_trace(go.Scatter3d(
#                         x=[cur_src_kp[i, 0], cur_src_kp[i, 0] + normal_length * cur_src_normals[i, 0]],
#                         y=[cur_src_kp[i, 1], cur_src_kp[i, 1] + normal_length * cur_src_normals[i, 1]],
#                         z=[cur_src_kp[i, 2], cur_src_kp[i, 2] + normal_length * cur_src_normals[i, 2]],
#                         mode='lines',
#                         line=dict(color='blue', width=2),
#                         name='Source Normals'
#                     ))

#                 # Add surface normals for target key points
#                 for i in range(cur_tgt_kp.shape[0]):
#                     fig.add_trace(go.Scatter3d(
#                         x=[cur_tgt_kp[i, 0], cur_tgt_kp[i, 0] + normal_length * cur_tgt_normals[i, 0]],
#                         y=[cur_tgt_kp[i, 1], cur_tgt_kp[i, 1] + normal_length * cur_tgt_normals[i, 1]],
#                         z=[cur_tgt_kp[i, 2], cur_tgt_kp[i, 2] + normal_length * cur_tgt_normals[i, 2]],
#                         mode='lines',
#                         line=dict(color='red', width=2),
#                         name='Target Normals'
#                     ))

#             # Set plot layout
#             fig.update_layout(
#                 scene=dict(
#                     xaxis_title='X',
#                     yaxis_title='Y',
#                     zaxis_title='Z'
#                 ),
#                 title='Key Point Mapping'
#             )

#             # Show the plot
#             fig.show()
#         # exit(0)

#     # # Find the boundary points between the two parts
#     # boundary_points1_indices, boundary_points2_indices = find_boundary_points(
#     #     tgt_kps[0], tgt_kps[1], threshold=0.04
#     # )
#     # boundary_indices.append(boundary_points1_indices)
#     # boundary_indices.append(boundary_points2_indices)

#     # # for each point in boundary_points1, find the nearest neighbors of it in boundary_points2
#     # tree = KDTree(tgt_kps[0][boundary_points1_indices].cpu().numpy())
#     # _, p1_bound_inds = tree.query(tgt_kps[1][boundary_points2_indices].cpu().numpy(), k=1)
#     # knn_sp = tgt_kps[0][boundary_points1_indices][p1_bound_inds].cpu().numpy()
#     # # print("@@@", knn_sp.shape)

#     # if PLOT_BOUNDARY:
#     #     # plot the target kps and mark the boundary points red
#     #     # Create a Plotly scatter plot
#     #     fig = go.Figure()

#     #     all_tgt_kps = torch.cat(tgt_kps, 0).cpu().numpy()
#     #     # Add source key points
#     #     fig.add_trace(
#     #         go.Scatter3d(
#     #             x=all_tgt_kps[:, 0],
#     #             y=all_tgt_kps[:, 1],
#     #             z=all_tgt_kps[:, 2],
#     #             mode="markers",
#     #             marker=dict(size=5, color="blue", opacity=0.5),
#     #             name="Source Key Points",
#     #         )
#     #     )

#     #     part_0_boundary_points = tgt_kps[0][boundary_points1_indices].cpu().numpy()

#     #     # Add target key points
#     #     fig.add_trace(go.Scatter3d(
#     #         x=part_0_boundary_points[:, 0],
#     #         y=part_0_boundary_points[:, 1],
#     #         z=part_0_boundary_points[:, 2],
#     #         mode='markers',
#     #         marker=dict(size=5, color='red'),
#     #         name='Target Key Points'
#     #     ))

#     #     part_1_boundary_points = tgt_kps[1][boundary_points2_indices].cpu().numpy()

#     #     # Add target key points
#     #     fig.add_trace(go.Scatter3d(
#     #         x=part_1_boundary_points[:, 0],
#     #         y=part_1_boundary_points[:, 1],
#     #         z=part_1_boundary_points[:, 2],
#     #         mode='markers',
#     #         marker=dict(size=5, color='cyan'),
#     #         name='Target Key Points'
#     #     ))

#     #     # add vectors showing the mapping
#     #     for i in range(len(knn_sp)):
#     #         fig.add_trace(go.Scatter3d
#     #                         (
#     #             x=[part_1_boundary_points[i, 0], knn_sp[i, 0]],
#     #             y=[part_1_boundary_points[i, 1], knn_sp[i, 1]],
#     #             z=[part_1_boundary_points[i, 2], knn_sp[i, 2]],
#     #             mode='lines',
#     #             line=dict(color='green', width=2),
#     #             name=f'Mapping'
#     #         ))

#     #     # Set plot layout
#     #     fig.update_layout(
#     #         scene=dict(
#     #             xaxis_title='X',
#     #             yaxis_title='Y',
#     #             zaxis_title='Z'
#     #         ),
#     #         title='Boundary Points'
#     #     )

#     #     # Show the plot
#     #     fig.show()


# # else:
# #     for part_idx in range(2):
# #         init_kps.append(
# #             torch.from_numpy(
# #                 np.load(os.path.join(log_dir, f"part{part_idx}_src_kp.npy"))
# #             ).float()
# #         )
# #         tgt_kps.append(
# #             torch.from_numpy(
# #                 np.load(os.path.join(log_dir, f"part{part_idx}_tgt_kp.npy"))
# #             ).float()
# #         )
# #         kp_indices.append(
# #             np.load(os.path.join(log_dir, f"part{part_idx}_src_kp_idx.pth"))
# #         )




# if REG_DEFORM_NET:
#     """
#     Regularize the point cloud
#     """

#     # n_iter = 2000
#     n_iter = 300
#     batch_size = 10000
#     robust_c = 0.2
#     # cd_loss_w = 1000.0
#     cd_loss_w = 1.0
#     # cd_loss_w = 2.0
#     rigid_loss_w = 1.0
#     # rigid_loss_w = 1000.0
#     ldas_loss_w = 0.0
#     # cd_weight_decay = 'zeros'
#     src_pc_dir = "/NAS/spa176/papr-retarget/point_clouds/butterfly"

#     # num_nn = 100
#     num_nn_wing = 5
#     # num_nn_body = 250
#     # num_nn_body = 90
#     num_nn_body = 24
#     # num_nn_wing = num_nn_body = 100
#     num_nns = [num_nn_wing, num_nn_wing, num_nn_body]
#     concat_feature = True
#     input_case = 0
#     # input_case = 1
#     # input_case = 2
#     # input_case = 3
#     no_scale = False
#     rotate_normal = True
#     # add key point
#     use_keypoints = True
#     # use_keypoints = False
#     flip_normal = True
#     # num_keypoints = 64
#     # num_keypoints = 96
#     # num_keypoints = 192
#     # num_keypoints_body = 64
#     # num_keypoints_body = 256
#     # num_keypoints_body = 96
#     # kp_knn = 5
#     # kp_knn_body = 5
#     # kp_knn_body = 50
#     # kp_knn = 20
#     kp_knn = 50
#     motion_frame_skip = 200
#     kp_knn_body = 24
#     all_kp_knn = [kp_knn, kp_knn, kp_knn_body]
#     # kp_knn_body = 24
#     # kp_knn_body = 48
#     # kp_knn_body = 256
#     # num_keypoints = 1024
#     # kp_knn = 5
#     total_part_num = 3

#     regularize_kp_only = True
#     double_side_chamfer = True
#     smooth_inp_seq = False
#     smooth_window_size = 35

#     # reg_displacement = True
#     reg_displacement = False

#     pytorch3d_est_normal = True

#     # test_transform_net = True
#     test_transform_net = False

#     # force_smooth = True
#     force_smooth = False

#     force_smooth_full = True
#     smooth_knn = 50

#     # reg_boundary = True
#     reg_boundary = False
#     reg_boundary_test = False

#     transform_L = 0
#     loss_type = "L1_fix"
#     # loss_type = "MSE"
#     # cur_log_dir = os.path.join(log_dir, f'test_deformed_pc_regularized_cdw{cd_loss_w}_wdecay{cd_weight_decay}')
#     if concat_feature:
#         if use_keypoints:
#             cur_log_dir = os.path.join(
#                 log_dir,
#                 # f"transform_wing_kp_{num_keypoints}_body_kp_{num_keypoints_body}_kpnn_{kp_knn}_bodykpnn_{kp_knn_body}_frame_skip_{motion_frame_skip}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nnwing{num_nn_wing}_nnbody{num_nn_body}_concat_{input_case}",
#                 f"{loss_type}_wing_kp_{num_keypoints_left}_body_kp_{num_keypoints_body}_kpnn_{kp_knn}_bodykpnn_{kp_knn_body}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_nnwing{num_nn_wing}_nnbody{num_nn_body}",
#             )
#             # cur_log_dir = os.path.join(
#             #     log_dir,
#             #     f"transform_wing_kp_{num_keypoints}_body_kp_{num_keypoints_body}_kpnn_{kp_knn}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nn{num_nn}_concat_{input_case}",
#             #     # f"test_deformed_pc",
#             # )
#         else:
#             cur_log_dir = os.path.join(
#                 log_dir,
#                 f"transform_L{transform_L}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nn{num_nn}_concat_{input_case}",
#                 # f"test_deformed_pc",
#             )
#         if regularize_kp_only:
#             cur_log_dir += "_kp_only"

#         # if double_side_chamfer:
#         #     cur_log_dir += "_double_side_chamfer"

#         if smooth_inp_seq:
#             cur_log_dir += "_smooth_inp_seq_size" + str(smooth_window_size)

#         if reg_displacement:
#             cur_log_dir += "_regD"

#         if pytorch3d_est_normal:
#             cur_log_dir += "_p3dNorm"

#         if input_case > 0:
#             cur_log_dir += "_inp" + str(input_case)

#         if reg_boundary:
#             cur_log_dir += "_RB"

#     os.makedirs(cur_log_dir, exist_ok=True)

#     # copy current file to the log directory
#     subprocess.run(["cp", __file__, cur_log_dir])

#     # deform_vectors = torch.load(deform_vectors_path)

#     start = 0
#     end = 30001
#     interval = motion_frame_skip

#     src_pcs = [[] for _ in range(total_part_num)]
#     # tgt_pcs = []
#     # for part_idx in range(total_part_num):
#     #     if part_idx == 0:
#     #         # tgt_pcs.append(torch.tensor(tgt_pc[bird_wing_indices]).float())
#     #         tgt_pcs.append(tgt_pc[bird_wing_indices])
#     #     else:
#     #         # tgt_pcs.append(torch.tensor(tgt_pc[bird_body_indices]).float())
#     #         tgt_pcs.append(tgt_pc[bird_body_indices])

#     deformed_pcs = [[] for _ in range(total_part_num)]
#     if use_keypoints:
#         # kp_indices = [[] for _ in range(total_part_num)]
#         # init_kps   = [[] for _ in range(total_part_num)]
#         key_points = [[] for _ in range(total_part_num)]

#     ########## Load the point cloud by parts ##########
#     ########## Optionally add key points ##########
#     scale = 10.0
#     raw_pcs = []
#     for idx in tqdm.tqdm(range(start, end, interval)):
#         src_pc_path = os.path.join(src_pc_dir, f"points_{idx}.npy")
#         cur_src_pc = np.load(src_pc_path)
#         cur_src_pc = cur_src_pc / scale
#         raw_pcs.append(cur_src_pc)

#     if smooth_inp_seq:
#         raw_pcs = smooth_point_cloud(raw_pcs, smooth_window_size)

#     # add input key points for all time steps
#     for idx in tqdm.tqdm(range(len(raw_pcs))):
#         if idx == 0:
#             src_pc = src_pc.cpu()
#         else:
#             src_pc = torch.tensor(raw_pcs[idx]).float()

#         for part_idx in range(total_part_num):
#             cur_src_pc = src_pc[src_part_indices[part_idx]]
#             if use_keypoints:
#                 key_points[part_idx].append(
#                     cur_src_pc[kp_indices[part_idx][0].cpu()].clone()
#                 )
#             else:
#                 src_pcs[part_idx].append(cur_src_pc)
#             # print(f"add key point on device {part_idx}: ", key_points[part_idx][-1].device)
#             # print(f"key point shape {part_idx}: ", key_points[part_idx].shape)

#     for part_idx in range(total_part_num):
#         if use_keypoints:
#             key_points[part_idx] = torch.stack(key_points[part_idx], dim=0)
#             print(f"key_points[{part_idx}]: ", key_points[part_idx].shape)
#         else:
#             src_pcs[part_idx] = torch.stack(src_pcs[part_idx], dim=0)
#             print(f"src_pcs[{part_idx}]: ", src_pcs[part_idx].shape)

#     # num_steps = src_pcs[0].shape[0]
#     if use_keypoints:
#         num_steps = key_points[0].shape[0]
#     else:
#         num_steps = src_pcs[0].shape[0]
#     num_pts = [key_points[part_idx].shape[1] for part_idx in range(total_part_num)]
#     # num_steps, num_pts, _ = src_pcs.shape
#     # if regularize_kp_only:
#     #     for part_idx in range(total_part_num):
#     #         num_pts[part_idx] = key_points[part_idx].shape[1]
#     if reg_displacement:
#         dist_to_nn = [
#             torch.empty(num_pts[part_idx], num_nns[part_idx], 3)
#             for part_idx in range(total_part_num)
#         ]
#     else:
#         dist_to_nn = [
#             torch.empty(num_pts[part_idx], num_nns[part_idx]) for part_idx in range(total_part_num)
#         ]
#     nn_indices = [torch.empty(num_pts[part_idx], num_nns[part_idx]) for part_idx in range(total_part_num)]
#     nn_weights = [torch.empty(num_pts[part_idx], num_nns[part_idx]) for part_idx in range(total_part_num)]
#     nn_init_positions = [torch.empty(num_pts[part_idx], num_nns[part_idx], 3) for part_idx in range(total_part_num)]

#     # dist_to_nn = torch.empty(num_pts, num_nn)
#     # nn_indices = torch.empty(num_pts, num_nn)
#     # nn_weights = torch.empty(num_pts, num_nn)

#     pc_images = []

#     sample_skip = num_steps // 6
#     sample_steps = torch.arange(0, num_steps + 1, sample_skip)

#     if use_keypoints:
#         pred_deformed_pcs = []
#         deformed_pc_weights = []
#         deformed_nn_inds = []
#         pred_normals = []
#         ori_kp_deforms = []
#         vis_ori_kp_deforms = []
#         for part_idx in range(total_part_num):
#             # parameterize the src point cloud
#             # pred_src_pcs, src_pc_weights, src_nn_inds = parametertize_pc(src_pcs[part_idx], init_kps[part_idx][0], kp_knn, step=12000)
#             # parameterize the deformed point cloud
#             # cur_kp_knn = all_kp_knn[part_idx]
#             # pred_deformed_pc, deformed_pc_weight, deformed_nn_ind = parametertize_pc(
#             #     tgt_pcs[part_idx].to(device),
#             #     tgt_kps[part_idx].to(device),
#             #     cur_kp_knn,
#             #     step=35000,
#             # )
#             # pred_deformed_pcs.append(pred_deformed_pc)
#             # deformed_pc_weights.append(deformed_pc_weight.to(device))
#             # deformed_nn_inds.append(deformed_nn_ind)
#             # # deformed_nn_inds = deformed_nn_inds.to(device)
#             # # deformed_pc_weights = deformed_pc_weights.to(device)

#             # # calculate the surface normals at each src_pcs[i] and store the rotation
#             # # normal_rotation_matrices = compute_rotation_matrices_for_batch(
#             # #     src_pcs[sample_steps], num_nn, flip_normal=flip_normal
#             # # )
#             # # deformed_kps = torch.matmul(
#             # #     normal_rotation_matrices[:, kp_indices[0], :, :],
#             # #     init_displacement[:, kp_indices[0], :].detach().cpu().expand(len(sample_steps), -1, -1).unsqueeze(-1),
#             # # ).squeeze(-1) + key_points[sample_steps]
#             # pc_img = plot_pointcloud(
#             #     pred_deformed_pc,
#             #     cur_log_dir,
#             #     title=f"Pred PC Part {part_idx}",
#             # )

#             normal_rotation_matrices, pred_normal = (
#                 compute_rotation_matrices_for_batch(
#                     key_points[part_idx], 10, flip_normal=flip_normal
#                 )
#             )
#             pred_normals.append(pred_normal)
#             ori_kp_deform = torch.matmul(
#                 normal_rotation_matrices,
#                 (tgt_kps[part_idx] - init_kps[part_idx])
#                 .expand(len(normal_rotation_matrices), -1, -1)
#                 .unsqueeze(-1)
#                 .cpu(),
#             ).squeeze(-1)
#             ori_kp_deforms.append(ori_kp_deform)
#             # for visualization
#             vis_ori_kp_deforms.append(ori_kp_deform[sample_steps])

#             # deformed_kps = torch.matmul(
#             #     normal_rotation_matrices,
#             #     init_displacement[:, kp_indices[0], :].detach().cpu().expand(len(sample_steps), -1, -1).unsqueeze(-1),
#             # ).squeeze(-1) + key_points[sample_steps]

#             # deformed_src_pcs = []
#             # for deformed_kp in deformed_kps:
#             #     deformed_src_pcs.append(
#             #         torch.einsum('ij,ijk->ik', deformed_pc_weights, deformed_kp[deformed_nn_inds])
#             #     )
#             # deformed_src_pcs = torch.stack(deformed_src_pcs, dim=0)

#             # apply the rotation matrix to init_displacement given the normal rotation matrices and add to scr_pcs
#             # deformed_src_pcs = torch.matmul(
#             #     normal_rotation_matrices,
#             #     init_displacement.detach().cpu().expand(len(sample_steps), -1, -1).unsqueeze(-1),
#             # ).squeeze(-1) + src_pcs[sample_steps]
#             # pc_img = plot_pointcloud(deformed_src_pcs, cur_log_dir, title=f"Deformed Point Cloud Iter {0}")
#             # print(a)
#     else:
#         pred_normals = []
#         ori_kp_deforms = []
#         vis_ori_kp_deforms = []
#         total_deformed_pc = []
#         for part_idx in range(total_part_num):
#             if part_idx == 0:
#                 normal_nn = 100
#             else:
#                 normal_nn = 100

#             normal_rotation_matrices, pred_normal = compute_rotation_matrices_for_batch(
#                 src_pcs[part_idx][sample_steps], normal_nn, flip_normal=flip_normal
#             )
#             pred_normals.append(pred_normal)
#             ori_pc_deform = torch.matmul(
#                 normal_rotation_matrices,
#                 (deformed_src_pc_start[part_idx].cpu() - src_pcs[part_idx][0])
#                 .expand(len(normal_rotation_matrices), -1, -1)
#                 .unsqueeze(-1),
#             ).squeeze(-1)
#             total_deformed_pc.append(ori_pc_deform + src_pcs[part_idx][sample_steps])
#         total_deformed_pc = torch.cat(total_deformed_pc, dim=1)
#         plot_pointcloud(
#             total_deformed_pc,
#             cur_log_dir,
#             title=f"Initial Deformed Point Cloud",
#         )
#         print(a)

#     if concat_feature:
#         if input_case == 0 or input_case == 2:
#             transform_net = AffineTransformationNet(9, L=transform_L).to(device)
#         elif input_case == 1 or input_case == 3:
#             transform_net = AffineTransformationNet(9, L=transform_L, non_pe_dim=1).to(device)
#         else:
#             transform_net = AffineTransformationNet(6, L=transform_L).to(device)
#     else:
#         transform_net = AffineTransformationNet(3, L=transform_L).to(device)

#     def visualize_time_steps(pc_images, cur_log_dir, num_steps, cur_iter):
#         with torch.no_grad():
#             # sample_steps = torch.arange(0, num_steps + 1, 5)

#             total_deformed_pc = []

#             for part_idx in range(total_part_num):

#                 if use_keypoints:
#                     if input_case == 2 or input_case == 3:
#                         pc_inp = key_points[part_idx][0:1].expand(len(sample_steps), -1, -1)
#                         base_pc = key_points[part_idx][sample_steps]
#                     else:
#                         pc_inp = base_pc = key_points[part_idx][sample_steps]

#                     vis_base_displacement = vis_ori_kp_deforms[part_idx]
#                 else:
#                     pc_inp = src_pcs[sample_steps]
#                     vis_base_displacement = init_displacement

#                 if concat_feature:
#                     if input_case == 0 or input_case == 2:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     pc_inp.to(device),
#                                     pred_normals[part_idx][0:1]
#                                     .expand(len(sample_steps), -1, -1)
#                                     .to(device),
#                                     pred_normals[part_idx][sample_steps].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                     elif input_case == 1 or input_case == 3:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     pc_inp.to(device),
#                                     pred_normals[part_idx][0:1]
#                                     .expand(len(sample_steps), -1, -1)
#                                     .to(device),
#                                     pred_normals[part_idx][sample_steps].to(device),
#                                 ],
#                                 dim=-1,
#                             ),
#                             non_pe=sample_steps.unsqueeze(-1)
#                             .unsqueeze(-1)
#                             .expand(-1, all_num_keypoints[part_idx], -1)
#                             .float()
#                             .to(device),
#                         )
#                     else:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     src_pcs[sample_steps].to(device),
#                                     init_displacement.expand(len(sample_steps), -1, -1),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                 else:
#                     scale, quaternion = transform_net(pc_inp.to(device))
#                 rotation_matrix = quaternion_to_matrix(quaternion)
#                 deformed_src_pcs = apply_transformation(
#                     vis_base_displacement.to(device),
#                     scale,
#                     rotation_matrix,
#                     no_scale=no_scale,
#                 ) + base_pc.to(device)
#                 # deformed_src_pcs = vis_base_displacement.to(device) + pc_inp.to(device)
#                 if use_keypoints and not regularize_kp_only:
#                     deformed_pcs = []
#                     for deformed_kp in deformed_src_pcs:
#                         deformed_pcs.append(
#                             torch.einsum(
#                                 "ij,ijk->ik",
#                                 deformed_pc_weights[part_idx],
#                                 deformed_kp[deformed_nn_inds[part_idx]],
#                             )
#                         )
#                     total_deformed_pc.append(torch.stack(deformed_pcs, dim=0))
#                 elif regularize_kp_only:
#                     total_deformed_pc.append(deformed_src_pcs)
#             total_deformed_pc = torch.cat(total_deformed_pc, dim=1)
#             pc_img = plot_pointcloud(
#                 total_deformed_pc,
#                 cur_log_dir,
#                 title=f"Deformed Point Cloud Iter {cur_iter}",
#             )
#             pc_images.append(pc_img)

#     def visualize_kps(cur_log_dir, num_steps):
#         with torch.no_grad():
#             # sample_steps = torch.arange(0, num_steps + 1, 5)
#             total_deformed_kps = []
#             frames = []
#             inp_all_frames = []
#             inp_kps = []

#             for part_idx in range(total_part_num):
#                 part_frames = []
#                 inp_frames = []

#                 if use_keypoints:
#                     if input_case == 2 or input_case == 3:
#                         pc_inp = key_points[part_idx][0:1].expand(num_steps, -1, -1)
#                         base_pc = key_points[part_idx]
#                     else:
#                         pc_inp = base_pc = key_points[part_idx]

#                     vis_base_displacement = ori_kp_deforms[part_idx]
#                 else:
#                     pc_inp = src_pcs[sample_steps]
#                     vis_base_displacement = init_displacement

#                 if concat_feature:
#                     if input_case == 0 or input_case == 2:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     pc_inp.to(device),
#                                     pred_normals[part_idx][0:1]
#                                     .expand(num_steps, -1, -1)
#                                     .to(device),
#                                     pred_normals[part_idx].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                     elif input_case == 1 or input_case == 3:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     pc_inp.to(device),
#                                     pred_normals[part_idx][0:1]
#                                     .expand(num_steps, -1, -1)
#                                     .to(device),
#                                     pred_normals[part_idx].to(device),
#                                 ],
#                                 dim=-1,
#                             ),
#                             non_pe=torch.arange(num_steps)
#                             .unsqueeze(-1)
#                             .unsqueeze(-1)
#                             .expand(-1, all_num_keypoints[part_idx], -1)
#                             .float()
#                             .to(device),
#                         )
#                     else:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     src_pcs[sample_steps].to(device),
#                                     init_displacement.expand(len(sample_steps), -1, -1),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                 else:
#                     scale, quaternion = transform_net(pc_inp.to(device))
#                 rotation_matrix = quaternion_to_matrix(quaternion)
#                 deformed_src_pcs = apply_transformation(
#                     vis_base_displacement.to(device),
#                     scale,
#                     rotation_matrix,
#                     no_scale=no_scale,
#                 ) + base_pc.to(device)
#                 total_deformed_kps.append(deformed_src_pcs)
#                 inp_kps.append(base_pc)

#                 for t in range(num_steps):
#                     # pc_img = plot_pointcloud(
#                     #     deformed_src_pcs[t],
#                     #     None,
#                     #     title=f"Deformed KP Part {part_idx} Frame {t}",
#                     # )
#                     pc_img = plot_pointcloud(
#                         deformed_src_pcs[t],
#                         None,
#                         title=f"Deformed KP Part {part_idx} Frame {t}",
#                         extra_points=base_pc[t],
#                     )
#                     part_frames.append(pc_img)
#                     # inp_pc_img = plot_pointcloud(
#                     #     base_pc[t],
#                     #     None,
#                     #     title=f"Input KP Part {part_idx} Frame {t}",
#                     # )
#                     # inp_frames.append(inp_pc_img)
#                 # imageio.mimsave(
#                 #     os.path.join(cur_log_dir, f"deformed_kp_part_{part_idx}.mp4"), part_frames, fps=10
#                 # )
#                 imageio.mimsave(
#                     os.path.join(cur_log_dir, f"kp_part_{part_idx}.mp4"), part_frames, fps=10
#                 )
#                 # imageio.mimsave(
#                 #     os.path.join(cur_log_dir, f"inp_kp_part_{part_idx}.mp4"), inp_frames, fps=10
#                 # )

#             total_deformed_kps = torch.cat(total_deformed_kps, dim=1)
#             inp_kps = torch.cat(inp_kps, dim=1)
#             for t in range(num_steps):
#                 # pc_img = plot_pointcloud(
#                 #     total_deformed_kps[t],
#                 #     None,
#                 #     title=f"Deformed KP Frame {t}",
#                 # )
#                 pc_img = plot_pointcloud(
#                     total_deformed_kps[t],
#                     None,
#                     title=f"Deformed KP Frame {t}",
#                     extra_points=inp_kps[t],
#                 )
#                 frames.append(pc_img)
#             # imageio.mimsave(
#             #     os.path.join(cur_log_dir, f"deformed_kp_all.mp4"), frames, fps=10
#             # )
#             imageio.mimsave(
#                 os.path.join(cur_log_dir, f"kp_all.mp4"), frames, fps=10
#             )

#             # for t in range(num_steps):
#             #     inp_pc_img = plot_pointcloud(
#             #         inp_kps[t],
#             #         None,
#             #         title=f"Input KP Frame {t}",
#             #     )
#             #     inp_all_frames.append(inp_pc_img)
#             # imageio.mimsave(
#             #     os.path.join(cur_log_dir, f"inp_kp_all.mp4"), inp_all_frames, fps=10
#             # )

#     def save_all_deformed_pcs():
#         with torch.no_grad():
#             total_deformed_pc = []
#             cur_order = [1, 0] if reg_boundary_test else range(total_part_num)

#             # for part_idx in range(total_part_num):
#             for part_idx in cur_order:
#                 if use_keypoints:
#                     if input_case == 2 or input_case == 3:
#                         pc_inp = key_points[part_idx][0:1].expand(num_steps, -1, -1)
#                         base_pc = key_points[part_idx]
#                     else:
#                         pc_inp = base_pc = key_points[part_idx]
#                     vis_base_displacement = ori_kp_deforms[part_idx]
#                 else:
#                     pc_inp = src_pcs[sample_steps]
#                     vis_base_displacement = init_displacement

#                 if concat_feature:
#                     if input_case == 0 or input_case == 2:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     pc_inp.to(device),
#                                     pred_normals[part_idx][0:1]
#                                     .expand(num_steps, -1, -1)
#                                     .to(device),
#                                     pred_normals[part_idx].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                     elif input_case == 1 or input_case == 3:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     pc_inp.to(device),
#                                     pred_normals[part_idx][0:1]
#                                     .expand(num_steps, -1, -1)
#                                     .to(device),
#                                     pred_normals[part_idx].to(device),
#                                 ],
#                                 dim=-1,
#                             ),
#                             non_pe=torch.arange(num_steps)
#                             .unsqueeze(-1)
#                             .unsqueeze(-1)
#                             .expand(-1, all_num_keypoints[part_idx], -1)
#                             .float()
#                             .to(device),
#                         )
#                     else:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     src_pcs[sample_steps].to(device),
#                                     init_displacement.expand(len(sample_steps), -1, -1),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                 else:
#                     scale, quaternion = transform_net(pc_inp.to(device))
#                 rotation_matrix = quaternion_to_matrix(quaternion)
#                 deformed_src_pcs = apply_transformation(
#                     vis_base_displacement.to(device),
#                     scale,
#                     rotation_matrix,
#                     no_scale=no_scale,
#                 ) + base_pc.to(device)
#                 # deformed_src_pcs = vis_base_displacement.to(device) + pc_inp.to(device)
#                 if force_smooth:
#                     for time_step in range(1, num_steps):
#                         avg_displacement = (
#                             deformed_src_pcs[time_step]
#                             .view(num_pts[part_idx], 1, 3)
#                             .expand(num_pts[part_idx], num_nns[part_idx], 3)
#                             .gather(
#                                 0,
#                                 nn_indices[part_idx][:, :num_nns[part_idx]]
#                                 .to(device)
#                                 .unsqueeze(-1)
#                                 .expand(num_pts[part_idx], num_nns[part_idx], 3),
#                             )
#                             - nn_init_positions[part_idx]
#                             .view(num_pts[part_idx], 1, 3)
#                             .expand(num_pts[part_idx], num_pts[part_idx], 3)
#                             .gather(
#                                 0,
#                                 nn_indices[part_idx][:, :num_nns[part_idx]]
#                                 .to(device)
#                                 .unsqueeze(-1)
#                                 .expand(num_pts[part_idx], num_nns[part_idx], 3),
#                             )
#                         ).mean(dim=1)
#                         deformed_src_pcs[time_step] = (
#                             avg_displacement + nn_init_positions[part_idx]
#                         )

#                 if reg_boundary_test:
#                     # calculate the initial mean displacement between the sets of two boundary points
#                     # mean_bound_displacement = torch.mean(
#                     #     tgt_kps[0][boundary_indices[0]], dim=0) - torch.mean(tgt_kps[1][boundary_indices[1]], dim=0
#                     # )
#                     if part_idx == 1:
#                         anchor_kps = deformed_src_pcs.clone()
#                     else:
#                         cur_mean_bound_displacement = torch.mean(
#                             deformed_src_pcs[:, boundary_indices[0]], dim=1) - torch.mean(
#                             - anchor_kps[:, boundary_indices[1]],
#                             dim=1,
#                         ) # shape (num_steps, 3)
#                         # print(f"!!!!cur_mean_bound_displacement: {cur_mean_bound_displacement.shape}")
#                         # print("mean displacement", mean_bound_displacement)
#                         # print("$$$$$")
#                         # print("cur_mean_bound_displacement", cur_mean_bound_displacement[:5])
#                         # deformed_src_pcs = deformed_src_pcs + (mean_bound_displacement.to(device) - cur_mean_bound_displacement).unsqueeze(1).expand(-1, num_pts[part_idx], -1)
#                         deformed_src_pcs = deformed_src_pcs - (cur_mean_bound_displacement[0:1, :] - cur_mean_bound_displacement).unsqueeze(1).expand(-1, num_pts[part_idx], -1)

#                 if use_keypoints:
#                     deformed_pcs = []
#                     for deformed_kp in deformed_src_pcs:
#                         deformed_pcs.append(
#                             torch.einsum(
#                                 "ij,ijk->ik",
#                                 deformed_pc_weights[part_idx],
#                                 deformed_kp[deformed_nn_inds[part_idx]],
#                             )
#                         )
#                     total_deformed_pc.append(torch.stack(deformed_pcs, dim=0))

#                 if force_smooth_full:
#                     base_pc = total_deformed_pc[-1][0]
#                     cur_num_pts = base_pc.shape[0]

#                     nn_indices = torch.empty(cur_num_pts, smooth_knn, dtype=torch.int64, device=device)
#                     for pt_idx in range(cur_num_pts):
#                         # find the distance from the point at index i to all others points
#                         displacement_to_all_pts = (
#                             base_pc[pt_idx : pt_idx + 1, :].expand(cur_num_pts, 3)
#                             - base_pc
#                         )
#                         _, inds = torch.topk(
#                             displacement_to_all_pts.pow(2).sum(dim=1),
#                             smooth_knn + 1,
#                             largest=False,
#                             sorted=True,
#                         )
#                         nn_indices[pt_idx, :] = inds[1:].type(torch.int64)

#                     for time_step in range(1, num_steps):
#                         avg_displacement = (
#                             total_deformed_pc[-1][time_step]
#                             .view(cur_num_pts, 1, 3)
#                             .expand(cur_num_pts, cur_num_pts, 3)
#                             .gather(
#                                 0,
#                                 nn_indices
#                                 .to(device)
#                                 .unsqueeze(-1)
#                                 .expand(cur_num_pts, smooth_knn, 3),
#                             )
#                             - base_pc.view(cur_num_pts, 1, 3)
#                             .expand(cur_num_pts, cur_num_pts, 3)
#                             .gather(
#                                 0,
#                                 nn_indices
#                                 .to(device)
#                                 .unsqueeze(-1)
#                                 .expand(cur_num_pts, smooth_knn, 3),
#                             )
#                         ).mean(dim=1)
#                         total_deformed_pc[-1][time_step] = avg_displacement + base_pc

#             if reg_boundary_test:
#                 # swap the two parts in total_deformed_pc
#                 total_deformed_pc[0], total_deformed_pc[1] = total_deformed_pc[1], total_deformed_pc[0]
#             total_deformed_pc = torch.cat(total_deformed_pc, dim=1) # shape (num_steps, num_pts, 3)

#         if force_smooth:
#             torch.save(total_deformed_pc, os.path.join(cur_log_dir, "total_deformed_pc_smooth.pth"))
#         elif force_smooth_full:
#             if smooth_knn != 100:
#                 torch.save(total_deformed_pc, os.path.join(cur_log_dir, f"total_deformed_pc_smooth_full_{smooth_knn}.pth"))
#             else:
#                 torch.save(total_deformed_pc, os.path.join(cur_log_dir, "total_deformed_pc_smooth_full.pth"))
#         # elif reg_boundary:
#         #     torch.save(total_deformed_pc, os.path.join(cur_log_dir, "total_deformed_pc_rb.pth"))
#         else:
#             # save the total deformed point cloud as a pytorch tensor
#             torch.save(total_deformed_pc, os.path.join(cur_log_dir, "total_deformed_pc.pth"))
#         # save the initial deformed point cloud at start state
#         torch.save(
#             torch.cat(pred_deformed_pcs, dim=0),
#             os.path.join(cur_log_dir, "deformed_src_pc_start.pth"),
#         )

#     def save_deformed_kps(shift=False):
#         with torch.no_grad():
#             for part_idx in range(total_part_num):
#                 if use_keypoints:
#                     if input_case == 2 or input_case == 3:
#                         pc_inp = key_points[part_idx][0:1].expand(num_steps, -1, -1)
#                         base_pc = key_points[part_idx]
#                     else:
#                         pc_inp = base_pc = key_points[part_idx]
#                     vis_base_displacement = ori_kp_deforms[part_idx]
#                 else:
#                     pc_inp = src_pcs[sample_steps]
#                     vis_base_displacement = init_displacement

#                 if concat_feature:
#                     if input_case == 0 or input_case == 2:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     pc_inp.to(device),
#                                     pred_normals[part_idx][0:1]
#                                     .expand(num_steps, -1, -1)
#                                     .to(device),
#                                     pred_normals[part_idx].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                     elif input_case == 1 or input_case == 3:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     pc_inp.to(device),
#                                     pred_normals[part_idx][0:1]
#                                     .expand(num_steps, -1, -1)
#                                     .to(device),
#                                     pred_normals[part_idx].to(device),
#                                 ],
#                                 dim=-1,
#                             ),
#                             non_pe=torch.arange(num_steps)
#                             .unsqueeze(-1)
#                             .unsqueeze(-1)
#                             .expand(-1, all_num_keypoints[part_idx], -1)
#                             .float()
#                             .to(device),
#                         )
#                     else:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     src_pcs[sample_steps].to(device),
#                                     init_displacement.expand(len(sample_steps), -1, -1),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                 else:
#                     scale, quaternion = transform_net(pc_inp.to(device))
#                 rotation_matrix = quaternion_to_matrix(quaternion)
#                 deformed_src_pcs = apply_transformation(
#                     vis_base_displacement.to(device),
#                     scale,
#                     rotation_matrix,
#                     no_scale=no_scale,
#                 ) + base_pc.to(device)

#                 torch.save(
#                     deformed_src_pcs, os.path.join(cur_log_dir, f"total_deformed_kps_p{part_idx}.pth")
#                 )

#                 if shift:
#                     deformed_displacements = deformed_src_pcs - deformed_src_pcs[0:1]
#                     deformed_src_pcs = deformed_displacements + ori_tgt_kps[part_idx].unsqueeze(0).expand(deformed_src_pcs.shape[0], -1, -1).to(device)
#                     # save the total deformed point cloud as a pytorch tensor
#                     torch.save(
#                         deformed_src_pcs,
#                         os.path.join(
#                             cur_log_dir, f"total_deformed_kps_p{part_idx}_shifted.pth"
#                         ),
#                     )
#                 # if shift:
#                 #     # shift all the deformed_src_pcs
#                 #     deformed_displacements = deformed_src_pcs - deformed_src_pcs[0:1]
#                 #     deformed_src_pcs = deformed_displacements + ori_tgt_kps[part_idx].unsqueeze(0).expand(deformed_src_pcs.shape[0], -1, -1).to(device)
#                 #     # save the total deformed point cloud as a pytorch tensor
#                 #     torch.save(
#                 #         deformed_src_pcs,
#                 #         os.path.join(
#                 #             cur_log_dir, f"total_deformed_kps_p{part_idx}_shifted.pth"
#                 #         ),
#                 #     )
#                 # else:
#                 #     # save the total deformed point cloud as a pytorch tensor
#                 #     torch.save(
#                 #         deformed_src_pcs, os.path.join(cur_log_dir, f"total_deformed_kps_p{part_idx}.pth")
#                 #     )

#     # if test_transform_net:
#     #     transform_net.load_state_dict(
#     #         torch.load(os.path.join(cur_log_dir, "transform_net.pth"))
#     #     )
#     #     # visualize_time_steps(pc_images, cur_log_dir, num_steps, n_iter + 1)
#     #     # save_all_deformed_pcs()
#     #     save_deformed_kps()
#     #     exit(0)

#     # # ===== old optimizer of the deform net =====
#     # # optimizer = torch.optim.Adam(reg_net.parameters(), lr=0.0005)
#     # optimizer = torch.optim.Adam(transform_net.parameters(), lr=0.0005)

#     if use_keypoints:
#         if regularize_kp_only:
#             base_dist_pcs = tgt_kps
#             base_displacement = ori_kp_deforms
#         else:
#             base_dist_pcs = pred_deformed_pcs
#             base_displacement = ori_kp_deforms
#     else:
#         base_dist_pcs = deformed_src_pc
#         base_displacement = init_displacement

#     for part_idx in range(total_part_num):
#         for pt_idx in range(num_pts[part_idx]):
#             # find the distance from the point at index i to all others points
#             displacement_to_all_pts = (
#                 base_dist_pcs[part_idx][pt_idx : pt_idx + 1, :].expand(
#                     num_pts[part_idx], 3
#                 )
#                 - base_dist_pcs[part_idx]
#             )

#             # dist_to_all_pts = (
#             #     (
#             #         base_dist_pcs[part_idx][pt_idx : pt_idx + 1, :].expand(
#             #             num_pts[part_idx], 3
#             #         )
#             #         - base_dist_pcs[part_idx]
#             #     )
#             #     .pow(2)
#             #     .sum(dim=1)
#             # )
#             vals, inds = torch.topk(
#                 displacement_to_all_pts.pow(2).sum(dim=1), num_nns[part_idx] + 1, largest=False, sorted=True
#             )
#             if reg_displacement:
#                 dist_to_nn[part_idx][pt_idx, :, :], nn_indices[part_idx][pt_idx, :] = (
#                     displacement_to_all_pts[inds[1:]],
#                     inds[1:],
#                 )
#             else:
#                 dist_to_nn[part_idx][pt_idx, :], nn_indices[part_idx][pt_idx, :] = (
#                     vals[1:],
#                     inds[1:],
#                 )
#             # gaussian_weight = 0
#             # nn_weights[pt_idx, :] = (
#             #     (-dist_to_nn[pt_idx, :] * gaussian_weight).exp().detach()
#             # )

#         # calculate the current sum of distances from each point to its nearest neighbors
#         nn_indices[part_idx] = nn_indices[part_idx].type(torch.int64)
#         # dist_to_nn = (dist_to_nn * nn_weights).detach()
#         dist_to_nn[part_idx] = dist_to_nn[part_idx].detach()
#         # nn_init_positions = deformed_src_pc.detach().clone().to(device)
#         nn_init_positions[part_idx] = base_dist_pcs[part_idx].to(device)

#     # find the original distances between boundary points
#     # num_bound_pts = [
#     #     len(boundary_indices[part_idx]) for part_idx in range(total_part_num)
#     # ]
#     # find the the l2 distance bewtwen each pair of boundary points
#     # boundary_dists = torch.norm(
#     #     base_dist_pcs[0][boundary_indices[0]].unsqueeze(1).expand(
#     #         num_bound_pts[0], num_bound_pts[1], 3
#     #     ) - base_dist_pcs[1][boundary_indices[1]].unsqueeze(0).expand(
#     #         num_bound_pts[0], num_bound_pts[1], 3
#     #     ),
#     #     dim=2,
#     # )
#     # boundary_dists = torch.norm(
#     #     base_dist_pcs[0][boundary_indices[0]][p1_bound_inds] - base_dist_pcs[1][boundary_indices[1]],
#     #     dim=-1,
#     # )

#     if test_transform_net:
#         if reg_boundary_test:
#             # load the experiment log dir without "_RB"
#             transform_net.load_state_dict(
#                 torch.load(os.path.join(cur_log_dir[:-3], "transform_net.pth"))
#             )
#         else:
#             transform_net.load_state_dict(
#                 torch.load(os.path.join(cur_log_dir, "transform_net.pth"))
#             )
#         # visualize_time_steps(pc_images, cur_log_dir, num_steps, n_iter + 1)
#         # visualize_kps(cur_log_dir, num_steps)
#         # save_all_deformed_pcs()
#         # save_deformed_kps(shift=True)
#         save_deformed_kps(shift=False)
#         exit(0)

#     # ===== old optimizer of the deform net =====
#     # optimizer = torch.optim.Adam(reg_net.parameters(), lr=0.0005)
#     optimizer = torch.optim.Adam(transform_net.parameters(), lr=0.0005)

#     bird_body_indices = np.setdiff1d(np.arange(len(tgt_pc)), bird_wing_indices)
#     progress_bar = tqdm.tqdm(range(n_iter), desc="Training")

#     for i in progress_bar:
#         # Compute the target point cloud
#         # deform_vecs = reg_net(src_pcs.to(device))
#         # deformed_src_pcs = src_pcs.to(device) + deform_vecs

#         # total_cd_loss = chamfer_distance(deformed_src_pcs[0:1], tgt_pc.unsqueeze(0))[0]
#         if i % 20 == 0:
#             visualize_time_steps(pc_images, cur_log_dir, num_steps, i)
#             # print(a)
#         if i % 100 == 0 or i == 50:
#             torch.save(
#                 transform_net.state_dict(),
#                 os.path.join(cur_log_dir, "transform_net.pth"),
#             )
#             imageio.mimsave(
#                 os.path.join(cur_log_dir, "train_deformed_pc.mp4"), pc_images, fps=10
#             )
#             # save_all_deformed_pcs()
#             save_deformed_kps()
#             if i >= 100:
#                 visualize_kps(cur_log_dir, num_steps)

#         # total_reg_loss = 0
#         for j in range(1, num_steps):
#             # total_reg_loss = 0
#             # total_cd_loss = 0
#             # deform_vecs = reg_net(src_pcs.to(device))
#             # try reusing the first src_pc as the input to all the deformations
#             # deform_vecs = reg_net(src_pcs[0:1].to(device))
#             # deformed_src_pcs = src_pcs.to(device) + deform_vecs

#             # ===== adding transformation net =====
#             # scale, quaternion = transform_net(src_pcs.to(device))
#             # rotation_matrix = quaternion_to_matrix(quaternion)
#             # deformed_src_pcs = apply_transformation(
#             #     init_displacement, scale, rotation_matrix
#             # ) + src_pcs.to(device)
#             # total_cd_loss = chamfer_distance(deformed_src_pcs[0:1], tgt_pc.unsqueeze(0))[0]

#             if use_keypoints:
#                 inp_pcs = key_points
#             else:
#                 inp_pcs = src_pcs

#             for part_idx in range(total_part_num):
#                 num_nn = num_nns[part_idx]
#                 total_reg_loss = 0
#                 if regularize_kp_only:
#                     cur_tgt_pc = tgt_pcs[part_idx]
#                 # else:
#                 #     if part_idx == 0:
#                 #         cur_tgt_pc = tgt_pc[bird_wing_indices]
#                 #     else:
#                 #         cur_tgt_pc = tgt_pc[bird_body_indices]

#                 if concat_feature:
#                     if input_case == 0 or input_case == 2:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     inp_pcs[part_idx][0].to(device),
#                                     pred_normals[part_idx][0].to(device),
#                                     pred_normals[part_idx][0].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                     elif input_case == 1 or input_case == 3:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     inp_pcs[part_idx][0].to(device),
#                                     pred_normals[part_idx][0].to(device),
#                                     pred_normals[part_idx][0].to(device),
#                                 ],
#                                 dim=-1,
#                             ),
#                             non_pe=torch.tensor([0])
#                             .unsqueeze(-1)
#                             .expand(all_num_keypoints[part_idx], -1)
#                             .float()
#                             .to(device),
#                         )
#                     else:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     src_pcs[part_idx][0].to(device),
#                                     (
#                                         deformed_src_pc_start[part_idx]
#                                         - src_pcs[part_idx][0].to(device)
#                                     )[0].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                 else:
#                     scale, quaternion = transform_net(inp_pcs[0].to(device))
#                 # concate src_pcs[0] and init_displacement[0] along the last dimension

#                 rotation_matrix = quaternion_to_matrix(quaternion)
#                 deformed_src_pcs = apply_transformation(
#                     base_displacement[part_idx][0].to(device),
#                     scale,
#                     rotation_matrix,
#                     no_scale=no_scale,
#                 ) + inp_pcs[part_idx][0].to(device)
#                 # print(f"Loss: {loss.item()}, Reg Loss: {total_reg_loss.item()}, CD Loss: {total_cd_loss.item()}")
#                 if use_keypoints:
#                     if regularize_kp_only:
#                         final_deformed_src_pcs = deformed_src_pcs
#                     else:
#                         final_deformed_src_pcs = torch.einsum(
#                             "ij,ijk->ik",
#                             deformed_pc_weights[part_idx],
#                             deformed_src_pcs[deformed_nn_inds[part_idx]],
#                         )
#                 else:
#                     final_deformed_src_pcs = deformed_src_pcs
#                 # ===== vanilla loss without keypoint =====
#                 # rotation_matrix = quaternion_to_matrix(quaternion)
#                 # deformed_src_pcs = apply_transformation(
#                 #     init_displacement[0], scale, rotation_matrix, no_scale=no_scale
#                 # ) + inp_pcs[0].to(device)

#                 # if regularize_kp_only and not double_side_chamfer:
#                 #     total_cd_loss = chamfer_distance(
#                 #         final_deformed_src_pcs.unsqueeze(0), cur_tgt_pc.unsqueeze(0), single_directional=True
#                 #     )[0]
#                 # else:
#                 total_cd_loss = chamfer_distance(
#                     final_deformed_src_pcs.unsqueeze(0), cur_tgt_pc.unsqueeze(0).to(device)
#                 )[0]
#                 # ===== comment out the chamfer distance loss =====
#                 # cd_loss = chamfer_distance(deformed_src_pcs[j:j+1], tgt_pc.unsqueeze(0))[0]
#                 # if cd_weight_decay == 'exp':
#                 #     cd_weight = np.exp(-0.1 * j)
#                 # elif cd_weight_decay == 'linear':
#                 #     cd_weight = (num_steps - j) / num_steps
#                 # elif cd_weight_decay == 'zeros':
#                 #     cd_weight = 0
#                 # total_cd_loss += cd_loss * cd_weight
#                 # ===== comment out the chamfer distance loss =====
#                 if concat_feature:
#                     if input_case == 0:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     inp_pcs[part_idx][j].to(device),
#                                     pred_normals[part_idx][0].to(device),
#                                     pred_normals[part_idx][j].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                     elif input_case == 1:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     inp_pcs[part_idx][j].to(device),
#                                     pred_normals[part_idx][0].to(device),
#                                     pred_normals[part_idx][j].to(device),
#                                 ],
#                                 dim=-1,
#                             ),
#                             non_pe=torch.tensor([j])
#                             .unsqueeze(-1)
#                             .expand(all_num_keypoints[part_idx], -1)
#                             .float()
#                             .to(device),
#                         )
#                     elif input_case == 2:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     inp_pcs[part_idx][0].to(device),
#                                     pred_normals[part_idx][0].to(device),
#                                     pred_normals[part_idx][j].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                     elif input_case == 3:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     inp_pcs[part_idx][0].to(device),
#                                     pred_normals[part_idx][0].to(device),
#                                     pred_normals[part_idx][j].to(device),
#                                 ],
#                                 dim=-1,
#                             ),
#                             non_pe=torch.tensor([j])
#                             .unsqueeze(-1)
#                             .expand(all_num_keypoints[part_idx], -1)
#                             .float()
#                             .to(device),
#                         )
#                     else:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     src_pcs[part_idx][j].to(device),
#                                     (
#                                         deformed_src_pc_start[part_idx]
#                                         - src_pcs[part_idx][0].to(device)
#                                     )[0].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                 else:
#                     scale, quaternion = transform_net(inp_pcs[j].to(device))
#                 # scale, quaternion = transform_net(src_pcs[j].to(device))
#                 # scale, quaternion = transform_net(torch.cat([src_pcs[j].to(device), init_displacement[0]], dim=-1))

#                 if use_keypoints:
#                     displacement = base_displacement[part_idx][j]
#                 else:
#                     displacement = init_displacement[0]

#                 rotation_matrix = quaternion_to_matrix(quaternion)
#                 deformed_src_pcs = apply_transformation(
#                     displacement.to(device), scale, rotation_matrix, no_scale=no_scale
#                 ) + inp_pcs[part_idx][j].to(device)
#                 if use_keypoints:
#                     if regularize_kp_only:
#                         cur_deformed_src_pcs = deformed_src_pcs
#                     else:
#                         cur_deformed_src_pcs = torch.einsum(
#                             "ij,ijk->ik",
#                             deformed_pc_weights[part_idx],
#                             deformed_src_pcs[deformed_nn_inds[part_idx]],
#                         )
#                 else:
#                     cur_deformed_src_pcs = deformed_src_pcs

#                 # rotation_matrix = quaternion_to_matrix(quaternion)
#                 # deformed_src_pcs = apply_transformation(
#                 #     init_displacement[0], scale, rotation_matrix, no_scale=no_scale
#                 # ) + src_pcs[j].to(device)

#                 total_displacement_after_update = (
#                     cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3).expand(
#                         num_pts[part_idx], num_nn, 3
#                     )
#                     - cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3)
#                     .expand(num_pts[part_idx], num_pts[part_idx], 3)
#                     .gather(
#                         0,
#                         nn_indices[part_idx][:, :num_nn]
#                         .to(device)
#                         .unsqueeze(-1)
#                         .expand(num_pts[part_idx], num_nn, 3),
#                     )
#                 )
#                 # total_dist_to_nn_after_update = (
#                 #     (
#                 #         cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3).expand(
#                 #             num_pts[part_idx], num_nn, 3
#                 #         )
#                 #         - cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3)
#                 #         .expand(num_pts[part_idx], num_pts[part_idx], 3)
#                 #         .gather(
#                 #             0,
#                 #             nn_indices[part_idx][:, :num_nn]
#                 #             .to(device)
#                 #             .unsqueeze(-1)
#                 #             .expand(num_pts[part_idx], num_nn, 3),
#                 #         )
#                 #         # ).pow(2).sum(dim=2) * nn_weights[:, :num_nn].to(device)
#                 #     )
#                 #     .pow(2)
#                 #     .sum(dim=2)
#                 # )
#                 # total_dist_to_nn_after_update = (
#                 #     deformed_src_pcs[j].view(num_pts, 1, 3).expand(num_pts, num_nn, 3)
#                 #     - deformed_src_pcs[j]
#                 #     .view(num_pts, 1, 3)
#                 #     .expand(num_pts, num_pts, 3)
#                 #     .gather(
#                 #         0,
#                 #         nn_indices[:, :num_nn]
#                 #         .to(device)
#                 #         .unsqueeze(-1)
#                 #         .expand(num_pts, num_nn, 3),
#                 #     )
#                 # # ).pow(2).sum(dim=2) * nn_weights[:, :num_nn].to(device)
#                 # ).pow(2).sum(dim=2)

#                 if loss_type == "L1":
#                     reg_loss = (
#                         total_dist_to_nn_after_update.sum(dim=1) - dist_to_nn.to(device)
#                     ).abs().sum() / (num_pts[part_idx] * num_nn)
#                 elif loss_type == "L1_fix":
#                     if reg_displacement:
#                         reg_loss = (
#                             (
#                                 total_displacement_after_update
#                                 - dist_to_nn[part_idx][:, :num_nn, :].to(device)
#                             )
#                         ).abs().sum() / (num_pts[part_idx] * num_nn)
#                     else:
#                         reg_loss = (
#                             total_displacement_after_update.pow(2).sum(dim=2)
#                             - dist_to_nn[part_idx][:, :num_nn].to(device)
#                         ).abs().sum() / (num_pts[part_idx] * num_nn)
#                 elif loss_type == "MSE":
#                     if reg_displacement:
#                         reg_loss = (
#                             torch.nn.functional.mse_loss(
#                                 total_displacement_after_update,
#                                 dist_to_nn[part_idx][:, :num_nn, :].to(device),
#                             )
#                             / num_nn
#                         )
#                     else:
#                         reg_loss = torch.nn.functional.mse_loss(
#                             total_displacement_after_update.pow(2).sum(dim=2),
#                             dist_to_nn[part_idx][:, :num_nn].to(device),
#                         )
#                 elif loss_type == "robust":
#                     x = (
#                         total_dist_to_nn_after_update
#                         - dist_to_nn[part_idx][:, :num_nn].to(device)
#                     ) / float(robust_c)
#                     reg_loss = (2 * x.pow(2) / (x.pow(2) + 4)).sum() / (
#                         num_pts[part_idx] * num_nn
#                     )
#                 elif loss_type == "welsch":
#                     x = (
#                         total_dist_to_nn_after_update
#                         - dist_to_nn[part_idx][:, :num_nn].to(device)
#                     ) / float(robust_c)
#                     reg_loss = (1.0 - torch.exp(x.pow(2) * -0.5)).sum() / (
#                         num_pts[part_idx] * num_nn
#                     )

#                 total_reg_loss += reg_loss * rigid_loss_w

#                 # ==== add the movement loss ====
#                 # new_deformed_src_pcs = torch.einsum(
#                 #     "ij,ijk->ik",
#                 #     deformed_pc_weights[part_idx],
#                 #     deformed_src_pcs[deformed_nn_inds[part_idx]],
#                 # )

#                 # avg_displacement = (
#                 #     new_deformed_src_pcs.view(num_pts[part_idx], 1, 3)
#                 #     .expand(num_pts[part_idx], num_nn, 3)
#                 #     .gather(
#                 #         0,
#                 #         nn_indices[part_idx][:, :num_nn]
#                 #         .to(device)
#                 #         .unsqueeze(-1)
#                 #         .expand(num_pts[part_idx], num_nn, 3),
#                 #     )
#                 #     - nn_init_positions[part_idx]
#                 #     .view(num_pts[part_idx], 1, 3)
#                 #     .expand(num_pts[part_idx], num_pts[part_idx], 3)
#                 #     .gather(
#                 #         0,
#                 #         nn_indices[part_idx][:, :num_nn]
#                 #         .to(device)
#                 #         .unsqueeze(-1)
#                 #         .expand(num_pts[part_idx], num_nn, 3),
#                 #     )
#                 # ).mean(dim=1)
#                 # avg_displacement_loss = (
#                 #     (
#                 #         new_deformed_src_pcs
#                 #         - (avg_displacement + nn_init_positions[part_idx])
#                 #     )
#                 #     .pow(2)
#                 #     .sum(dim=1)
#                 # ).mean()

#                 # ====================== try this version ======================
#                 # avg_displacement = (
#                 #     cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3)
#                 #     .expand(num_pts[part_idx], num_nn, 3)
#                 #     .gather(
#                 #         0,
#                 #         nn_indices[part_idx][:, :num_nn]
#                 #         .to(device)
#                 #         .unsqueeze(-1)
#                 #         .expand(num_pts[part_idx], num_nn, 3),
#                 #     )
#                 #     - nn_init_positions[part_idx]
#                 #     .view(num_pts[part_idx], 1, 3)
#                 #     .expand(num_pts[part_idx], num_pts[part_idx], 3)
#                 #     .gather(
#                 #         0,
#                 #         nn_indices[part_idx][:, :num_nn]
#                 #         .to(device)
#                 #         .unsqueeze(-1)
#                 #         .expand(num_pts[part_idx], num_nn, 3),
#                 #     )
#                 # ).mean(dim=1)
#                 # avg_displacement_loss = (
#                 #     (
#                 #         cur_deformed_src_pcs
#                 #         - (avg_displacement + nn_init_positions[part_idx])
#                 #     )
#                 #     .pow(2)
#                 #     .sum(dim=1)
#                 # ).mean()
#                 # total_reg_loss += avg_displacement_loss * ldas_loss_w

#                 # ==== add the displacement regularizer ====
#                 # Design 1: preserve the distance that each point in src travels
#                 #

#                 loss = total_reg_loss + cd_loss_w * total_cd_loss
#                 optimizer.zero_grad()
#                 # loss.backward(retain_graph=True)
#                 loss.backward()
#                 optimizer.step()
#                 # tqdm.tqdm.write(
#                 #     f"Loss: {loss.item()}, Reg Loss: {total_reg_loss.item()}, CD Loss: {total_cd_loss.item()}"
#                 # )
#             if i % 10 == 0:
#                 progress_bar.set_description(
#                     f"Training (Loss: {loss.item():.4f}) Reg Loss: {total_reg_loss.item():.4f}, CD Loss: {total_cd_loss.item():.4f}"
#                 )

#             if reg_boundary:
#                 deformed_bound_pts = []
#                 loss = 0.0
#                 for part_idx in range(total_part_num):
#                     cur_tgt_pc = tgt_pcs[part_idx]
#                     if input_case == 0 or input_case == 2:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     inp_pcs[part_idx][0].to(device),
#                                     pred_normals[part_idx][0].to(device),
#                                     pred_normals[part_idx][0].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                     rotation_matrix = quaternion_to_matrix(quaternion)
#                     deformed_src_pcs = apply_transformation(
#                         base_displacement[part_idx][0].to(
#                             device
#                         ),
#                         scale,
#                         rotation_matrix,
#                         no_scale=no_scale,
#                     ) + inp_pcs[part_idx][0].to(device)
#                     loss += chamfer_distance(
#                         deformed_src_pcs.unsqueeze(0), cur_tgt_pc.unsqueeze(0)
#                     )[0]

#                     if input_case == 0:
#                         scale, quaternion = transform_net(
#                             torch.cat(
#                                 [
#                                     inp_pcs[part_idx][j][boundary_indices[part_idx]].to(
#                                         device
#                                     ),
#                                     pred_normals[part_idx][0][
#                                         boundary_indices[part_idx]
#                                     ].to(device),
#                                     pred_normals[part_idx][j][
#                                         boundary_indices[part_idx]
#                                     ].to(device),
#                                 ],
#                                 dim=-1,
#                             )
#                         )
#                     rotation_matrix = quaternion_to_matrix(quaternion)
#                     deformed_bound_pts.append(apply_transformation(
#                         base_displacement[part_idx][j][boundary_indices[part_idx]].to(
#                             device
#                         ),
#                         scale,
#                         rotation_matrix,
#                         no_scale=no_scale,
#                     ) + inp_pcs[part_idx][j][boundary_indices[part_idx]].to(device))

#                 # add boundary loss
#                 # cur_boundary_dists = torch.norm(
#                 #     deformed_bound_pts[0].unsqueeze(1).expand(
#                 #         num_bound_pts[0], num_bound_pts[1], 3
#                 #     )
#                 #     - deformed_bound_pts[1]
#                 #     .unsqueeze(0)
#                 #     .expand(num_bound_pts[0], num_bound_pts[1], 3),
#                 #     dim=2,
#                 # )

#                 cur_boundary_dists = torch.norm(
#                     deformed_bound_pts[0][p1_bound_inds] - deformed_bound_pts[1],
#                     dim=-1,
#                 )
#                 optimizer.zero_grad()
#                 # loss += (cur_boundary_dists - boundary_dists.to(device)).abs().sum() / (
#                 #     num_bound_pts[0] * num_bound_pts[1]
#                 # )
#                 loss += (cur_boundary_dists - boundary_dists.to(device)).abs().sum() / (
#                     num_bound_pts[1]
#                 )
#                 loss.backward()
#                 optimizer.step()

#             # loss = total_reg_loss + cd_loss_w * total_cd_loss
#             # optimizer.zero_grad()
#             # loss.backward()
#             # optimizer.step()

#     # torch.save(reg_net.state_dict(), os.path.join(cur_log_dir, "reg_deform_net.pth"))
#     visualize_time_steps(pc_images, cur_log_dir, num_steps, n_iter)
#     torch.save(
#         transform_net.state_dict(), os.path.join(cur_log_dir, "transform_net.pth")
#     )
#     imageio.mimsave(os.path.join(cur_log_dir, 'train_deformed_pc.mp4'), pc_images, fps=10)
#     # save the deformed pcs
#     save_all_deformed_pcs()
#     visualize_kps(cur_log_dir, num_steps)
