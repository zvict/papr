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
from pointnet2_utils import (
    PointNetSetAbstractionMsg,
    PointNetSetAbstraction,
    PointNetFeaturePropagation,
)
from pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import KDTree, cKDTree
from scipy.optimize import linear_sum_assignment
import plotly.graph_objects as go
import open3d as o3d
from pytorch3d.ops import estimate_pointcloud_normals

os.chdir("/NAS/spa176/papr-retarget")


class pointnet2(nn.Module):
    def __init__(self,num_class,normal_channel=False):
        super(pointnet2, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [32, 64, 128], 3+additional_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.4,0.8], [64, 128], 128+128+64, [[128, 128, 256], [128, 196, 256]])
        self.sa3 = PointNetSetAbstraction(npoint=None, radius=None, nsample=None, in_channel=512 + 3, mlp=[256, 512, 1024], group_all=True)
        self.fp3 = PointNetFeaturePropagation(in_channel=1536, mlp=[256, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=576, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=150+additional_channel, mlp=[128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_class, 1)

    def forward(self, xyz):
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        return l1_points, l2_points, l3_points


class pointnet(nn.Module):
    def __init__(self, part_num=50, normal_channel=True):
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
        # out_max = torch.max(out5, 2, keepdim=True)[0]
        # out_max = out_max.view(-1, 2048)

        # out_max = torch.cat([out_max,label.squeeze(1)],1)
        # expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)
        # concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        # net = F.relu(self.bns1(self.convs1(concat)))
        # net = F.relu(self.bns2(self.convs2(net)))
        # net = F.relu(self.bns3(self.convs3(net)))
        # net = self.convs4(net)
        # net = net.transpose(2, 1).contiguous()
        # net = F.log_softmax(net.view(-1, self.part_num), dim=-1)
        # net = net.view(B, N, self.part_num) # [B, N, 50]

        # return net, trans_feat
        return out5


def pc_normalize(pc):
    l = pc.shape[0]
    centroid = torch.mean(pc, dim=0)
    pc = pc - centroid
    m = torch.max(torch.sqrt(torch.sum(pc**2, dim=1)))
    pc = pc / m
    return pc


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

def plot_pointcloud(points, save_dir, title=""):
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

    plt.savefig(os.path.join(save_dir, title + ".png"))
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()
    return np.array(img)


class DeformNet(nn.Module):
    def __init__(self, in_dim, out_dim=3, hidden_dim=256, num_layers=3, L=0, pt_dim=-1):
        super().__init__()

        self.pose_enc = PoseEnc()
        self.L = L
        if pt_dim > 0:
            in_dim = in_dim + pt_dim + pt_dim * 2 * L
        # in_dim = in_dim + in_dim * 2 * L
        self.mlp = MLP(in_dim, num_layers, hidden_dim, out_dim, use_wn=False, act_type="relu", last_act_type="none")

    def forward(self, x, pt=None):
        if pt is not None:
            x = torch.cat([x, self.pose_enc(pt, self.L)], dim=-1)
        # x = self.pose_enc(x, self.L)
        return self.mlp(x)


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


USE_ICP = True
# USE_FPS = False
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


# fps_k = 1000
num_keypoints = 192
num_keypoints_body = 256
all_num_keypoints = [num_keypoints, num_keypoints_body]

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
exp_dir = f'ot_wing{num_keypoints}_body{num_keypoints_body}'
if USE_COS:
    exp_dir += "_cos"
log_dir = os.path.join(log_dir, exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)
save_img_dir = os.path.join(log_dir, "images")
if not os.path.exists(save_img_dir):
    os.makedirs(save_img_dir, exist_ok=True)

src_pc_path = "/NAS/spa176/papr-retarget/point_clouds/butterfly/points_0.npy"
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

if USE_ICP:
    # ICP
    converged, rmse, Xt, RTs, t_history = icp(tgt_pc.unsqueeze(0), src_pc.unsqueeze(0))
    print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
    plot_pointcloud(Xt.squeeze(), log_dir, title="ICP Point Cloud")
    tgt_pc = Xt.squeeze(0)
    # exit(0)

# print("R :", RTs.R)
# print("t :", RTs.T)
# print("s :", RTs.s)
# # save the RTs to disk as torch tensor
# torch.save(RTs, "bird_to_but_RTs.pth")

# # # save the tgt_pc to disk as torch tensor
# # torch.save(tgt_pc, "rotated_bird_pc.pth")
# print(a)


# load wing indices
but_wing_indices = np.load("but_wing_indices.npy")
but_body_indices = np.setdiff1d(np.arange(len(src_pc)), but_wing_indices)
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

init_kps = []
kp_indices = []
tgt_kps = []
boundary_indices = []


if TRAIN_DEFORM_NET:
    total_result = []
    total_kp_result = []
    if USE_POINTNET:
        pointnet = pointnet(normal_channel=False).to(device)
        checkpoint = torch.load("best_model.pth")
        pointnet.load_state_dict(checkpoint["model_state_dict"])
        pointnet.eval()

        # if DENSIFY_BODY:
        #     # add points to the body of cur_src_pc by inserting one point between each pair of nearest neighbors
        #     cur_src_pc = src_pc[but_body_indices]
        #     cur_src_pc_np = cur_src_pc.cpu().numpy()
        #     tree = KDTree(cur_src_pc_np)
        #     nn_dists, nn_inds = tree.query(cur_src_pc_np, k=2)
        #     new_points = (cur_src_pc[nn_inds].sum(1) / 2)
        #     src_pc = torch.cat([src_pc, new_points], 0)

        with torch.no_grad():
            point_feat = pointnet(
                pc_normalize(src_pc).unsqueeze(0).transpose(2, 1)
            ).transpose(2, 1)[0]

    for part_idx in range(2):
        # for part_idx in range(1, 2):
        # for part_idx in range(1):

        if part_idx == 0:
            cur_src_pc = src_pc[but_wing_indices]
            cur_point_feat = point_feat[but_wing_indices]
            cur_tgt_pc = tgt_pc[bird_wing_indices]
            if USE_FPS:
                # FPS
                cur_src_kp, src_kp_idx = sfp(cur_src_pc.unsqueeze(0), K=num_keypoints)
                # cur_src_kp = cur_src_kp.squeeze(0)
                cur_kp_feat = cur_point_feat[src_kp_idx]
                cur_tgt_kp = sfp(cur_tgt_pc.unsqueeze(0), K=num_keypoints)[0]
        else:
            if DENSIFY_BODY:
                cur_idx = but_body_indices_concat
            else:
                cur_idx = but_body_indices
                # but_body_indices = np.setdiff1d(np.arange(len(src_pc)), but_wing_indices)
            cur_src_pc = src_pc[cur_idx]
            cur_point_feat = point_feat[cur_idx]
            # bird_body_indices = np.setdiff1d(np.arange(len(tgt_pc)), bird_wing_indices)
            cur_tgt_pc = tgt_pc[bird_body_indices]
            if USE_FPS:
                # FPS
                cur_src_kp, src_kp_idx = sfp(
                    src_pc[but_body_indices].unsqueeze(0), K=num_keypoints_body
                )
                # cur_src_kp = cur_src_kp.squeeze(0)
                cur_kp_feat = cur_point_feat[src_kp_idx]
                cur_tgt_kp = sfp(cur_tgt_pc.unsqueeze(0), K=num_keypoints_body)[
                    0
                ]
            # if DENSIFY_BODY:
            #     # add points to the body of cur_src_pc by inserting one point between each pair of nearest neighbors
            #     cur_src_pc_np = cur_src_pc.cpu().numpy()
            #     tree = KDTree(cur_src_pc_np)
            #     nn_dists, nn_inds = tree.query(cur_src_pc_np, k=2)
            #     new_points = (cur_src_pc[nn_inds].sum(1) / 2).cpu()
            #     cur_src_pc = torch.cat([cur_src_pc, new_points], 0)
        plot_pointcloud(cur_src_kp.squeeze(), log_dir, title=f"Part{part_idx} Src KP")
        plot_pointcloud(
            cur_tgt_kp.squeeze(), log_dir, title=f"Part{part_idx} Target KP"
        )

        PLOT_MATCHING = False
        # PLOT_MATCHING = True
        PLOT_NORMAL = False

        PLOT_BOUNDARY = False

        cur_src_kp = cur_src_kp.squeeze(0).cpu().numpy()  # Shape: (N, 3)
        cur_tgt_kp = cur_tgt_kp.squeeze(0).cpu().numpy()

        # Estimate surface normals for source and target key points
        cur_src_normals = estimate_normals_pytorch3d(cur_src_kp)
        cur_tgt_normals = estimate_normals_pytorch3d(cur_tgt_kp)

        # Compute the Euclidean distance matrix
        euclidean_cost_matrix = np.linalg.norm(
            cur_src_kp[:, np.newaxis, :] - cur_tgt_kp[np.newaxis, :, :],
            axis=2,
        )

        # Compute the feature distance matrix (e.g., Euclidean distance between normals)
        if USE_COS:
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
            alpha = 0.75  # Weighting factor for combining the distances
            # alpha = 0.5  # Weighting factor for combining the distances
            cost_matrix = alpha * euclidean_cost_matrix + (1 - alpha) * feature_cost_matrix

        # cost_matrix = np.linalg.norm(
        #     cur_src_kp[:, np.newaxis, :] - cur_tgt_kp[np.newaxis, :, :],
        #     axis=2,
        # )

        # Solve the optimal transport problem using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # The row_ind and col_ind arrays give the optimal 1-to-1 mapping
        # cur_src_kp[row_ind[i]] is mapped to cur_tgt_kp[col_ind[i]]
        mapping = list(zip(row_ind, col_ind))

        new_tgt_kp = np.zeros_like(cur_src_kp)
        for src_idx, tgt_idx in mapping:
            new_tgt_kp[src_idx] = cur_tgt_kp[tgt_idx]

        init_kps.append(torch.from_numpy(cur_src_kp).float())
        kp_indices.append(src_kp_idx)
        tgt_kps.append(torch.from_numpy(new_tgt_kp).float())

        # save the key point to disk
        np.save(os.path.join(log_dir, f"part{part_idx}_src_kp.npy"), cur_src_kp)
        np.save(os.path.join(log_dir, f"part{part_idx}_tgt_kp.npy"), new_tgt_kp)
        # save src_kp_idx
        # np.save(os.path.join(log_dir, f"part{part_idx}_src_kp_idx.npy"), src_kp_idx)
        torch.save(src_kp_idx, os.path.join(log_dir, f"part{part_idx}_src_kp_idx.pth"))

        if PLOT_MATCHING:

            # Create a Plotly scatter plot
            fig = go.Figure()

            # Add source key points
            fig.add_trace(go.Scatter3d(
                x=cur_src_kp[:, 0],
                y=cur_src_kp[:, 1],
                z=cur_src_kp[:, 2],
                mode='markers',
                marker=dict(size=5, color='blue'),
                name='Source Key Points'
            ))

            # Add target key points
            fig.add_trace(go.Scatter3d(
                x=cur_tgt_kp[:, 0],
                y=cur_tgt_kp[:, 1],
                z=cur_tgt_kp[:, 2],
                mode='markers',
                marker=dict(size=5, color='red'),
                name='Target Key Points'
            ))

            # Add vectors showing the mapping
            for i, j in mapping:
                fig.add_trace(go.Scatter3d(
                    x=[cur_src_kp[i, 0], cur_tgt_kp[j, 0]],
                    y=[cur_src_kp[i, 1], cur_tgt_kp[j, 1]],
                    z=[cur_src_kp[i, 2], cur_tgt_kp[j, 2]],
                    mode='lines',
                    line=dict(color='green', width=2),
                    name=f'Mapping {i}->{j}'
                ))

            if PLOT_NORMAL:
                # Add surface normals for source key points
                normal_length = 0.1  # Adjust the length of the normals as needed
                for i in range(cur_src_kp.shape[0]):
                    fig.add_trace(go.Scatter3d(
                        x=[cur_src_kp[i, 0], cur_src_kp[i, 0] + normal_length * cur_src_normals[i, 0]],
                        y=[cur_src_kp[i, 1], cur_src_kp[i, 1] + normal_length * cur_src_normals[i, 1]],
                        z=[cur_src_kp[i, 2], cur_src_kp[i, 2] + normal_length * cur_src_normals[i, 2]],
                        mode='lines',
                        line=dict(color='blue', width=2),
                        name='Source Normals'
                    ))

                # Add surface normals for target key points
                for i in range(cur_tgt_kp.shape[0]):
                    fig.add_trace(go.Scatter3d(
                        x=[cur_tgt_kp[i, 0], cur_tgt_kp[i, 0] + normal_length * cur_tgt_normals[i, 0]],
                        y=[cur_tgt_kp[i, 1], cur_tgt_kp[i, 1] + normal_length * cur_tgt_normals[i, 1]],
                        z=[cur_tgt_kp[i, 2], cur_tgt_kp[i, 2] + normal_length * cur_tgt_normals[i, 2]],
                        mode='lines',
                        line=dict(color='red', width=2),
                        name='Target Normals'
                    ))

            # Set plot layout
            fig.update_layout(
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Z'
                ),
                title='Key Point Mapping'
            )

            # Show the plot
            fig.show()
        # exit(0)

    # find the boundary points
    def find_boundary_points(part1, part2, threshold=0.1):
        # Convert tensors to numpy arrays for distance computation
        part1_np = part1.cpu().numpy()
        part2_np = part2.cpu().numpy()

        # Create KD-Trees for efficient nearest neighbor search
        tree1 = cKDTree(part1_np)
        tree2 = cKDTree(part2_np)

        # Find points in part1 that are close to any point in part2
        dists1, indices1 = tree1.query(part2_np, distance_upper_bound=threshold)
        boundary_points1 = part1_np[indices1[dists1 < threshold]]

        # Find points in part2 that are close to any point in part1
        dists2, indices2 = tree2.query(part1_np, distance_upper_bound=threshold)
        boundary_points2 = part2_np[indices2[dists2 < threshold]]

        # Convert boundary points back to tensors
        boundary_points1_indices = indices1[dists1 < threshold]
        boundary_points2_indices = indices2[dists2 < threshold]

        return boundary_points1_indices, boundary_points2_indices

    boundary_threshold = 0.75
    # Find the boundary points between the two parts
    boundary_points1_indices, boundary_points2_indices = find_boundary_points(
        tgt_kps[0], tgt_kps[1], threshold=boundary_threshold
    )
    boundary_indices.append(boundary_points1_indices)
    boundary_indices.append(boundary_points2_indices)

    if PLOT_BOUNDARY:
        # plot the target kps and mark the boundary points red
        # Create a Plotly scatter plot
        fig = go.Figure()

        all_tgt_kps = torch.cat(tgt_kps, 0).cpu().numpy()
        # Add source key points
        fig.add_trace(
            go.Scatter3d(
                x=all_tgt_kps[:, 0],
                y=all_tgt_kps[:, 1],
                z=all_tgt_kps[:, 2],
                mode="markers",
                marker=dict(size=5, color="blue"),
                name="Source Key Points",
            )
        )

        part_0_boundary_points = tgt_kps[0][boundary_points1_indices].cpu().numpy()

        # Add target key points
        fig.add_trace(go.Scatter3d(
            x=part_0_boundary_points[:, 0],
            y=part_0_boundary_points[:, 1],
            z=part_0_boundary_points[:, 2],
            mode='markers',
            marker=dict(size=5, color='red'),
            name='Target Key Points'
        ))

        part_1_boundary_points = tgt_kps[1][boundary_points2_indices].cpu().numpy()

        # Add target key points
        fig.add_trace(go.Scatter3d(
            x=part_1_boundary_points[:, 0],
            y=part_1_boundary_points[:, 1],
            z=part_1_boundary_points[:, 2],
            mode='markers',
            marker=dict(size=5, color='purple'),
            name='Target Key Points'
        ))

        # Set plot layout
        fig.update_layout(
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title='Boundary Points'
        )

        # Show the plot
        fig.show()


else:
    for part_idx in range(2):
        init_kps.append(
            torch.from_numpy(
                np.load(os.path.join(log_dir, f"part{part_idx}_src_kp.npy"))
            ).float()
        )
        tgt_kps.append(
            torch.from_numpy(
                np.load(os.path.join(log_dir, f"part{part_idx}_tgt_kp.npy"))
            ).float()
        )
        kp_indices.append(
            np.load(os.path.join(log_dir, f"part{part_idx}_src_kp_idx.pth"))
        )


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

        # Calculate the average point cloud for this window
        average_point_cloud = np.mean(window, axis=0)

        # Add the average point cloud to the list of smoothed point clouds
        smoothed_point_clouds.append(average_point_cloud)

    # Set the first and last point clouds to be the same as the original ones
    smoothed_point_clouds[0] = point_clouds[0]
    smoothed_point_clouds[-1] = point_clouds[-1]

    return smoothed_point_clouds


if REG_DEFORM_NET:
    """
    Regularize the point cloud
    """

    # n_iter = 2000
    n_iter = 300
    batch_size = 10000
    robust_c = 0.2
    # cd_loss_w = 1000.0
    cd_loss_w = 1.0
    # cd_loss_w = 2.0
    rigid_loss_w = 1.0
    # rigid_loss_w = 1000.0
    ldas_loss_w = 0.0
    # cd_weight_decay = 'zeros'
    src_pc_dir = "/NAS/spa176/papr-retarget/point_clouds/butterfly"

    # num_nn = 100
    num_nn_wing = 5
    # num_nn_body = 250
    # num_nn_body = 90
    num_nn_body = 24
    # num_nn_wing = num_nn_body = 100
    num_nns = [num_nn_wing, num_nn_body]
    concat_feature = True
    input_case = 0
    # input_case = 1
    # input_case = 2
    # input_case = 3
    no_scale = False
    rotate_normal = True
    # add key point
    use_keypoints = True
    # use_keypoints = False
    flip_normal = True
    # num_keypoints = 64
    # num_keypoints = 96
    # num_keypoints = 192
    # num_keypoints_body = 64
    # num_keypoints_body = 256
    # num_keypoints_body = 96
    # kp_knn = 5
    # kp_knn_body = 5
    # kp_knn_body = 50
    # kp_knn = 20
    kp_knn = 50
    motion_frame_skip = 200
    kp_knn_body = 24
    # kp_knn_body = 24
    # kp_knn_body = 48
    # kp_knn_body = 256
    # num_keypoints = 1024
    # kp_knn = 5
    total_part_num = 2

    regularize_kp_only = True
    double_side_chamfer = True
    smooth_inp_seq = False
    smooth_window_size = 35

    # reg_displacement = True
    reg_displacement = False

    pytorch3d_est_normal = True

    # test_transform_net = True
    test_transform_net = False

    # force_smooth = True
    force_smooth = False

    force_smooth_full = True
    smooth_knn = 50

    reg_boundary = True
    reg_boundary_test = False

    transform_L = 0
    loss_type = "L1_fix"
    # loss_type = "MSE"
    # cur_log_dir = os.path.join(log_dir, f'test_deformed_pc_regularized_cdw{cd_loss_w}_wdecay{cd_weight_decay}')
    if concat_feature:
        if use_keypoints:
            cur_log_dir = os.path.join(
                log_dir,
                # f"transform_wing_kp_{num_keypoints}_body_kp_{num_keypoints_body}_kpnn_{kp_knn}_bodykpnn_{kp_knn_body}_frame_skip_{motion_frame_skip}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nnwing{num_nn_wing}_nnbody{num_nn_body}_concat_{input_case}",
                f"{loss_type}_wing_kp_{num_keypoints}_body_kp_{num_keypoints_body}_kpnn_{kp_knn}_bodykpnn_{kp_knn_body}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_nnwing{num_nn_wing}_nnbody{num_nn_body}",
            )
            # cur_log_dir = os.path.join(
            #     log_dir,
            #     f"transform_wing_kp_{num_keypoints}_body_kp_{num_keypoints_body}_kpnn_{kp_knn}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nn{num_nn}_concat_{input_case}",
            #     # f"test_deformed_pc",
            # )
        else:
            cur_log_dir = os.path.join(
                log_dir,
                f"transform_L{transform_L}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nn{num_nn}_concat_{input_case}",
                # f"test_deformed_pc",
            )
        if regularize_kp_only:
            cur_log_dir += "_kp_only"

        # if double_side_chamfer:
        #     cur_log_dir += "_double_side_chamfer"

        if smooth_inp_seq:
            cur_log_dir += "_smooth_inp_seq_size" + str(smooth_window_size)

        if reg_displacement:
            cur_log_dir += "_regD"

        if pytorch3d_est_normal:
            cur_log_dir += "_p3dNorm"

        if input_case > 0:
            cur_log_dir += "_inp" + str(input_case)

        if reg_boundary:
            cur_log_dir += f"_RB{boundary_threshold}"

    os.makedirs(cur_log_dir, exist_ok=True)

    # copy current file to the log directory
    subprocess.run(["cp", __file__, cur_log_dir])

    # deform_vectors = torch.load(deform_vectors_path)

    start = 0
    end = 30001
    interval = motion_frame_skip

    src_pcs = [[] for _ in range(total_part_num)]
    tgt_pcs = []
    for part_idx in range(total_part_num):
        if part_idx == 0:
            # tgt_pcs.append(torch.tensor(tgt_pc[bird_wing_indices]).float())
            tgt_pcs.append(tgt_pc[bird_wing_indices])
        else:
            # tgt_pcs.append(torch.tensor(tgt_pc[bird_body_indices]).float())
            tgt_pcs.append(tgt_pc[bird_body_indices])

    deformed_pcs = [[] for _ in range(total_part_num)]
    if use_keypoints:
        # kp_indices = [[] for _ in range(total_part_num)]
        # init_kps   = [[] for _ in range(total_part_num)]
        key_points = [[] for _ in range(total_part_num)]

    ########## Load the point cloud by parts ##########
    ########## Optionally add key points ##########
    scale = 10.0
    raw_pcs = []
    for idx in tqdm.tqdm(range(start, end, interval)):
        src_pc_path = os.path.join(src_pc_dir, f"points_{idx}.npy")
        cur_src_pc = np.load(src_pc_path)
        cur_src_pc = cur_src_pc / scale
        raw_pcs.append(cur_src_pc)

    if smooth_inp_seq:
        raw_pcs = smooth_point_cloud(raw_pcs, smooth_window_size)

    for idx in tqdm.tqdm(range(len(raw_pcs))):
        if idx == 0:
            src_pc = src_pc.cpu()
        else:
            src_pc = torch.tensor(raw_pcs[idx]).float()

        for part_idx in range(total_part_num):
            if part_idx == 0:
                cur_src_pc = src_pc[but_wing_indices]
            else:
                if DENSIFY_BODY and idx == 0:
                    cur_src_pc = src_pc[but_body_indices_concat]
                else:
                    # but_body_indices = np.setdiff1d(
                    #     np.arange(len(src_pc)), but_wing_indices
                    # )
                    cur_src_pc = src_pc[but_body_indices]
            # if idx == start:
            #     if use_keypoints:
            #         src_pcs[part_idx] = cur_src_pc.unsqueeze(0)
            #     else:
            #         src_pcs[part_idx].append(cur_src_pc)

            if idx == 0:
                full_src_pc = src_pc.clone()
                if use_keypoints:
                    src_pcs[part_idx] = cur_src_pc.unsqueeze(0)

                    if part_idx == 0:
                        cur_num_keypoints = num_keypoints
                    else:
                        cur_num_keypoints = num_keypoints_body
                        if DENSIFY_BODY:
                            cur_src_pc = src_pc[but_body_indices]
                    # apply farthest point sampling to get the keypoints
                    # init_kps[part_idx], kp_indices[part_idx] = sfp(
                    #     cur_src_pc.unsqueeze(0), K=cur_num_keypoints
                    # )

            if use_keypoints:
                key_points[part_idx].append(
                    cur_src_pc[kp_indices[part_idx][0].cpu()].clone()
                )
            else:
                src_pcs[part_idx].append(cur_src_pc)
            # print(f"add key point on device {part_idx}: ", key_points[part_idx][-1].device)
            # print(f"key point shape {part_idx}: ", key_points[part_idx].shape)

    for part_idx in range(total_part_num):
        if use_keypoints:
            key_points[part_idx] = torch.stack(key_points[part_idx], dim=0)
            print(f"key_points[{part_idx}]: ", key_points[part_idx].shape)
        else:
            src_pcs[part_idx] = torch.stack(src_pcs[part_idx], dim=0)
            print(f"src_pcs[{part_idx}]: ", src_pcs[part_idx].shape)

    # augment the key points to include the boundary points
    if reg_boundary:
        augmented_key_points = [
            torch.cat([key_points[0], key_points[1][:, boundary_indices[1], :]], dim=1),
            torch.cat([key_points[1], key_points[0][:, boundary_indices[0], :]], dim=1)
        ]
        augmented_tgt_kps = [
            torch.cat([tgt_kps[0], tgt_kps[1][boundary_indices[1]]], dim=0),
            torch.cat([tgt_kps[1], tgt_kps[0][boundary_indices[0]]], dim=0)
        ]
        num_augmented_pts = [kp.shape[1] for kp in augmented_key_points]

    # num_steps = src_pcs[0].shape[0]
    if use_keypoints:
        num_steps = key_points[0].shape[0]
    else:
        num_steps = src_pcs[0].shape[0]
    num_pts = [src_pcs[part_idx].shape[1] for part_idx in range(total_part_num)]
    # num_steps, num_pts, _ = src_pcs.shape
    if regularize_kp_only:
        for part_idx in range(total_part_num):
            num_pts[part_idx] = key_points[part_idx].shape[1] 
    if reg_boundary:
        num_pts = [kp.shape[1] for kp in augmented_key_points]
    if reg_displacement:
        dist_to_nn = [
            torch.empty(num_pts[part_idx], num_nns[part_idx], 3)
            for part_idx in range(total_part_num)
        ]
    else:
        dist_to_nn = [
            torch.empty(num_pts[part_idx], num_nns[part_idx]) for part_idx in range(total_part_num)
        ]
    nn_indices = [torch.empty(num_pts[part_idx], num_nns[part_idx]) for part_idx in range(total_part_num)]
    nn_weights = [torch.empty(num_pts[part_idx], num_nns[part_idx]) for part_idx in range(total_part_num)]
    nn_init_positions = [torch.empty(num_pts[part_idx], num_nns[part_idx], 3) for part_idx in range(total_part_num)]

    # dist_to_nn = torch.empty(num_pts, num_nn)
    # nn_indices = torch.empty(num_pts, num_nn)
    # nn_weights = torch.empty(num_pts, num_nn)

    pc_images = []

    # release the memory of reg_net
    # del reg_net
    if USE_POINTNET:
        del pointnet

    sample_skip = num_steps // 6
    sample_steps = torch.arange(0, num_steps + 1, sample_skip)

    if use_keypoints:
        pred_deformed_pcs = []
        deformed_pc_weights = []
        deformed_nn_inds = []
        pred_normals = []
        ori_kp_deforms = []
        vis_ori_kp_deforms = []
        for part_idx in range(total_part_num):
            # parameterize the src point cloud
            # pred_src_pcs, src_pc_weights, src_nn_inds = parametertize_pc(src_pcs[part_idx], init_kps[part_idx][0], kp_knn, step=12000)
            # parameterize the deformed point cloud
            if part_idx == 0:
                cur_kp_knn = kp_knn
            else:
                cur_kp_knn = kp_knn_body
            pred_deformed_pc, deformed_pc_weight, deformed_nn_ind = parametertize_pc(
                tgt_pcs[part_idx].to(device),
                tgt_kps[part_idx].to(device),
                cur_kp_knn,
                step=35000,
            )
            pred_deformed_pcs.append(pred_deformed_pc)
            deformed_pc_weights.append(deformed_pc_weight.to(device))
            deformed_nn_inds.append(deformed_nn_ind)
            # deformed_nn_inds = deformed_nn_inds.to(device)
            # deformed_pc_weights = deformed_pc_weights.to(device)

            # calculate the surface normals at each src_pcs[i] and store the rotation
            # normal_rotation_matrices = compute_rotation_matrices_for_batch(
            #     src_pcs[sample_steps], num_nn, flip_normal=flip_normal
            # )
            # deformed_kps = torch.matmul(
            #     normal_rotation_matrices[:, kp_indices[0], :, :],
            #     init_displacement[:, kp_indices[0], :].detach().cpu().expand(len(sample_steps), -1, -1).unsqueeze(-1),
            # ).squeeze(-1) + key_points[sample_steps]
            pc_img = plot_pointcloud(
                pred_deformed_pc,
                cur_log_dir,
                title=f"Pred PC Part {part_idx}",
            )

            normal_rotation_matrices, pred_normal = (
                compute_rotation_matrices_for_batch(
                    key_points[part_idx], 10, flip_normal=flip_normal
                )
            )
            pred_normals.append(pred_normal)
            ori_kp_deform = torch.matmul(
                normal_rotation_matrices,
                (tgt_kps[part_idx] - init_kps[part_idx])
                .expand(len(normal_rotation_matrices), -1, -1)
                .unsqueeze(-1)
                .cpu(),
            ).squeeze(-1)
            ori_kp_deforms.append(ori_kp_deform)
            # for visualization
            vis_ori_kp_deforms.append(ori_kp_deform[sample_steps])

            # deformed_kps = torch.matmul(
            #     normal_rotation_matrices,
            #     init_displacement[:, kp_indices[0], :].detach().cpu().expand(len(sample_steps), -1, -1).unsqueeze(-1),
            # ).squeeze(-1) + key_points[sample_steps]

            # deformed_src_pcs = []
            # for deformed_kp in deformed_kps:
            #     deformed_src_pcs.append(
            #         torch.einsum('ij,ijk->ik', deformed_pc_weights, deformed_kp[deformed_nn_inds])
            #     )
            # deformed_src_pcs = torch.stack(deformed_src_pcs, dim=0)

            # apply the rotation matrix to init_displacement given the normal rotation matrices and add to scr_pcs
            # deformed_src_pcs = torch.matmul(
            #     normal_rotation_matrices,
            #     init_displacement.detach().cpu().expand(len(sample_steps), -1, -1).unsqueeze(-1),
            # ).squeeze(-1) + src_pcs[sample_steps]
            # pc_img = plot_pointcloud(deformed_src_pcs, cur_log_dir, title=f"Deformed Point Cloud Iter {0}")
            # print(a)
    else:
        pred_normals = []
        ori_kp_deforms = []
        vis_ori_kp_deforms = []
        total_deformed_pc = []
        for part_idx in range(total_part_num):
            if part_idx == 0:
                normal_nn = 100
            else:
                normal_nn = 100

            normal_rotation_matrices, pred_normal = compute_rotation_matrices_for_batch(
                src_pcs[part_idx][sample_steps], normal_nn, flip_normal=flip_normal
            )
            pred_normals.append(pred_normal)
            ori_pc_deform = torch.matmul(
                normal_rotation_matrices,
                (deformed_src_pc_start[part_idx].cpu() - src_pcs[part_idx][0])
                .expand(len(normal_rotation_matrices), -1, -1)
                .unsqueeze(-1),
            ).squeeze(-1)
            total_deformed_pc.append(ori_pc_deform + src_pcs[part_idx][sample_steps])
        total_deformed_pc = torch.cat(total_deformed_pc, dim=1)
        plot_pointcloud(
            total_deformed_pc,
            cur_log_dir,
            title=f"Initial Deformed Point Cloud",
        )
        print(a)

    if concat_feature:
        if input_case == 0 or input_case == 2:
            transform_net = AffineTransformationNet(9, L=transform_L).to(device)
        elif input_case == 1 or input_case == 3:
            transform_net = AffineTransformationNet(9, L=transform_L, non_pe_dim=1).to(device)
        else:
            transform_net = AffineTransformationNet(6, L=transform_L).to(device)
    else:
        transform_net = AffineTransformationNet(3, L=transform_L).to(device)

    def visualize_time_steps(pc_images, cur_log_dir, num_steps, cur_iter):
        with torch.no_grad():
            # sample_steps = torch.arange(0, num_steps + 1, 5)

            total_deformed_pc = []

            for part_idx in range(total_part_num):

                if use_keypoints:
                    if input_case == 2 or input_case == 3:
                        pc_inp = key_points[part_idx][0:1].expand(len(sample_steps), -1, -1)
                        base_pc = key_points[part_idx][sample_steps]
                    else:
                        pc_inp = base_pc = key_points[part_idx][sample_steps]

                    vis_base_displacement = vis_ori_kp_deforms[part_idx]
                else:
                    pc_inp = src_pcs[sample_steps]
                    vis_base_displacement = init_displacement

                if concat_feature:
                    if input_case == 0 or input_case == 2:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    pc_inp.to(device),
                                    pred_normals[part_idx][0:1]
                                    .expand(len(sample_steps), -1, -1)
                                    .to(device),
                                    pred_normals[part_idx][sample_steps].to(device),
                                ],
                                dim=-1,
                            )
                        )
                    elif input_case == 1 or input_case == 3:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    pc_inp.to(device),
                                    pred_normals[part_idx][0:1]
                                    .expand(len(sample_steps), -1, -1)
                                    .to(device),
                                    pred_normals[part_idx][sample_steps].to(device),
                                ],
                                dim=-1,
                            ),
                            non_pe=sample_steps.unsqueeze(-1)
                            .unsqueeze(-1)
                            .expand(-1, all_num_keypoints[part_idx], -1)
                            .float()
                            .to(device),
                        )
                    else:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    src_pcs[sample_steps].to(device),
                                    init_displacement.expand(len(sample_steps), -1, -1),
                                ],
                                dim=-1,
                            )
                        )
                else:
                    scale, quaternion = transform_net(pc_inp.to(device))
                rotation_matrix = quaternion_to_matrix(quaternion)
                deformed_src_pcs = apply_transformation(
                    vis_base_displacement.to(device),
                    scale,
                    rotation_matrix,
                    no_scale=no_scale,
                ) + base_pc.to(device)
                # deformed_src_pcs = vis_base_displacement.to(device) + pc_inp.to(device)
                if use_keypoints:
                    deformed_pcs = []
                    for deformed_kp in deformed_src_pcs:
                        deformed_pcs.append(
                            torch.einsum(
                                "ij,ijk->ik",
                                deformed_pc_weights[part_idx],
                                deformed_kp[deformed_nn_inds[part_idx]],
                            )
                        )
                    total_deformed_pc.append(torch.stack(deformed_pcs, dim=0))
            total_deformed_pc = torch.cat(total_deformed_pc, dim=1)
            pc_img = plot_pointcloud(
                total_deformed_pc,
                cur_log_dir,
                title=f"Deformed Point Cloud Iter {cur_iter}",
            )
            pc_images.append(pc_img)

    def save_all_deformed_pcs():
        with torch.no_grad():
            total_deformed_pc = []
            cur_order = [1, 0] if reg_boundary_test else range(total_part_num)

            # for part_idx in range(total_part_num):
            for part_idx in cur_order:
                if use_keypoints:
                    if input_case == 2 or input_case == 3:
                        pc_inp = key_points[part_idx][0:1].expand(num_steps, -1, -1)
                        base_pc = key_points[part_idx]
                    else:
                        pc_inp = base_pc = key_points[part_idx]
                    vis_base_displacement = ori_kp_deforms[part_idx]
                else:
                    pc_inp = src_pcs[sample_steps]
                    vis_base_displacement = init_displacement

                if concat_feature:
                    if input_case == 0 or input_case == 2:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    pc_inp.to(device),
                                    pred_normals[part_idx][0:1]
                                    .expand(num_steps, -1, -1)
                                    .to(device),
                                    pred_normals[part_idx].to(device),
                                ],
                                dim=-1,
                            )
                        )
                    elif input_case == 1 or input_case == 3:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    pc_inp.to(device),
                                    pred_normals[part_idx][0:1]
                                    .expand(num_steps, -1, -1)
                                    .to(device),
                                    pred_normals[part_idx].to(device),
                                ],
                                dim=-1,
                            ),
                            non_pe=torch.arange(num_steps)
                            .unsqueeze(-1)
                            .unsqueeze(-1)
                            .expand(-1, all_num_keypoints[part_idx], -1)
                            .float()
                            .to(device),
                        )
                    else:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    src_pcs[sample_steps].to(device),
                                    init_displacement.expand(len(sample_steps), -1, -1),
                                ],
                                dim=-1,
                            )
                        )
                else:
                    scale, quaternion = transform_net(pc_inp.to(device))
                rotation_matrix = quaternion_to_matrix(quaternion)
                deformed_src_pcs = apply_transformation(
                    vis_base_displacement.to(device),
                    scale,
                    rotation_matrix,
                    no_scale=no_scale,
                ) + base_pc.to(device)
                # deformed_src_pcs = vis_base_displacement.to(device) + pc_inp.to(device)
                if force_smooth:
                    for time_step in range(1, num_steps):
                        avg_displacement = (
                            deformed_src_pcs[time_step]
                            .view(num_pts[part_idx], 1, 3)
                            .expand(num_pts[part_idx], num_nns[part_idx], 3)
                            .gather(
                                0,
                                nn_indices[part_idx][:, :num_nns[part_idx]]
                                .to(device)
                                .unsqueeze(-1)
                                .expand(num_pts[part_idx], num_nns[part_idx], 3),
                            )
                            - nn_init_positions[part_idx]
                            .view(num_pts[part_idx], 1, 3)
                            .expand(num_pts[part_idx], num_pts[part_idx], 3)
                            .gather(
                                0,
                                nn_indices[part_idx][:, :num_nns[part_idx]]
                                .to(device)
                                .unsqueeze(-1)
                                .expand(num_pts[part_idx], num_nns[part_idx], 3),
                            )
                        ).mean(dim=1)
                        deformed_src_pcs[time_step] = (
                            avg_displacement + nn_init_positions[part_idx]
                        )

                if reg_boundary_test:
                    # calculate the initial mean displacement between the sets of two boundary points
                    # mean_bound_displacement = torch.mean(
                    #     tgt_kps[0][boundary_indices[0]], dim=0) - torch.mean(tgt_kps[1][boundary_indices[1]], dim=0
                    # )
                    if part_idx == 1:
                        anchor_kps = deformed_src_pcs.clone()
                    else:
                        cur_mean_bound_displacement = torch.mean(
                            deformed_src_pcs[:, boundary_indices[0]], dim=1) - torch.mean(
                            - anchor_kps[:, boundary_indices[1]],
                            dim=1,
                        ) # shape (num_steps, 3)
                        # print(f"!!!!cur_mean_bound_displacement: {cur_mean_bound_displacement.shape}")
                        # print("mean displacement", mean_bound_displacement)
                        # print("$$$$$")
                        # print("cur_mean_bound_displacement", cur_mean_bound_displacement[:5])
                        # deformed_src_pcs = deformed_src_pcs + (mean_bound_displacement.to(device) - cur_mean_bound_displacement).unsqueeze(1).expand(-1, num_pts[part_idx], -1)
                        deformed_src_pcs = deformed_src_pcs - (cur_mean_bound_displacement[0:1, :] - cur_mean_bound_displacement).unsqueeze(1).expand(-1, num_pts[part_idx], -1)

                if use_keypoints:
                    deformed_pcs = []
                    for deformed_kp in deformed_src_pcs:
                        deformed_pcs.append(
                            torch.einsum(
                                "ij,ijk->ik",
                                deformed_pc_weights[part_idx],
                                deformed_kp[deformed_nn_inds[part_idx]],
                            )
                        )
                    total_deformed_pc.append(torch.stack(deformed_pcs, dim=0))

                if force_smooth_full:
                    base_pc = total_deformed_pc[-1][0]
                    cur_num_pts = base_pc.shape[0]

                    nn_indices = torch.empty(cur_num_pts, smooth_knn, dtype=torch.int64, device=device)
                    for pt_idx in range(cur_num_pts):
                        # find the distance from the point at index i to all others points
                        displacement_to_all_pts = (
                            base_pc[pt_idx : pt_idx + 1, :].expand(cur_num_pts, 3)
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
                            total_deformed_pc[-1][time_step]
                            .view(cur_num_pts, 1, 3)
                            .expand(cur_num_pts, cur_num_pts, 3)
                            .gather(
                                0,
                                nn_indices
                                .to(device)
                                .unsqueeze(-1)
                                .expand(cur_num_pts, smooth_knn, 3),
                            )
                            - base_pc.view(cur_num_pts, 1, 3)
                            .expand(cur_num_pts, cur_num_pts, 3)
                            .gather(
                                0,
                                nn_indices
                                .to(device)
                                .unsqueeze(-1)
                                .expand(cur_num_pts, smooth_knn, 3),
                            )
                        ).mean(dim=1)
                        total_deformed_pc[-1][time_step] = avg_displacement + base_pc

            if reg_boundary_test:
                # swap the two parts in total_deformed_pc
                total_deformed_pc[0], total_deformed_pc[1] = total_deformed_pc[1], total_deformed_pc[0]
            total_deformed_pc = torch.cat(total_deformed_pc, dim=1) # shape (num_steps, num_pts, 3)

        if force_smooth:
            torch.save(total_deformed_pc, os.path.join(cur_log_dir, "total_deformed_pc_smooth.pth"))
        elif force_smooth_full:
            if smooth_knn != 100:
                torch.save(total_deformed_pc, os.path.join(cur_log_dir, f"total_deformed_pc_smooth_full_{smooth_knn}.pth"))
            else:
                torch.save(total_deformed_pc, os.path.join(cur_log_dir, "total_deformed_pc_smooth_full.pth"))
        # elif reg_boundary:
        #     torch.save(total_deformed_pc, os.path.join(cur_log_dir, "total_deformed_pc_rb.pth"))
        else:
            # save the total deformed point cloud as a pytorch tensor
            torch.save(total_deformed_pc, os.path.join(cur_log_dir, "total_deformed_pc.pth"))
        # save the initial deformed point cloud at start state
        torch.save(
            torch.cat(pred_deformed_pcs, dim=0),
            os.path.join(cur_log_dir, "deformed_src_pc_start.pth"),
        )

    def save_deformed_kps():
        with torch.no_grad():
            for part_idx in range(total_part_num):
                if use_keypoints:
                    if input_case == 2 or input_case == 3:
                        pc_inp = key_points[part_idx][0:1].expand(num_steps, -1, -1)
                        base_pc = key_points[part_idx]
                    else:
                        pc_inp = base_pc = key_points[part_idx]
                    vis_base_displacement = ori_kp_deforms[part_idx]
                else:
                    pc_inp = src_pcs[sample_steps]
                    vis_base_displacement = init_displacement

                if concat_feature:
                    if input_case == 0 or input_case == 2:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    pc_inp.to(device),
                                    pred_normals[part_idx][0:1]
                                    .expand(num_steps, -1, -1)
                                    .to(device),
                                    pred_normals[part_idx].to(device),
                                ],
                                dim=-1,
                            )
                        )
                    elif input_case == 1 or input_case == 3:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    pc_inp.to(device),
                                    pred_normals[part_idx][0:1]
                                    .expand(num_steps, -1, -1)
                                    .to(device),
                                    pred_normals[part_idx].to(device),
                                ],
                                dim=-1,
                            ),
                            non_pe=torch.arange(num_steps)
                            .unsqueeze(-1)
                            .unsqueeze(-1)
                            .expand(-1, all_num_keypoints[part_idx], -1)
                            .float()
                            .to(device),
                        )
                    else:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    src_pcs[sample_steps].to(device),
                                    init_displacement.expand(len(sample_steps), -1, -1),
                                ],
                                dim=-1,
                            )
                        )
                else:
                    scale, quaternion = transform_net(pc_inp.to(device))
                rotation_matrix = quaternion_to_matrix(quaternion)
                deformed_src_pcs = apply_transformation(
                    vis_base_displacement.to(device),
                    scale,
                    rotation_matrix,
                    no_scale=no_scale,
                ) + base_pc.to(device)

                # save the total deformed point cloud as a pytorch tensor
                torch.save(
                    deformed_src_pcs, os.path.join(cur_log_dir, f"total_deformed_kps_p{part_idx}.pth")
                )

    # if test_transform_net:
    #     transform_net.load_state_dict(
    #         torch.load(os.path.join(cur_log_dir, "transform_net.pth"))
    #     )
    #     # visualize_time_steps(pc_images, cur_log_dir, num_steps, n_iter + 1)
    #     # save_all_deformed_pcs()
    #     save_deformed_kps()
    #     exit(0)

    # # ===== old optimizer of the deform net =====
    # # optimizer = torch.optim.Adam(reg_net.parameters(), lr=0.0005)
    # optimizer = torch.optim.Adam(transform_net.parameters(), lr=0.0005)

    if use_keypoints:
        if regularize_kp_only:
            base_dist_pcs = tgt_kps
            base_displacement = ori_kp_deforms
        else:
            base_dist_pcs = pred_deformed_pcs
            base_displacement = ori_kp_deforms
        if reg_boundary:
            base_dist_pcs = augmented_tgt_kps
            base_displacement = augmented_ori_kp_deforms = [
                torch.cat(
                    [ori_kp_deforms[0], ori_kp_deforms[1][:, boundary_indices[1]]], dim=1
                ),
                torch.cat(
                    [ori_kp_deforms[1], ori_kp_deforms[0][:, boundary_indices[0]]], dim=1
                ),
            ]
            augmented_pred_normals = [
                torch.cat(
                    [pred_normals[0], pred_normals[1][:, boundary_indices[1]]], dim=1
                ),
                torch.cat(
                    [pred_normals[1], pred_normals[0][:, boundary_indices[0]]], dim=1
                ),
            ]
    else:
        base_dist_pcs = deformed_src_pc
        base_displacement = init_displacement

    for part_idx in range(total_part_num):
        for pt_idx in range(num_pts[part_idx]):
            # find the distance from the point at index i to all others points
            displacement_to_all_pts = (
                base_dist_pcs[part_idx][pt_idx : pt_idx + 1, :].expand(
                    num_pts[part_idx], 3
                )
                - base_dist_pcs[part_idx]
            )

            # dist_to_all_pts = (
            #     (
            #         base_dist_pcs[part_idx][pt_idx : pt_idx + 1, :].expand(
            #             num_pts[part_idx], 3
            #         )
            #         - base_dist_pcs[part_idx]
            #     )
            #     .pow(2)
            #     .sum(dim=1)
            # )
            vals, inds = torch.topk(
                displacement_to_all_pts.pow(2).sum(dim=1), num_nns[part_idx] + 1, largest=False, sorted=True
            )
            if reg_displacement:
                dist_to_nn[part_idx][pt_idx, :, :], nn_indices[part_idx][pt_idx, :] = (
                    displacement_to_all_pts[inds[1:]],
                    inds[1:],
                )
            else:
                dist_to_nn[part_idx][pt_idx, :], nn_indices[part_idx][pt_idx, :] = (
                    vals[1:],
                    inds[1:],
                )
            # gaussian_weight = 0
            # nn_weights[pt_idx, :] = (
            #     (-dist_to_nn[pt_idx, :] * gaussian_weight).exp().detach()
            # )

        # calculate the current sum of distances from each point to its nearest neighbors
        nn_indices[part_idx] = nn_indices[part_idx].type(torch.int64)
        # dist_to_nn = (dist_to_nn * nn_weights).detach()
        dist_to_nn[part_idx] = dist_to_nn[part_idx].detach()
        # nn_init_positions = deformed_src_pc.detach().clone().to(device)
        nn_init_positions[part_idx] = base_dist_pcs[part_idx].to(device)

    # find the original distances between boundary points
    num_bound_pts = [
        len(boundary_indices[part_idx]) for part_idx in range(total_part_num)
    ]
    # # find the the l2 distance bewtwen each pair of boundary points
    # boundary_dists = torch.norm(
    #     base_dist_pcs[0][boundary_indices[0]].unsqueeze(1).expand(
    #         num_bound_pts[0], num_bound_pts[1], 3
    #     ) - base_dist_pcs[1][boundary_indices[1]].unsqueeze(0).expand(
    #         num_bound_pts[0], num_bound_pts[1], 3
    #     ),
    #     dim=2,
    # )

    if test_transform_net:
        if reg_boundary_test:
            # load the experiment log dir without "_RB"
            transform_net.load_state_dict(
                torch.load(os.path.join(cur_log_dir[:-3], "transform_net.pth"))
            )
        else:
            transform_net.load_state_dict(
                torch.load(os.path.join(cur_log_dir, "transform_net.pth"))
            )
        # visualize_time_steps(pc_images, cur_log_dir, num_steps, n_iter + 1)
        save_all_deformed_pcs()
        # save_deformed_kps()
        exit(0)

    # ===== old optimizer of the deform net =====
    # optimizer = torch.optim.Adam(reg_net.parameters(), lr=0.0005)
    optimizer = torch.optim.Adam(transform_net.parameters(), lr=0.0005)

    bird_body_indices = np.setdiff1d(np.arange(len(tgt_pc)), bird_wing_indices)
    progress_bar = tqdm.tqdm(range(n_iter), desc="Training")

    for i in progress_bar:
        # Compute the target point cloud
        # deform_vecs = reg_net(src_pcs.to(device))
        # deformed_src_pcs = src_pcs.to(device) + deform_vecs

        # total_cd_loss = chamfer_distance(deformed_src_pcs[0:1], tgt_pc.unsqueeze(0))[0]
        if i % 20 == 0:
            visualize_time_steps(pc_images, cur_log_dir, num_steps, i)
            # print(a)
        if i % 100 == 0 or i == 50:
            torch.save(
                transform_net.state_dict(),
                os.path.join(cur_log_dir, "transform_net.pth"),
            )
            imageio.mimsave(
                os.path.join(cur_log_dir, "train_deformed_pc.mp4"), pc_images, fps=10
            )
            save_all_deformed_pcs()
            save_deformed_kps()

        # total_reg_loss = 0
        for j in range(1, num_steps):
            # total_reg_loss = 0
            # total_cd_loss = 0
            # deform_vecs = reg_net(src_pcs.to(device))
            # try reusing the first src_pc as the input to all the deformations
            # deform_vecs = reg_net(src_pcs[0:1].to(device))
            # deformed_src_pcs = src_pcs.to(device) + deform_vecs

            # ===== adding transformation net =====
            # scale, quaternion = transform_net(src_pcs.to(device))
            # rotation_matrix = quaternion_to_matrix(quaternion)
            # deformed_src_pcs = apply_transformation(
            #     init_displacement, scale, rotation_matrix
            # ) + src_pcs.to(device)
            # total_cd_loss = chamfer_distance(deformed_src_pcs[0:1], tgt_pc.unsqueeze(0))[0]

            for part_idx in range(total_part_num):
                if use_keypoints:
                    inp_pcs = key_points
                    base_displacement = ori_kp_deforms
                else:
                    inp_pcs = src_pcs

                num_nn = num_nns[part_idx]
                total_reg_loss = 0
                if regularize_kp_only:
                    cur_tgt_pc = tgt_pcs[part_idx]
                else:
                    if part_idx == 0:
                        cur_tgt_pc = tgt_pc[bird_wing_indices]
                    else:
                        cur_tgt_pc = tgt_pc[bird_body_indices]

                if concat_feature:
                    if input_case == 0 or input_case == 2:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    inp_pcs[part_idx][0].to(device),
                                    pred_normals[part_idx][0].to(device),
                                    pred_normals[part_idx][0].to(device),
                                ],
                                dim=-1,
                            )
                        )
                    elif input_case == 1 or input_case == 3:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    inp_pcs[part_idx][0].to(device),
                                    pred_normals[part_idx][0].to(device),
                                    pred_normals[part_idx][0].to(device),
                                ],
                                dim=-1,
                            ),
                            non_pe=torch.tensor([0])
                            .unsqueeze(-1)
                            .expand(all_num_keypoints[part_idx], -1)
                            .float()
                            .to(device),
                        )
                    else:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    src_pcs[part_idx][0].to(device),
                                    (
                                        deformed_src_pc_start[part_idx]
                                        - src_pcs[part_idx][0].to(device)
                                    )[0].to(device),
                                ],
                                dim=-1,
                            )
                        )
                else:
                    scale, quaternion = transform_net(inp_pcs[0].to(device))
                # concate src_pcs[0] and init_displacement[0] along the last dimension

                rotation_matrix = quaternion_to_matrix(quaternion)
                deformed_src_pcs = apply_transformation(
                    base_displacement[part_idx][0].to(device),
                    scale,
                    rotation_matrix,
                    no_scale=no_scale,
                ) + inp_pcs[part_idx][0].to(device)
                # print(f"Loss: {loss.item()}, Reg Loss: {total_reg_loss.item()}, CD Loss: {total_cd_loss.item()}")
                if use_keypoints:
                    if regularize_kp_only:
                        final_deformed_src_pcs = deformed_src_pcs
                    else:
                        final_deformed_src_pcs = torch.einsum(
                            "ij,ijk->ik",
                            deformed_pc_weights[part_idx],
                            deformed_src_pcs[deformed_nn_inds[part_idx]],
                        )
                else:
                    final_deformed_src_pcs = deformed_src_pcs
                # ===== vanilla loss without keypoint =====
                # rotation_matrix = quaternion_to_matrix(quaternion)
                # deformed_src_pcs = apply_transformation(
                #     init_displacement[0], scale, rotation_matrix, no_scale=no_scale
                # ) + inp_pcs[0].to(device)

                # if regularize_kp_only and not double_side_chamfer:
                #     total_cd_loss = chamfer_distance(
                #         final_deformed_src_pcs.unsqueeze(0), cur_tgt_pc.unsqueeze(0), single_directional=True
                #     )[0]
                # else:
                total_cd_loss = chamfer_distance(
                    final_deformed_src_pcs.unsqueeze(0), cur_tgt_pc.unsqueeze(0)
                )[0]
                # ===== comment out the chamfer distance loss =====
                # cd_loss = chamfer_distance(deformed_src_pcs[j:j+1], tgt_pc.unsqueeze(0))[0]
                # if cd_weight_decay == 'exp':
                #     cd_weight = np.exp(-0.1 * j)
                # elif cd_weight_decay == 'linear':
                #     cd_weight = (num_steps - j) / num_steps
                # elif cd_weight_decay == 'zeros':
                #     cd_weight = 0
                # total_cd_loss += cd_loss * cd_weight
                # ===== comment out the chamfer distance loss =====
                if reg_boundary:
                    inp_pcs = augmented_key_points
                    cur_pred_normals = augmented_pred_normals
                    base_displacement = augmented_ori_kp_deforms
                else:
                    cur_pred_normals = pred_normals
                if concat_feature:
                    if input_case == 0:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    inp_pcs[part_idx][j].to(device),
                                    cur_pred_normals[part_idx][0].to(device),
                                    cur_pred_normals[part_idx][j].to(device),
                                ],
                                dim=-1,
                            )
                        )
                    elif input_case == 1:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    inp_pcs[part_idx][j].to(device),
                                    cur_pred_normals[part_idx][0].to(device),
                                    cur_pred_normals[part_idx][j].to(device),
                                ],
                                dim=-1,
                            ),
                            non_pe=torch.tensor([j])
                            .unsqueeze(-1)
                            .expand(all_num_keypoints[part_idx], -1)
                            .float()
                            .to(device),
                        )
                    elif input_case == 2:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    inp_pcs[part_idx][0].to(device),
                                    cur_pred_normals[part_idx][0].to(device),
                                    cur_pred_normals[part_idx][j].to(device),
                                ],
                                dim=-1,
                            )
                        )
                    elif input_case == 3:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    inp_pcs[part_idx][0].to(device),
                                    cur_pred_normals[part_idx][0].to(device),
                                    cur_pred_normals[part_idx][j].to(device),
                                ],
                                dim=-1,
                            ),
                            non_pe=torch.tensor([j])
                            .unsqueeze(-1)
                            .expand(all_num_keypoints[part_idx], -1)
                            .float()
                            .to(device),
                        )
                    else:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    src_pcs[part_idx][j].to(device),
                                    (
                                        deformed_src_pc_start[part_idx]
                                        - src_pcs[part_idx][0].to(device)
                                    )[0].to(device),
                                ],
                                dim=-1,
                            )
                        )
                else:
                    scale, quaternion = transform_net(inp_pcs[j].to(device))
                # scale, quaternion = transform_net(src_pcs[j].to(device))
                # scale, quaternion = transform_net(torch.cat([src_pcs[j].to(device), init_displacement[0]], dim=-1))

                if use_keypoints:
                    displacement = base_displacement[part_idx][j]
                else:
                    displacement = init_displacement[0]

                rotation_matrix = quaternion_to_matrix(quaternion)
                deformed_src_pcs = apply_transformation(
                    displacement.to(device), scale, rotation_matrix, no_scale=no_scale
                ) + inp_pcs[part_idx][j].to(device)
                if use_keypoints:
                    if regularize_kp_only:
                        cur_deformed_src_pcs = deformed_src_pcs
                    else:
                        cur_deformed_src_pcs = torch.einsum(
                            "ij,ijk->ik",
                            deformed_pc_weights[part_idx],
                            deformed_src_pcs[deformed_nn_inds[part_idx]],
                        )
                else:
                    cur_deformed_src_pcs = deformed_src_pcs

                # rotation_matrix = quaternion_to_matrix(quaternion)
                # deformed_src_pcs = apply_transformation(
                #     init_displacement[0], scale, rotation_matrix, no_scale=no_scale
                # ) + src_pcs[j].to(device)

                total_displacement_after_update = (
                    cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3).expand(
                        num_pts[part_idx], num_nn, 3
                    )
                    - cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3)
                    .expand(num_pts[part_idx], num_pts[part_idx], 3)
                    .gather(
                        0,
                        nn_indices[part_idx][:, :num_nn]
                        .to(device)
                        .unsqueeze(-1)
                        .expand(num_pts[part_idx], num_nn, 3),
                    )
                )
                # total_dist_to_nn_after_update = (
                #     (
                #         cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3).expand(
                #             num_pts[part_idx], num_nn, 3
                #         )
                #         - cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3)
                #         .expand(num_pts[part_idx], num_pts[part_idx], 3)
                #         .gather(
                #             0,
                #             nn_indices[part_idx][:, :num_nn]
                #             .to(device)
                #             .unsqueeze(-1)
                #             .expand(num_pts[part_idx], num_nn, 3),
                #         )
                #         # ).pow(2).sum(dim=2) * nn_weights[:, :num_nn].to(device)
                #     )
                #     .pow(2)
                #     .sum(dim=2)
                # )
                # total_dist_to_nn_after_update = (
                #     deformed_src_pcs[j].view(num_pts, 1, 3).expand(num_pts, num_nn, 3)
                #     - deformed_src_pcs[j]
                #     .view(num_pts, 1, 3)
                #     .expand(num_pts, num_pts, 3)
                #     .gather(
                #         0,
                #         nn_indices[:, :num_nn]
                #         .to(device)
                #         .unsqueeze(-1)
                #         .expand(num_pts, num_nn, 3),
                #     )
                # # ).pow(2).sum(dim=2) * nn_weights[:, :num_nn].to(device)
                # ).pow(2).sum(dim=2)

                if loss_type == "L1":
                    reg_loss = (
                        total_dist_to_nn_after_update.sum(dim=1) - dist_to_nn.to(device)
                    ).abs().sum() / (num_pts[part_idx] * num_nn)
                elif loss_type == "L1_fix":
                    if reg_displacement:
                        reg_loss = (
                            (
                                total_displacement_after_update
                                - dist_to_nn[part_idx][:, :num_nn, :].to(device)
                            )
                        ).abs().sum() / (num_pts[part_idx] * num_nn)
                    else:
                        reg_loss = (
                            total_displacement_after_update.pow(2).sum(dim=2)
                            - dist_to_nn[part_idx][:, :num_nn].to(device)
                        ).abs().sum() / (num_pts[part_idx] * num_nn)
                elif loss_type == "MSE":
                    if reg_displacement:
                        reg_loss = (
                            torch.nn.functional.mse_loss(
                                total_displacement_after_update,
                                dist_to_nn[part_idx][:, :num_nn, :].to(device),
                            )
                            / num_nn
                        )
                    else:
                        reg_loss = torch.nn.functional.mse_loss(
                            total_displacement_after_update.pow(2).sum(dim=2),
                            dist_to_nn[part_idx][:, :num_nn].to(device),
                        )
                elif loss_type == "robust":
                    x = (
                        total_dist_to_nn_after_update
                        - dist_to_nn[part_idx][:, :num_nn].to(device)
                    ) / float(robust_c)
                    reg_loss = (2 * x.pow(2) / (x.pow(2) + 4)).sum() / (
                        num_pts[part_idx] * num_nn
                    )
                elif loss_type == "welsch":
                    x = (
                        total_dist_to_nn_after_update
                        - dist_to_nn[part_idx][:, :num_nn].to(device)
                    ) / float(robust_c)
                    reg_loss = (1.0 - torch.exp(x.pow(2) * -0.5)).sum() / (
                        num_pts[part_idx] * num_nn
                    )

                total_reg_loss += reg_loss * rigid_loss_w

                # ==== add the movement loss ====
                # new_deformed_src_pcs = torch.einsum(
                #     "ij,ijk->ik",
                #     deformed_pc_weights[part_idx],
                #     deformed_src_pcs[deformed_nn_inds[part_idx]],
                # )

                # avg_displacement = (
                #     new_deformed_src_pcs.view(num_pts[part_idx], 1, 3)
                #     .expand(num_pts[part_idx], num_nn, 3)
                #     .gather(
                #         0,
                #         nn_indices[part_idx][:, :num_nn]
                #         .to(device)
                #         .unsqueeze(-1)
                #         .expand(num_pts[part_idx], num_nn, 3),
                #     )
                #     - nn_init_positions[part_idx]
                #     .view(num_pts[part_idx], 1, 3)
                #     .expand(num_pts[part_idx], num_pts[part_idx], 3)
                #     .gather(
                #         0,
                #         nn_indices[part_idx][:, :num_nn]
                #         .to(device)
                #         .unsqueeze(-1)
                #         .expand(num_pts[part_idx], num_nn, 3),
                #     )
                # ).mean(dim=1)
                # avg_displacement_loss = (
                #     (
                #         new_deformed_src_pcs
                #         - (avg_displacement + nn_init_positions[part_idx])
                #     )
                #     .pow(2)
                #     .sum(dim=1)
                # ).mean()

                # ====================== try this version ======================
                # avg_displacement = (
                #     cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3)
                #     .expand(num_pts[part_idx], num_nn, 3)
                #     .gather(
                #         0,
                #         nn_indices[part_idx][:, :num_nn]
                #         .to(device)
                #         .unsqueeze(-1)
                #         .expand(num_pts[part_idx], num_nn, 3),
                #     )
                #     - nn_init_positions[part_idx]
                #     .view(num_pts[part_idx], 1, 3)
                #     .expand(num_pts[part_idx], num_pts[part_idx], 3)
                #     .gather(
                #         0,
                #         nn_indices[part_idx][:, :num_nn]
                #         .to(device)
                #         .unsqueeze(-1)
                #         .expand(num_pts[part_idx], num_nn, 3),
                #     )
                # ).mean(dim=1)
                # avg_displacement_loss = (
                #     (
                #         cur_deformed_src_pcs
                #         - (avg_displacement + nn_init_positions[part_idx])
                #     )
                #     .pow(2)
                #     .sum(dim=1)
                # ).mean()
                # total_reg_loss += avg_displacement_loss * ldas_loss_w

                # ==== add the displacement regularizer ====
                # Design 1: preserve the distance that each point in src travels
                #

                loss = total_reg_loss + cd_loss_w * total_cd_loss
                optimizer.zero_grad()
                # loss.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                # tqdm.tqdm.write(
                #     f"Loss: {loss.item()}, Reg Loss: {total_reg_loss.item()}, CD Loss: {total_cd_loss.item()}"
                # )
            if i % 10 == 0:
                progress_bar.set_description(
                    f"Training (Loss: {loss.item():.4f}) Reg Loss: {total_reg_loss.item():.4f}, CD Loss: {total_cd_loss.item():.4f}"
                )

            # if reg_boundary:
            #     deformed_bound_pts = []
            #     loss = 0.0
            #     for part_idx in range(total_part_num):
            #         cur_tgt_pc = tgt_pcs[part_idx]
            #         if input_case == 0 or input_case == 2:
            #             scale, quaternion = transform_net(
            #                 torch.cat(
            #                     [
            #                         inp_pcs[part_idx][0].to(device),
            #                         pred_normals[part_idx][0].to(device),
            #                         pred_normals[part_idx][0].to(device),
            #                     ],
            #                     dim=-1,
            #                 )
            #             )
            #         rotation_matrix = quaternion_to_matrix(quaternion)
            #         deformed_src_pcs = apply_transformation(
            #             base_displacement[part_idx][0].to(
            #                 device
            #             ),
            #             scale,
            #             rotation_matrix,
            #             no_scale=no_scale,
            #         ) + inp_pcs[part_idx][0].to(device)
            #         loss += chamfer_distance(
            #             deformed_src_pcs.unsqueeze(0), cur_tgt_pc.unsqueeze(0)
            #         )[0]

            #         if input_case == 0:
            #             scale, quaternion = transform_net(
            #                 torch.cat(
            #                     [
            #                         inp_pcs[part_idx][j][boundary_indices[part_idx]].to(
            #                             device
            #                         ),
            #                         pred_normals[part_idx][0][
            #                             boundary_indices[part_idx]
            #                         ].to(device),
            #                         pred_normals[part_idx][j][
            #                             boundary_indices[part_idx]
            #                         ].to(device),
            #                     ],
            #                     dim=-1,
            #                 )
            #             )
            #         rotation_matrix = quaternion_to_matrix(quaternion)
            #         deformed_bound_pts.append(apply_transformation(
            #             base_displacement[part_idx][j][boundary_indices[part_idx]].to(
            #                 device
            #             ),
            #             scale,
            #             rotation_matrix,
            #             no_scale=no_scale,
            #         ) + inp_pcs[part_idx][j][boundary_indices[part_idx]].to(device))

            #     # add boundary loss
            #     cur_boundary_dists = torch.norm(
            #         deformed_bound_pts[0].unsqueeze(1).expand(
            #             num_bound_pts[0], num_bound_pts[1], 3
            #         )
            #         - deformed_bound_pts[1]
            #         .unsqueeze(0)
            #         .expand(num_bound_pts[0], num_bound_pts[1], 3),
            #         dim=2,
            #     )
            #     optimizer.zero_grad()
            #     loss += (cur_boundary_dists - boundary_dists.to(device)).abs().sum() / (
            #         num_bound_pts[0] * num_bound_pts[1]
            #     )
            #     loss.backward()
            #     optimizer.step()

            # loss = total_reg_loss + cd_loss_w * total_cd_loss
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

    # torch.save(reg_net.state_dict(), os.path.join(cur_log_dir, "reg_deform_net.pth"))
    visualize_time_steps(pc_images, cur_log_dir, num_steps, n_iter)
    torch.save(
        transform_net.state_dict(), os.path.join(cur_log_dir, "transform_net.pth")
    )
    imageio.mimsave(os.path.join(cur_log_dir, 'train_deformed_pc.mp4'), pc_images, fps=10)
    # save the deformed pcs
    save_all_deformed_pcs()
