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
from scipy.spatial import KDTree


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
    def __init__(self, input_dim, L=0):
        super(AffineTransformationNet, self).__init__()
        self.pose_enc = PoseEnc()
        self.L = L
        in_dim = input_dim + input_dim * 2 * L
        self.fc1 = nn.Linear(in_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)  # 1 for scale, 4 for quaternion

    def forward(self, x):
        x = self.pose_enc(x, self.L)
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
        normals[t] = estimate_surface_normals(point_clouds[t], num_nn)

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


USE_ICP = True
USE_FPS = False
# TRAIN_DEFORM_NET = True
TRAIN_DEFORM_NET = False
TEST_DEFORM_NET = False
REG_DEFORM_NET = True
# REG_DEFORM_NET = False
# TEST_REG_DEFORM_NET = True
USE_POINTNET = True
DENSIFY_BODY = True
# DENSIFY_BODY = False
CONCAT_POS_DEFORM = True


fps_k = 1000
scale = 10.0
num_layers = 3
hidden_dim = 256
L = 8
# L = 0
log_dir = "fit_pointcloud_logs"
if USE_POINTNET:
    # if DENSIFY_BODY:
    #     exp_dir = f"multi_mlp_icp_shift_pe{L}_pointnet_densify"
    # elif CONCAT_POS_DEFORM:
    #     exp_dir = f"multi_mlp_icp_shift_pe{L}_pointnet_concat"
    # else:
    exp_dir = f'multi_mlp_icp_shift_pe{L}_pointnet'
    if DENSIFY_BODY:
        exp_dir += "_densify"
    if CONCAT_POS_DEFORM:
        exp_dir += "_concat"
else:
    exp_dir = f"multi_mlp_icp_shift_pe{L}"
# exp_dir = f'learn_mlp_icp_shift_pe{L}_pointnet'
log_dir = os.path.join(log_dir, exp_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

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

if USE_FPS:
    # FPS
    src_pc = sfp(src_pc.unsqueeze(0), K=fps_k)[0].squeeze(0)
    tgt_pc = sfp(tgt_pc.unsqueeze(0), K=fps_k)[0].squeeze(0)
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

# net = DeformNet(3, 3, hidden_dim=hidden_dim, num_layers=num_layers, L=L).to(device)


# deform_points = torch.full(src_pc.shape, 0.0, device=device, requires_grad=True)

if TRAIN_DEFORM_NET:
    total_result = []
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

    # for part_idx in range(2):
    for part_idx in range(1, 2):

        if USE_POINTNET:
            if CONCAT_POS_DEFORM:
                net = DeformNet(2048, 3, hidden_dim=hidden_dim, num_layers=num_layers, L=L, pt_dim=3).to(
                    device
                )
            else:
                net = DeformNet(2048, 3, hidden_dim=hidden_dim, num_layers=num_layers, L=L).to(
                    device
                )
        else:
            net = DeformNet(3, 3, hidden_dim=hidden_dim, num_layers=num_layers, L=L).to(device)

        if part_idx == 0:
            cur_src_pc = src_pc[but_wing_indices]
            cur_point_feat = point_feat[but_wing_indices]
            cur_tgt_pc = tgt_pc[bird_wing_indices]
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
            # if DENSIFY_BODY:
            #     # add points to the body of cur_src_pc by inserting one point between each pair of nearest neighbors
            #     cur_src_pc_np = cur_src_pc.cpu().numpy()
            #     tree = KDTree(cur_src_pc_np)
            #     nn_dists, nn_inds = tree.query(cur_src_pc_np, k=2)
            #     new_points = (cur_src_pc[nn_inds].sum(1) / 2).cpu()
            #     cur_src_pc = torch.cat([cur_src_pc, new_points], 0)

        # optimizer = torch.optim.Adam([deform_points], lr=0.001)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
        pc_images = []
        n_iter = 2000 if part_idx == 0 else 10000
        for i in tqdm.tqdm(range(n_iter)):
            optimizer.zero_grad()

            # Compute the target point cloud
            if USE_POINTNET:
                if CONCAT_POS_DEFORM:
                    deform_points = net(cur_point_feat, pt=cur_src_pc)
                else:
                    deform_points = net(cur_point_feat)
            else:
                deform_points = net(cur_src_pc)
            deformed_src_pc = cur_src_pc + deform_points
            # deformed_src_pc = net(src_pc)

            # Compute the loss
            loss, _ = chamfer_distance(
                deformed_src_pc.unsqueeze(0), cur_tgt_pc.unsqueeze(0)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Loss: {loss.item()}")
                pc_img = plot_pointcloud(
                    deformed_src_pc,
                    log_dir,
                    title=f"Part {part_idx} Deformed Point Cloud Iter {i}",
                )
                pc_images.append(pc_img)

        total_result.append(deformed_src_pc)

        torch.save(
            net.state_dict(), os.path.join(log_dir, f"deform_net_{part_idx}.pth")
        )
        imageio.mimsave(
            os.path.join(log_dir, f"train_deformed_pc_{part_idx}.mp4"), pc_images, fps=10
        )
    total_result = torch.cat(total_result, dim=0)
    pc_img = plot_pointcloud(
        total_result,
        log_dir,
        title=f"Total Final Deformed Point Cloud",
    )


""" 
Deform the point cloud
"""

if TEST_DEFORM_NET:
    src_pc_dir = "/NAS/yza629/codes/papr-retarget/point_clouds/butterfly"
    deform_vectors_path = "/NAS/yza629/codes/papr-retarget/fit_pointcloud_logs/deform_points.pth"

    cur_log_dir = os.path.join(log_dir, 'test_deformed_pc')
    os.makedirs(cur_log_dir, exist_ok=True)

    # deform_vectors = torch.load(deform_vectors_path)

    start = 0
    end = 30001
    interval = 1000
    src_pcs = []
    deformed_pcs = []
    for idx in tqdm.tqdm(range(start, end, interval)):
        src_pc_path = os.path.join(src_pc_dir, f"points_{idx}.npy")
        src_pc = np.load(src_pc_path)

        src_pc = torch.tensor(src_pc).float().to(device)

        scale = 10.0
        src_pc = src_pc / scale
        # print("src_pc: ", src_pc.shape, src_pc.min(), src_pc.max())

        if idx == start:
            t0_src_pc = src_pc.clone()

        # src_pc = sfp(src_pc.unsqueeze(0), K=1000)[0].squeeze(0)
        
        # Deform the point cloud
        with torch.no_grad():
            deform_points = net(src_pc)
            deformed_pc = src_pc + deform_points
            # deformed_pc = t0_src_pc + deform_points
            # deformed_pc = net(src_pc)

        if idx == start:
            t0_deformed_pc = deformed_pc.clone()

        # Plot the point clouds
        src_pc_plot = plot_pointcloud(src_pc, cur_log_dir, title=f"frame_{idx}_source")
        deformed_pc_plot = plot_pointcloud(deformed_pc, cur_log_dir, title=f"frame_{idx}_deformed")

        src_pcs.append(src_pc_plot)
        deformed_pcs.append(deformed_pc_plot)

    imageio.mimsave(os.path.join(cur_log_dir, 'test_source_pc.mp4'), src_pcs, fps=10)
    imageio.mimsave(os.path.join(cur_log_dir, 'test_deformed_pc.mp4'), deformed_pcs, fps=10)


if REG_DEFORM_NET:
    """
    Regularize the point cloud
    """

    n_iter = 2000
    batch_size = 10000
    robust_c = 0.2
    cd_loss_w = 1000.0
    # cd_loss_w = 1.0
    rigid_loss_w = 1.0
    # rigid_loss_w = 10.0
    ldas_loss_w = 0.0
    # cd_weight_decay = 'zeros'
    src_pc_dir = "/NAS/spa176/papr-retarget/point_clouds/butterfly"

    num_nn = 100
    # num_nn = 50
    concat_feature = True
    input_case = 0
    no_scale = False
    rotate_normal = True
    # add key point
    use_keypoints = True
    # use_keypoints = False
    flip_normal = True
    num_keypoints = 64
    # num_keypoints = 32
    # num_keypoints = 128
    # num_keypoints_body = 64
    # num_keypoints_body = 128
    num_keypoints_body = 96
    kp_knn = 5
    # kp_knn_body = 5
    # kp_knn_body = 50
    # kp_knn = 20
    kp_knn_body = 20
    # num_keypoints = 1024
    # kp_knn = 5
    total_part_num = 2

    test_transform_net = True
    # test_transform_net = False

    transform_L = 0
    loss_type = "L1_fix"
    # cur_log_dir = os.path.join(log_dir, f'test_deformed_pc_regularized_cdw{cd_loss_w}_wdecay{cd_weight_decay}')
    if concat_feature:
        if use_keypoints:
            cur_log_dir = os.path.join(
                log_dir,
                f"transform_wing_kp_{num_keypoints}_body_kp_{num_keypoints_body}_kpnn_{kp_knn}_bodykpnn_{kp_knn_body}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nn{num_nn}_concat_{input_case}",
                # f"test_deformed_pc",
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

    elif no_scale:
        cur_log_dir = os.path.join(
            log_dir,
            f"transform_L{transform_L}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nn{num_nn}_no_scale",
            # f"test_deformed_pc",
        )
    elif use_keypoints:
        if flip_normal:
            cur_log_dir = os.path.join(
                log_dir,
                f"transform_keypoint_num_{num_keypoints}_kpnn_{kp_knn}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nn{num_nn}_flip",
                # f"test_deformed_pc",
            )
        else:
            cur_log_dir = os.path.join(
                log_dir,
                f"keypoint_num_{num_keypoints}_kpnn_{kp_knn}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nn{num_nn}",
                # f"test_deformed_pc",
            )
    elif rotate_normal:
        cur_log_dir = os.path.join(
            log_dir,
            f"transform_L{transform_L}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nn{num_nn}_rotate_normal",
            # f"test_deformed_pc",
        )
    else:
        cur_log_dir = os.path.join(
            log_dir,
            f"transform_L{transform_L}_cdw{cd_loss_w}_rigidw{rigid_loss_w}_ldasw{ldas_loss_w}_nn{num_nn}",
            # f"test_deformed_pc",
        )
    os.makedirs(cur_log_dir, exist_ok=True)

    # copy current file to the log directory
    subprocess.run(["cp", __file__, cur_log_dir])

    # deform_vectors = torch.load(deform_vectors_path)

    start = 0
    end = 30001
    interval = 1000

    src_pcs = [[] for _ in range(total_part_num)]
    deformed_pcs = [[] for _ in range(total_part_num)]
    if use_keypoints:
        kp_indices = [[] for _ in range(total_part_num)]
        init_kps   = [[] for _ in range(total_part_num)]
        key_points = [[] for _ in range(total_part_num)]

    ########## Load the point cloud by parts ##########
    ########## Optionally add key points ##########
    for idx in tqdm.tqdm(range(start, end, interval)):
        if idx > 0:
            src_pc_path = os.path.join(src_pc_dir, f"points_{idx}.npy")
            src_pc = np.load(src_pc_path)

            # src_pc = torch.tensor(src_pc).float().to(device)
            src_pc = torch.tensor(src_pc).float()

            scale = 10.0
            src_pc = src_pc / scale
        else:
            src_pc = src_pc.cpu()

        for part_idx in range(total_part_num):
            if part_idx == 0:
                cur_src_pc = src_pc[but_wing_indices]
            else:
                if DENSIFY_BODY and idx == start:
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

            if idx == start:
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
                    init_kps[part_idx], kp_indices[part_idx] = sfp(
                        cur_src_pc.unsqueeze(0), K=cur_num_keypoints
                    )

            if use_keypoints:
                key_points[part_idx].append(
                    cur_src_pc[kp_indices[part_idx][0]].cpu().clone()
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

    # num_steps = src_pcs[0].shape[0]
    if use_keypoints:
        num_steps = key_points[0].shape[0]
    else:
        num_steps = src_pcs[0].shape[0]
    num_pts = [src_pcs[part_idx].shape[1] for part_idx in range(total_part_num)]
    # num_steps, num_pts, _ = src_pcs.shape
    dist_to_nn = [torch.empty(num_pts[part_idx], num_nn) for part_idx in range(total_part_num)]
    nn_indices = [torch.empty(num_pts[part_idx], num_nn) for part_idx in range(total_part_num)]
    nn_weights = [torch.empty(num_pts[part_idx], num_nn) for part_idx in range(total_part_num)]
    nn_init_positions = [torch.empty(num_pts[part_idx], num_nn, 3) for part_idx in range(total_part_num)]
    # dist_to_nn = torch.empty(num_pts, num_nn)
    # nn_indices = torch.empty(num_pts, num_nn)
    # nn_weights = torch.empty(num_pts, num_nn)

    if USE_POINTNET:
        if CONCAT_POS_DEFORM:
            reg_net = DeformNet(
                2048, 3, hidden_dim=hidden_dim, num_layers=num_layers, L=L, pt_dim=3
            ).to(device)
        else:
            reg_net = DeformNet(2048, 3, hidden_dim=hidden_dim, num_layers=num_layers, L=L).to(device)
    else:
        reg_net = DeformNet(3, 3, hidden_dim=hidden_dim, num_layers=num_layers, L=L).to(device)

    pc_images = []
    deformed_src_pc_start = []

    with torch.no_grad():
        if USE_POINTNET:
            pointnet = pointnet(normal_channel=False).to(device)
            checkpoint = torch.load("best_model.pth")
            pointnet.load_state_dict(checkpoint["model_state_dict"])
            pointnet.eval()
            point_feat = pointnet(
                pc_normalize(full_src_pc.to(device)).unsqueeze(0).transpose(2, 1)
            ).transpose(2, 1)[0]
        #     init_displacement = reg_net(point_feat.unsqueeze(0))
        # else:
        #     init_displacement = reg_net(src_pcs[0:1].to(device))
        for part_idx in range(total_part_num):
            reg_net.load_state_dict(torch.load(os.path.join(log_dir, f"deform_net_{part_idx}.pth")))

            if part_idx == 0:
                cur_src_pc = full_src_pc[but_wing_indices]
                cur_point_feat = point_feat[but_wing_indices]
            else:
                # but_body_indices = np.setdiff1d(np.arange(len(src_pc)), but_wing_indices)
                if DENSIFY_BODY:
                    cur_src_pc = full_src_pc[but_body_indices_concat]
                    cur_point_feat = point_feat[but_body_indices_concat]
                else:
                    cur_src_pc = full_src_pc[but_body_indices]
                    cur_point_feat = point_feat[but_body_indices]

            if USE_POINTNET:
                if CONCAT_POS_DEFORM:
                    init_displacement = reg_net(cur_point_feat, pt=cur_src_pc.to(device))
                else:
                    init_displacement = reg_net(cur_point_feat)
            else:
                init_displacement = reg_net(src_pcs[part_idx].to(device))
            # deformed_src_pc = src_pcs[part_idx] + init_displacement
            # deformed_src_pc_start.append(src_pcs[part_idx][0].cpu() + init_displacement.detach().cpu().clone())
            deformed_src_pc_start.append(
                src_pcs[part_idx][0].to(device) + init_displacement.detach().clone()
            )
        # deformed_src_pc = src_pcs[0] + init_displacement[0].detach().cpu().clone()
        # deformed_src_pc = (
        #     src_pcs[0] + reg_net(src_pcs[0].to(device)).detach().cpu().clone()
        # )

    # release the memory of reg_net
    del reg_net
    if USE_POINTNET:
        del pointnet

    sample_steps = torch.arange(0, num_steps + 1, 5)

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
                deformed_src_pc_start[part_idx],
                deformed_src_pc_start[part_idx][kp_indices[part_idx][0]],
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

            normal_rotation_matrices, pred_normal = (
                compute_rotation_matrices_for_batch(
                    key_points[part_idx], 10, flip_normal=flip_normal
                )
            )
            pred_normals.append(pred_normal)
            ori_kp_deform = torch.matmul(
                normal_rotation_matrices,
                (deformed_src_pc_start[part_idx] - src_pcs[part_idx][0].to(device))[
                    kp_indices[part_idx][0]
                ]
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
        if input_case == 0:
            transform_net = AffineTransformationNet(9, L=transform_L).to(device)
        elif input_case == 1:
            transform_net = AffineTransformationNet(9, L=transform_L).to(device)
        else:
            transform_net = AffineTransformationNet(6, L=transform_L).to(device)
    else:
        transform_net = AffineTransformationNet(3, L=transform_L).to(device)

    def visualize_time_steps(pc_images, cur_log_dir, num_steps, cur_iter):
        with torch.no_grad():
            sample_steps = torch.arange(0, num_steps + 1, 5)

            total_deformed_pc = []

            for part_idx in range(total_part_num):

                if use_keypoints:
                    pc_inp = key_points[part_idx][sample_steps]
                    vis_base_displacement = vis_ori_kp_deforms[part_idx]
                else:
                    pc_inp = src_pcs[sample_steps]
                    vis_base_displacement = init_displacement

                if concat_feature:
                    if input_case == 0:
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
                ) + pc_inp.to(device)
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

            for part_idx in range(total_part_num):
                if use_keypoints:
                    pc_inp = key_points[part_idx]
                    vis_base_displacement = ori_kp_deforms[part_idx]
                else:
                    pc_inp = src_pcs[sample_steps]
                    vis_base_displacement = init_displacement

                if concat_feature:
                    if input_case == 0:
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
                ) + pc_inp.to(device)
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
            total_deformed_pc = torch.cat(total_deformed_pc, dim=1) # shape (num_steps, num_pts, 3)

        # save the total deformed point cloud as a pytorch tensor
        torch.save(total_deformed_pc, os.path.join(cur_log_dir, "total_deformed_pc.pth"))
        # save the initial deformed point cloud at start state
        torch.save(
            torch.cat(deformed_src_pc_start, dim=0),
            os.path.join(cur_log_dir, "deformed_src_pc_start.pth"),
        )

    if test_transform_net:
        transform_net.load_state_dict(
            torch.load(os.path.join(cur_log_dir, "transform_net.pth"))
        )
        # visualize_time_steps(pc_images, cur_log_dir, num_steps, n_iter + 1)
        save_all_deformed_pcs()
        exit(0)
        print(a)

    # ===== old optimizer of the deform net =====
    # optimizer = torch.optim.Adam(reg_net.parameters(), lr=0.0005)
    optimizer = torch.optim.Adam(transform_net.parameters(), lr=0.0005)

    if use_keypoints:
        base_dist_pcs = pred_deformed_pcs
        base_displacement = ori_kp_deforms
    else:
        base_dist_pcs = deformed_src_pc
        base_displacement = init_displacement

    for part_idx in range(total_part_num):
        for pt_idx in range(num_pts[part_idx]):
            # find the distance from the point at index i to all others points
            dist_to_all_pts = (
                (
                    base_dist_pcs[part_idx][pt_idx : pt_idx + 1, :].expand(
                        num_pts[part_idx], 3
                    )
                    - base_dist_pcs[part_idx]
                )
                .pow(2)
                .sum(dim=1)
            )
            vals, inds = torch.topk(
                dist_to_all_pts, num_nn + 1, largest=False, sorted=True
            )
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
        if i % 100 == 0:
            torch.save(
                transform_net.state_dict(),
                os.path.join(cur_log_dir, "transform_net.pth"),
            )
            imageio.mimsave(
                os.path.join(cur_log_dir, "train_deformed_pc.mp4"), pc_images, fps=10
            )

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

            if use_keypoints:
                inp_pcs = key_points
            else:
                inp_pcs = src_pcs

            for part_idx in range(total_part_num):
                total_reg_loss = 0
                if part_idx == 0:
                    cur_tgt_pc = tgt_pc[bird_wing_indices]
                else:
                    cur_tgt_pc = tgt_pc[bird_body_indices]

                if concat_feature:
                    if input_case == 0:
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
                if concat_feature:
                    if input_case == 0:
                        scale, quaternion = transform_net(
                            torch.cat(
                                [
                                    inp_pcs[part_idx][j].to(device),
                                    pred_normals[part_idx][0].to(device),
                                    pred_normals[part_idx][j].to(device),
                                ],
                                dim=-1,
                            )
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
                total_dist_to_nn_after_update = (
                    (
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
                        # ).pow(2).sum(dim=2) * nn_weights[:, :num_nn].to(device)
                    )
                    .pow(2)
                    .sum(dim=2)
                )
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
                    reg_loss = (
                        total_dist_to_nn_after_update
                        - dist_to_nn[part_idx][:, :num_nn].to(device)
                    ).abs().sum() / (num_pts[part_idx] * num_nn)
                elif loss_type == "MSE":
                    reg_loss = torch.nn.functional.mse_loss(
                        total_dist_to_nn_after_update,
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

                avg_displacement = (
                    cur_deformed_src_pcs.view(num_pts[part_idx], 1, 3)
                    .expand(num_pts[part_idx], num_nn, 3)
                    .gather(
                        0,
                        nn_indices[part_idx][:, :num_nn]
                        .to(device)
                        .unsqueeze(-1)
                        .expand(num_pts[part_idx], num_nn, 3),
                    )
                    - nn_init_positions[part_idx]
                    .view(num_pts[part_idx], 1, 3)
                    .expand(num_pts[part_idx], num_pts[part_idx], 3)
                    .gather(
                        0,
                        nn_indices[part_idx][:, :num_nn]
                        .to(device)
                        .unsqueeze(-1)
                        .expand(num_pts[part_idx], num_nn, 3),
                    )
                ).mean(dim=1)
                avg_displacement_loss = (
                    (
                        cur_deformed_src_pcs
                        - (avg_displacement + nn_init_positions[part_idx])
                    )
                    .pow(2)
                    .sum(dim=1)
                ).mean()
                total_reg_loss += avg_displacement_loss * ldas_loss_w

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
