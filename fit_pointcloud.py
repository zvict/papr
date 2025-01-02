import os
import sys
import torch
import subprocess
import pytorch3d
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl
import tqdm

device = torch.device("cuda:0")
log_dir = "fit_pointcloud_logs"
exp_dir = 'learn_params_shift'
log_dir = os.path.join(log_dir, exp_dir)
os.makedirs(log_dir, exist_ok=True)

def plot_pointcloud(points, title=""):
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    # Use z axis as colors
    colors = x
    ax.scatter3D(x, y, z, c=colors, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(9, -143)
    # plt.show()
    plt.savefig(os.path.join(log_dir, title + ".png"))


src_pc_path = "/NAS/yza629/codes/papr-retarget/point_clouds/butterfly/points_0.npy"
tgt_pc_path = "/NAS/yza629/codes/papr-retarget/point_clouds/hummingbird/points_0.npy"

src_pc = np.load(src_pc_path)
tgt_pc = np.load(tgt_pc_path)

src_pc = torch.tensor(src_pc).float().to(device)
tgt_pc = torch.tensor(tgt_pc).float().to(device)

scale = 10.0
src_pc = src_pc / scale
tgt_pc = tgt_pc / scale
print("src_pc: ", src_pc.shape, src_pc.min(), src_pc.max())
print("tgt_pc: ", tgt_pc.shape, tgt_pc.min(), tgt_pc.max())

# Plot the point clouds
plot_pointcloud(src_pc, title="Source Point Cloud")
plot_pointcloud(tgt_pc, title="Target Point Cloud")


deform_points = torch.full(src_pc.shape, 0.0, device=device, requires_grad=True)

optimizer = torch.optim.Adam([deform_points], lr=0.001)

n_iter = 2000
for i in tqdm.tqdm(range(n_iter)):
    optimizer.zero_grad()

    # Compute the target point cloud
    deformed_src_pc = src_pc + deform_points

    # Compute the loss
    loss, _ = chamfer_distance(deformed_src_pc.unsqueeze(0), tgt_pc.unsqueeze(0))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        print(f"Loss: {loss.item()}")
        plot_pointcloud(deformed_src_pc, title=f"Deformed Point Cloud Iter {i}")

torch.save(deform_points, os.path.join(log_dir, "deform_points.pth"))