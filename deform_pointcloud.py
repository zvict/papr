
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
log_dir = "deform_pointcloud_logs"
os.makedirs(log_dir, exist_ok=True)

def plot_pointcloud(points, title=""):
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, y, z)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(9, -143)
    # plt.show()
    plt.savefig(os.path.join(log_dir, title + ".png"))
    plt.close()


src_pc_dir = "/NAS/yza629/codes/papr-retarget/point_clouds/butterfly"
deform_vectors_path = "/NAS/yza629/codes/papr-retarget/fit_pointcloud_logs/deform_points.pth"

deform_vectors = torch.load(deform_vectors_path)

start = 0
end = 30000
interval = 1000

for idx in tqdm.tqdm(range(start, end, interval)):
    src_pc_path = os.path.join(src_pc_dir, f"points_{idx}.npy")
    src_pc = np.load(src_pc_path)

    src_pc = torch.tensor(src_pc).float().to(device)

    scale = 10.0
    src_pc = src_pc / scale
    # print("src_pc: ", src_pc.shape, src_pc.min(), src_pc.max())
    
    # Deform the point cloud
    deformed_pc = src_pc + deform_vectors

    # Plot the point clouds
    plot_pointcloud(src_pc, title=f"frame_{idx}_source")
    plot_pointcloud(deformed_pc, title=f"frame_{idx}_deformed")
