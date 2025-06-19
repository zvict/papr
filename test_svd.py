import os
import torch
from torch import nn
import torch.nn.functional as F
from pytorch3d.ops import sample_farthest_points as sfp
import numpy as np
import matplotlib.pyplot as plt
import imageio
import trimesh # For loading meshes

# --- Constants and Setup ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PART_NAME = "Head" # Part to align
NUM_KEYPOINTS = 64
LEARNING_RATE = 1e-2 # Adjusted for potentially faster convergence on this simpler task
ITERATIONS = 200
PLOT_EVERY_N_ITERATIONS = 10

# --- SVDAlignmentLoss Class (Copied from your provided file) ---
class SVDAlignmentLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, keypoints_a, joint_a, keypoints_b, joint_b):
        if keypoints_a.shape[0] == 0 or keypoints_b.shape[0] == 0:
            return torch.tensor(0.0, device=keypoints_a.device, dtype=keypoints_a.dtype)
        if keypoints_a.shape != keypoints_b.shape:
            # Allow for broadcasting if one of them has an extra batch dim for some reason
            if keypoints_a.ndim == keypoints_b.ndim + 1 and keypoints_a.shape[0] == 1:
                keypoints_a = keypoints_a.squeeze(0)
            elif keypoints_b.ndim == keypoints_a.ndim + 1 and keypoints_b.shape[0] == 1:
                keypoints_b = keypoints_b.squeeze(0)
            else:
                raise ValueError(f"Keypoint sets A ({keypoints_a.shape}) and B ({keypoints_b.shape}) must have the same shape or be broadcastable.")


        j_a = joint_a.view(1, 3)
        j_b = joint_b.view(1, 3)

        vecs_a = keypoints_a - j_a
        vecs_b = keypoints_b - j_b

        norm_vecs_a = F.normalize(vecs_a, p=2, dim=1, eps=self.epsilon)
        norm_vecs_b = F.normalize(vecs_b, p=2, dim=1, eps=self.epsilon)

        H = torch.matmul(norm_vecs_a.T, norm_vecs_b)

        try:
            U, S, Vh = torch.linalg.svd(H)
            V = Vh.T
        except torch.linalg.LinAlgError:
            diff = norm_vecs_a - norm_vecs_b
            loss = torch.sum(diff * diff)
            return loss

        R = torch.matmul(V, U.T)
        if torch.linalg.det(R) < 0:
            V_corrected = V.clone()
            V_corrected[:, -1] *= -1
            R = torch.matmul(V_corrected, U.T)

        rotated_norm_vecs_b = torch.matmul(R, norm_vecs_b.T).T
        diff_after_rotation = norm_vecs_a - rotated_norm_vecs_b
        loss = torch.sum(diff_after_rotation * diff_after_rotation)
        return loss

# --- Visualization Function ---
def plot_alignment_step(pts_a, j_a, pts_b, j_b, iteration, save_dir="svd_loss_test_plots"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Detach and move to CPU for plotting
    pts_a_np, j_a_np = pts_a.detach().cpu().numpy(), j_a.detach().cpu().numpy()
    pts_b_np, j_b_np = pts_b.detach().cpu().numpy(), j_b.detach().cpu().numpy()

    ax.scatter(pts_a_np[:, 0], pts_a_np[:, 1], pts_a_np[:, 2], c='blue', label='Keypoints A (Target)', s=20)
    ax.scatter(j_a_np[0], j_a_np[1], j_a_np[2], c='cyan', marker='X', s=200, label='Joint A', edgecolors='black')

    ax.scatter(pts_b_np[:, 0], pts_b_np[:, 1], pts_b_np[:, 2], c='red', label='Keypoints B (Learnable)', s=20)
    ax.scatter(j_b_np[0], j_b_np[1], j_b_np[2], c='magenta', marker='P', s=200, label='Joint B', edgecolors='black')
    
    # For visualizing relative vectors (optional, can be cluttered)
    # for i in range(pts_a_np.shape[0]):
    #     ax.plot([j_a_np[0], pts_a_np[i,0]], [j_a_np[1], pts_a_np[i,1]], [j_a_np[2], pts_a_np[i,2]], 'b-', alpha=0.3)
    # for i in range(pts_b_np.shape[0]):
    #     ax.plot([j_b_np[0], pts_b_np[i,0]], [j_b_np[1], pts_b_np[i,1]], [j_b_np[2], pts_b_np[i,2]], 'r-', alpha=0.3)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"SVD Alignment - Iteration {iteration}")
    ax.legend()

    # Consistent view and limits
    all_pts = np.vstack((pts_a_np, pts_b_np, j_a_np.reshape(1,3), j_b_np.reshape(1,3)))
    center = all_pts.mean(axis=0)
    max_range = (all_pts.max(axis=0) - all_pts.min(axis=0)).max() / 2.0 * 1.2 # Add some padding
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    ax.view_init(elev=20, azim=45)


    filepath = os.path.join(save_dir, f"frame_{iteration:04d}.png")
    plt.savefig(filepath)
    plt.close(fig)
    return filepath

# --- Main Script ---
if __name__ == "__main__":
    # 1. Load Meshes and Part Indices
    reference_character = "Ch24_nonPBR"
    deforming_character = "Ch17_nonPBR"

    # Assume these files are in a 'data' subdirectory relative to the script
    # You might need to adjust paths
    data_base_path = "/NAS/spa176/" # Or your specific data path
    
    try:
        ref_obj_path = os.path.join(data_base_path, f"data/Mixamo/Animated/{reference_character}/0001.obj") # Using a T-pose/rest frame
        ref_indices_path = os.path.join(data_base_path, f"data/Mixamo/Segmentation/{reference_character}_semantic_masks.npz")
        
        deform_verts_faces_path = os.path.join(data_base_path, f"data/Mixamo/Segmentation/{deforming_character}_vertex_faces.npz")
        deform_indices_path = os.path.join(data_base_path, f"data/Mixamo/Segmentation/{deforming_character}_semantic_masks.npz")

        ref_mesh = trimesh.load(ref_obj_path, process=False)
        ref_vertices_all = torch.tensor(ref_mesh.vertices, dtype=torch.float32, device=DEVICE)
        reference_part_indices = np.load(ref_indices_path)

        deform_data = np.load(deform_verts_faces_path)
        deform_vertices_all = torch.tensor(deform_data['vertices'], dtype=torch.float32, device=DEVICE)
        deforming_part_indices = np.load(deform_indices_path)

    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please ensure the data paths are correct and files exist.")
        print("Expected structure might be: ./data/Mixamo/...")
        exit()

    # 2. Extract Keypoints for the chosen part
    ref_part_vert_indices = torch.tensor(reference_part_indices[PART_NAME], dtype=torch.long, device=DEVICE)
    deform_part_vert_indices = torch.tensor(deforming_part_indices[PART_NAME], dtype=torch.long, device=DEVICE)

    keypoints_a_full = ref_vertices_all[ref_part_vert_indices]
    keypoints_b_full = deform_vertices_all[deform_part_vert_indices]

    if keypoints_a_full.shape[0] < NUM_KEYPOINTS or keypoints_b_full.shape[0] < NUM_KEYPOINTS:
        print(f"Warning: Part '{PART_NAME}' has fewer than {NUM_KEYPOINTS} vertices.")
        print(f"Ref part has {keypoints_a_full.shape[0]}, Deform part has {keypoints_b_full.shape[0]}")
        # Adjust NUM_KEYPOINTS or handle this case (e.g., skip if too few points)
        NUM_KEYPOINTS = min(keypoints_a_full.shape[0], keypoints_b_full.shape[0])
        if NUM_KEYPOINTS == 0:
            print("Error: Zero keypoints for the selected part. Exiting.")
            exit()


    keypoints_a_target, _ = sfp(keypoints_a_full.unsqueeze(0), K=NUM_KEYPOINTS)
    keypoints_a_target = keypoints_a_target.squeeze(0)

    keypoints_b_initial, _ = sfp(keypoints_b_full.unsqueeze(0), K=NUM_KEYPOINTS)
    keypoints_b_initial = keypoints_b_initial.squeeze(0)

    # 3. Initialize Joints and Learnable Parameters
    joint_a_target = keypoints_a_target.mean(dim=0) # Centroid as joint

    # Learnable parameters for B
    keypoints_b_learnable = nn.Parameter(keypoints_b_initial.clone())
    joint_b_learnable = nn.Parameter(joint_a_target.clone()) # Initialize joint_b at joint_a's position

    # 4. Optimizer and Loss
    optimizer = torch.optim.Adam([keypoints_b_learnable, joint_b_learnable], lr=LEARNING_RATE)
    criterion = SVDAlignmentLoss()

    # 5. Optimization Loop
    frame_files = []
    print(f"Starting optimization to align '{PART_NAME}'...")
    for i in range(ITERATIONS):
        optimizer.zero_grad()
        loss = criterion(keypoints_a_target, joint_a_target, keypoints_b_learnable, joint_b_learnable)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Iteration {i}/{ITERATIONS}, Loss: {loss.item():.6f}")

        if i % PLOT_EVERY_N_ITERATIONS == 0 or i == ITERATIONS - 1:
            frame_path = plot_alignment_step(keypoints_a_target, joint_a_target,
                                             keypoints_b_learnable, joint_b_learnable,
                                             i)
            frame_files.append(frame_path)

    # 6. Create Animation
    output_animation_path = "svd_loss_alignment_animation.mp4"
    with imageio.get_writer(output_animation_path, mode='I', fps=10) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)
    print(f"Optimization finished. Animation saved to {output_animation_path}")
    print(f"Individual frames saved in 'svd_loss_test_plots' directory.")
