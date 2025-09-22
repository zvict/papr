"""
Robust End-to-End Recipe for Mesh Deformation using Optimal Transport.

This script implements a four-stage pipeline to deform a canonical source mesh
to a target pose. It has been updated to use a "match-then-optimize" strategy
with unbalanced Optimal Transport (OT) for a more robust loss function that
preserves mesh connectivity.

The pipeline is designed to be robust and does not rely on perfect mesh
connectivity, making it adaptable for various point clouds and meshes.

Key Features of this version:
- Loss Function: L1 distance based on OT correspondence instead of Chamfer.
- Annealed Segmentation Cost: The influence of segmentation in the OT cost
  decays over time, making the process robust to noisy target labels.
- Rich Cost Matrix: OT cost considers vertex positions, surface normals,
  and semantic part labels.
- Dependencies: numpy, scikit-learn, torch, trimesh, ot, tqdm, imageio.

To run this code:
1. Make sure you have the required libraries:
   pip install numpy scikit-learn torch trimesh ot tqdm imageio
2. Prepare your data:
   - `source_vertices`: (N, 3) numpy array for the canonical mesh.
   - `source_faces`: (F, 3) numpy array for the mesh connectivity.
   - `target_vertices`: (N, 3) numpy array for the posed mesh.
   - `vertex_segmentation`: (N,) numpy array of integer labels for each vertex.
3. Pass the data to the `run_pipeline` function.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.neighbors import kneighbors_graph
import os
import trimesh
import shutil
from tqdm import tqdm
import ot  # Python Optimal Transport library
import plotly.graph_objects as go
from pytorch3d.ops import sample_farthest_points, estimate_pointcloud_normals
from pytorch3d.ops.points_alignment import iterative_closest_point, SimilarityTransform
from geomloss import SamplesLoss  # GPU Sinkhorn OT
import itertools
import math
import matplotlib.cm as cm  # for error colormap export

DEBUG_STAGE3B_ROT_WEIGHT = True

# --- GPU Sinkhorn for correspondence (unbalanced approximation) ---

def sinkhorn_transport_plan(C: torch.Tensor, eps=0.01, tau=0.1, n_iters=40):
    """Compute (approx) unbalanced entropic OT transport plan on GPU.
    C: (N,M) cost matrix.
    eps: entropic regularization.
    tau: unbalanced mass penalty (larger -> closer to balanced). If set to None uses balanced updates.
    Returns transport plan P (N,M) with row/col sums ~ uniform.
    """
    with torch.no_grad():
        N, M = C.shape
        a = torch.full((N,), 1.0 / N, device=C.device)
        b = torch.full((M,), 1.0 / M, device=C.device)
        K = torch.exp(-C / eps)  # (N,M)
        # Avoid zeros
        K = K + 1e-9
        u = torch.ones_like(a)
        v = torch.ones_like(b)
        if tau is None:
            # Balanced Sinkhorn
            for _ in range(n_iters):
                Kv = K @ v
                u = a / (Kv + 1e-9)
                Kt_u = K.t() @ u
                v = b / (Kt_u + 1e-9)
        else:
            # Unbalanced (KL) scaling exponents
            alpha = tau / (tau + eps)
            for _ in range(n_iters):
                Kv = K @ v + 1e-9
                u = (a / Kv).pow(alpha)
                Kt_u = K.t() @ u + 1e-9
                v = (b / Kt_u).pow(alpha)
        P = u.unsqueeze(1) * K * v.unsqueeze(0)
        # Normalise rows (optional) for argmax stability
        row_sums = P.sum(1, keepdim=True) + 1e-9
        P = P / row_sums
        return P

# --- Utility Functions & Modules ---
def _nearest_distance_map(
    target_pts: torch.Tensor, posed_pts: torch.Tensor, batch_size: int = 4096
) -> torch.Tensor:
    """
    For each target point, compute distance to nearest posed point.
    Uses batching over target points to limit memory.
    Returns (Nt,) tensor of distances (float).
    """
    device = target_pts.device
    Nt = target_pts.shape[0]
    posed_pts = posed_pts.detach()
    dists_out = torch.empty(Nt, device=device)
    with torch.no_grad():
        for start in range(0, Nt, batch_size):
            end = min(start + batch_size, Nt)
            tgt_batch = target_pts[start:end]  # (b,3)
            d = torch.cdist(tgt_batch, posed_pts)  # (b, Ns)
            d_min, _ = d.min(dim=1)
            dists_out[start:end] = d_min
    return dists_out


def _export_error_colormap_mesh(
    target_vertices_tensor: torch.Tensor,
    posed_vertices_tensor: torch.Tensor,
    out_path: str,
    faces=None,
    colormap: str = "plasma",
    assume_aligned: bool = True,
):
    """
    Builds a mesh (or point cloud if faces=None) whose vertex colors encode
    nearest distance from each target vertex to the posed (deformed) mesh.
    """
    try:
        import trimesh
    except ImportError:
        print("[ErrorMap] trimesh not available; skipping export.")
        return

    with torch.no_grad():
        if (
            assume_aligned
            and posed_vertices_tensor.shape[0] == target_vertices_tensor.shape[0]
        ):
            dists = (posed_vertices_tensor - target_vertices_tensor).norm(dim=1)
            mode = "aligned L2"
        else:
            dists = _nearest_distance_map(target_vertices_tensor, posed_vertices_tensor)
            mode = "NN"
        d_cpu = dists.detach().cpu()
        d_min = float(d_cpu.min())
        d_max = float(d_cpu.max())
        if d_max - d_min < 1e-12:
            norm = torch.zeros_like(d_cpu)
        else:
            norm = (d_cpu - d_min) / (d_max - d_min + 1e-8)

    cmap = cm.get_cmap(colormap)
    colors = (cmap(norm.numpy()) * 255).astype(np.uint8)  # RGBA
    mesh = trimesh.Trimesh(
        vertices=target_vertices_tensor.detach().cpu().numpy(),
        faces=faces if faces is not None else None,
        process=False,
    )
    mesh.visual.vertex_colors = colors
    mesh.export(out_path)
    print(
        f"[ErrorMap] Exported error mesh '{out_path}' "
        f"(min={d_min:.6f}, max={d_max:.6f})"
    )


def plot_pointcloud(points, save_dir, title="", extra_points=None, matches=None, scale=1.0):
    """
    Generates and saves an interactive 3D plot using Plotly.
    """
    if points.dim() > 2:
        points = points.squeeze(0)
    if extra_points is not None and extra_points.dim() > 2:
        extra_points = extra_points.squeeze(0)

    fig = go.Figure()

    # Add source points
    fig.add_trace(
        go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="markers",
            marker=dict(size=2, color="blue", opacity=0.8),
            name="Source/Posed",
        )
    )

    # Add target points
    if extra_points is not None:
        fig.add_trace(
            go.Scatter3d(
                x=extra_points[:, 0],
                y=extra_points[:, 1],
                z=extra_points[:, 2],
                mode="markers",
                marker=dict(size=2, color="red", opacity=0.8),
                name="Target",
            )
        )

    # Add match lines
    if matches is not None:
        matches_np = matches.cpu().numpy()
        lines_x, lines_y, lines_z = [], [], []
        for line in matches_np:
            lines_x.extend([line[0, 0], line[1, 0], None])
            lines_y.extend([line[0, 1], line[1, 1], None])
            lines_z.extend([line[0, 2], line[1, 2], None])
        fig.add_trace(
            go.Scatter3d(
                x=lines_x,
                y=lines_y,
                z=lines_z,
                mode="lines",
                line=dict(color="green", width=1),
                name="Matches",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title="X", range=[-scale, scale]),
            yaxis=dict(title="Y", range=[-scale, scale]),
            zaxis=dict(title="Z", range=[-scale, scale]),
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_path = os.path.join(save_dir, f"{title}.html")
        fig.write_html(file_path)

    return None # No longer returns an image array


def fps(points, num_samples):
    """
    Farthest Point Sampling using pytorch3d.
    """
    if points.dim() == 2:
        points = points.unsqueeze(0)  # Add batch dimension

    # sample_farthest_points returns indices in a LongTensor of shape (B, K)
    _, indices = sample_farthest_points(points, K=num_samples)

    return indices.squeeze(0)  # Remove batch dimension


def compute_normals(vertices, neighborhood_size=16):
    """
    Computes vertex normals using pytorch3d, based on point cloud neighborhoods.
    """
    if vertices.dim() == 2:
        vertices = vertices.unsqueeze(0)  # Add batch dimension

    normals = estimate_pointcloud_normals(vertices, neighborhood_size=neighborhood_size)

    return normals.squeeze(0)  # Remove batch dimension


def local_scale(pts: torch.Tensor, k: int = 8):
    """
    Simple local scale descriptor: mean distance to k nearest neighbors (excluding self).
    """
    if pts.shape[0] == 0:
        return torch.zeros(0, 1, device=pts.device, dtype=pts.dtype)
    with torch.no_grad():
        d2 = torch.cdist(pts, pts)
        k_eff = min(k, pts.shape[0])
        knn_d2, _ = torch.topk(d2, k_eff, largest=False)
        # exclude self (col 0)
        if k_eff > 1:
            return knn_d2[:, 1:].mean(1, keepdim=True)
        else:
            return torch.zeros(pts.shape[0], 1, device=pts.device, dtype=pts.dtype)


def generate_so3_rotations(num_rots, device="cpu", seed=None):
    """
    Generate a set of initial rotations.
    Strategy:
      1. Start with the 24-element cube (octahedral) rotation group (covers all
         axis permutations with proper sign flips, determinant=1).
      2. If more rotations requested, append random uniform SO(3) samples.
      3. Truncate to exactly num_rots.
    Ensures identity is first.
    """
    if num_rots <= 0:
        return torch.empty(0, 3, 3, device=device)

    # Build cube group (24) rotations
    basis = torch.eye(3)
    mats = []
    for perm in itertools.permutations([0, 1, 2]):           # 6 permutations
        P = basis[list(perm)]
        for signs in itertools.product([1.0, -1.0], repeat=3):  # 8 sign choices
            M = P * torch.tensor(signs).unsqueeze(1)
            if torch.det(M) > 0:  # proper rotation
                mats.append(M)
    # Deduplicate (some permutations/signs collapse)
    cube_group = []
    seen = set()
    for M in mats:
        key = tuple((M.round(decimals=6)).view(-1).tolist())
        if key not in seen:
            seen.add(key)
            cube_group.append(M)
    cube_group = torch.stack(cube_group, dim=0).to(device)  # (24,3,3) typically

    # Ensure identity first
    # Find identity (closest)
    I = torch.eye(3, device=device).unsqueeze(0)
    d = torch.norm(cube_group - I, dim=(1,2))
    idx_id = torch.argmin(d)
    if idx_id != 0:
        cube_group[[0, idx_id]] = cube_group[[idx_id, 0]]

    rotations = cube_group

    # If need more, append random uniform quaternions
    need_extra = max(0, num_rots - rotations.shape[0])
    if need_extra > 0:
        if seed is not None:
            torch.manual_seed(seed)
        # Uniform quaternion sampling (Shoemake)
        u1 = torch.rand(need_extra, device=device)
        u2 = torch.rand(need_extra, device=device)
        u3 = torch.rand(need_extra, device=device)
        q1 = torch.sqrt(1 - u1) * torch.sin(2 * math.pi * u2)
        q2 = torch.sqrt(1 - u1) * torch.cos(2 * math.pi * u2)
        q3 = torch.sqrt(u1) * torch.sin(2 * math.pi * u3)
        q4 = torch.sqrt(u1) * torch.cos(2 * math.pi * u3)
        quats = torch.stack([q4, q1, q2, q3], dim=1)  # (w,x,y,z)
        # Convert to rotation matrices
        w, x, y, z = quats.unbind(1)
        R = torch.stack([
            1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w),
            2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w),
            2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)
        ], dim=1).view(-1,3,3)
        rotations = torch.cat([rotations, R], dim=0)

    # Truncate to requested number
    rotations = rotations[:num_rots]

    return rotations


def _simple_alignment_metric(src_aligned: torch.Tensor,
                             tgt: torch.Tensor,
                             tau_factor: float = 2.5):
    """
    Lightweight robust metric.
    Returns (score, metrics) where score = median - 0.3 * inlier_ratio.
    """
    with torch.no_grad():
        if src_aligned.numel() == 0 or tgt.numel() == 0:
            inf = torch.tensor(float("inf"), device=src_aligned.device)
            return inf, {}
        dmat = torch.cdist(src_aligned, tgt)
        d_st, _ = dmat.min(dim=1)
        median = d_st.median()
        tau = tau_factor * median.clamp(min=1e-6)
        inlier_ratio = (d_st < tau).float().mean()
        score = median - 0.3 * inlier_ratio
        return score, {
            "median": float(median),
            "inlier_ratio": float(inlier_ratio),
            "tau": float(tau),
            "score": float(score)
        }


def greedy_unique_assignment(P: torch.Tensor, nn_idx: torch.Tensor) -> torch.Tensor:
    """
    Enforce near one-to-one matches greedily.
    P: (S, T_full) row-normalized transport plan (after pruning large costs).
       Only entries at candidate indices in nn_idx are meaningful (others large-masked).
    nn_idx: (S, K) candidate target indices (global indexing into columns of P).
    Returns:
        final_idx: (S,) chosen target index per source (may contain limited duplicates
                    if sources > targets or all candidates taken).
    """
    device = P.device
    S, K = nn_idx.shape
    # Gather candidate probabilities for each row
    row_probs = P[torch.arange(S, device=device).unsqueeze(1), nn_idx]  # (S,K)
    # Compute "peakiness" = best - second best probability for ordering
    top2 = torch.topk(row_probs, k=min(2, K), dim=1).values  # (S, <=2)
    if top2.shape[1] == 1:
        peakiness = top2[:, 0]
    else:
        peakiness = top2[:, 0] - top2[:, 1]
    order = torch.argsort(-peakiness)  # most decisive first
    used = torch.zeros(P.shape[1], dtype=torch.bool, device=device)
    final_idx = torch.empty(S, dtype=torch.long, device=device)

    for s in order:
        cands = nn_idx[s]  # (K,)
        probs = row_probs[s]
        # Sort candidates for this source by descending probability
        sort_order = torch.argsort(-probs)
        assigned = None
        for j in sort_order:
            tgt = cands[j].item()
            if not used[tgt]:
                assigned = tgt
                used[tgt] = True
                break
        if assigned is None:
            # All candidates taken -> allow reuse of best
            assigned = cands[sort_order[0]].item()
        final_idx[s] = assigned
    # Revert to original source ordering (order is permutation)
    inv_order = torch.empty_like(order)
    inv_order[order] = torch.arange(S, device=device)
    final_idx = final_idx[inv_order]
    return final_idx


def _fast_rotation_pre_score(src_points: torch.Tensor,
                             tgt_points: torch.Tensor,
                             R: torch.Tensor,
                             subsample: int = 256):
    """
    Quickly estimate suitability of a rotation before full ICP:
    1. Rotate source
    2. Translate both to zero-mean
    3. Compute median NN distance on small subsample
    Lower is better.
    """
    with torch.no_grad():
        if subsample > 0 and src_points.shape[0] > subsample:
            # uniform random (fast); could switch to FPS if desired
            idx = torch.randperm(src_points.shape[0], device=src_points.device)[:subsample]
            src_small = src_points[idx]
        else:
            src_small = src_points
        if subsample > 0 and tgt_points.shape[0] > subsample:
            jdx = torch.randperm(tgt_points.shape[0], device=tgt_points.device)[:subsample]
            tgt_small = tgt_points[jdx]
        else:
            tgt_small = tgt_points
        Rs = R @ src_small.T  # (3,Ns)
        src_rot = Rs.T
        # center
        src_c = src_rot - src_rot.mean(0, keepdim=True)
        tgt_c = tgt_small - tgt_small.mean(0, keepdim=True)
        d = torch.cdist(src_c, tgt_c).min(dim=1)[0]
        return d.median()


def multi_run_icp(src_points,
                  tgt_points,
                  num_inits=24,
                  max_iterations=100,
                  preselect_k=8,
                  subsample_icp=0,
                  early_thresh=0.002):
    """
    Faster multi-init ICP:
      - Pre-score all rotations cheaply.
      - Run ICP only for top-k.
      - Use subsampled sets for ICP (indices shared across all inits).
      - Simple robust metric (median - 0.3*inlier_ratio).
    """
    if src_points.shape[0] == 0 or tgt_points.shape[0] == 0:
        return None

    device = src_points.device
    # Shared subsample for ICP
    if subsample_icp > 0 and src_points.shape[0] > subsample_icp:
        s_idx = torch.randperm(src_points.shape[0], device=device)[:subsample_icp]
        src_sub = src_points[s_idx]
    else:
        s_idx = None
        src_sub = src_points
    if subsample_icp > 0 and tgt_points.shape[0] > subsample_icp:
        t_idx = torch.randperm(tgt_points.shape[0], device=device)[:subsample_icp]
        tgt_sub = tgt_points[t_idx]
    else:
        t_idx = None
        tgt_sub = tgt_points

    rots = generate_so3_rotations(num_inits, device=device)
    # Pre-score
    pre_scores = []
    for i in range(rots.shape[0]):
        pre_scores.append(_fast_rotation_pre_score(src_sub, tgt_sub, rots[i]))
    pre_scores = torch.stack(pre_scores)
    # Select best rotations
    keep = torch.topk(-pre_scores, k=min(preselect_k, rots.shape[0])).indices
    selected_rots = rots[keep]

    best_score = float("inf")
    best_result = None
    best_metrics = None

    for i, R in enumerate(selected_rots):
        init_transform = SimilarityTransform(
            R=R.unsqueeze(0),
            T=torch.zeros(1, 3, device=device),
            s=torch.ones(1, device=device)
        )
        # Run ICP on subsampled sets
        result = iterative_closest_point(
            src_sub.unsqueeze(0),
            tgt_sub.unsqueeze(0),
            init_transform=init_transform,
            max_iterations=max_iterations,
            allow_reflection=False
        )
        aligned_src = result.Xt.squeeze(0)
        score, metrics = _simple_alignment_metric(aligned_src, tgt_sub)
        if score + 1e-8 < best_score:
            best_score = score
            best_result = result
            best_metrics = metrics
            if best_metrics["median"] < early_thresh:
                print(f"[ICP] Early accept at init {i+1}/{selected_rots.shape[0]}: {best_metrics}")
                break  # early accept

    # (Optional) Could refine transform on full sets if we used subsampling
    return best_result


def unbalanced_ot_loss(x: torch.Tensor, y: torch.Tensor, eps=0.01, tau=0.05) -> torch.Tensor:
    """Unbalanced entropic OT (Sinkhorn) between two feature point clouds (N, F) and (M, F).
    Uses GeomLoss on GPU. Features can concatenate xyz + normals (+ optional semantics)."""
    loss_fn = SamplesLoss("sinkhorn", p=1, blur=eps, reach=tau, debias=False)
    return loss_fn(x, y)


def solve_ot_correspondence(
    source_verts,
    target_verts,
    source_weights,
    target_seg_onehot,
    source_normals,
    target_normals,
    lambda_pos=1.0,
    lambda_normal=0.1,
    lambda_seg=0.5,
    reg=0.01,
):
    """
    Solves for unbalanced OT correspondence.
    Returns the indices of the target points matched to each source point.
    """
    with torch.no_grad():
        # Build the cost matrix C
        cost_pos = torch.cdist(source_verts, target_verts) ** 2
        cost_normal = torch.cdist(source_normals, target_normals) ** 2
        cost_seg = torch.cdist(source_weights, target_seg_onehot) ** 2

        C = lambda_pos * cost_pos + lambda_normal * cost_normal + lambda_seg * cost_seg

        C_np = C.detach().cpu().numpy()

        # Solve unbalanced OT problem using Sinkhorn algorithm
        # This is often faster and more stable for point clouds.
        # It returns a transport plan (correspondence matrix).
        transport_plan = ot.unbalanced.sinkhorn_knopp_unbalanced(
            ot.unif(C_np.shape[0]),
            ot.unif(C_np.shape[1]),
            C_np,
            reg,
            reg_m=1.0,  # Regularization for marginals
        )

        # Find the most likely target for each source point
        matched_indices = np.argmax(transport_plan, axis=1)
        return torch.from_numpy(matched_indices).long().to(source_verts.device)


def normalize_mesh(vertices, pelvis_index=0):
    """Normalizes a mesh to be centered at the pelvis and have unit height."""
    center = vertices[pelvis_index].copy()
    vertices -= center
    height = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    if height > 1e-6:
        vertices /= height
    return vertices, center, 1.0 / height


def build_knn_graph(vertices, k=16):
    """Builds a k-NN graph on the vertices."""
    adjacency_matrix = kneighbors_graph(
        vertices, k, mode="connectivity", include_self=False
    )
    return adjacency_matrix.toarray()


def heat_diffusion(one_hot_weights, knn_graph, iterations=10):
    """Diffuses one-hot weights across the k-NN graph to create soft weights."""
    weights = one_hot_weights.clone()
    knn_graph_tensor = torch.from_numpy(knn_graph).float().to(weights.device)
    D = torch.diag(knn_graph_tensor.sum(1))
    L = D - knn_graph_tensor
    alpha = 0.1
    I = torch.eye(L.shape[0]).to(weights.device)
    diffusion_matrix = I - alpha * L
    with torch.no_grad():
        for _ in range(iterations):
            weights = torch.matmul(diffusion_matrix, weights)
            weights = torch.clamp(weights, min=0.0)
            weights /= weights.sum(dim=1, keepdim=True) + 1e-8
    return weights


def rotation_6d_to_matrix(d6):
    """Converts 6D rotation representation to 3x3 rotation matrix."""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = nn.functional.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = nn.functional.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def apply_skinning(vertices, weights, rotations, translations):
    """Applies linear blend skinning to the vertices.
    vertices: (N,3)
    weights:  (N,K)
    rotations: (K,3,3)  (R_k)
    translations: (K,3)
    Returns blended (N,3)
    """
    # Correct rigid per-cluster rotation: rotated_points[n,k,i] = sum_j R_k[i,j]*v_n[j]
    rotated_points = torch.einsum("nj,kij->nki", vertices, rotations)  # (N,K,3)
    transformed_points = rotated_points + translations.unsqueeze(0)    # (N,K,3)
    blended_points = torch.einsum("nk,nki->ni", weights, transformed_points)
    return blended_points

def _gram_schmidt_ortho(M: torch.Tensor):
    """
    Fast per-matrix Gram-Schmidt (batch) to project 3x3 matrices to SO(3).
    M: (N,3,3)
    Returns R: (N,3,3) (right-handed; fixes reflection).
    """
    # Take columns
    a1 = M[:, :, 0]
    a2 = M[:, :, 1]

    b1 = nn.functional.normalize(a1, dim=1)
    # Remove b1 component from a2
    a2_proj = (b1 * a2).sum(1, keepdim=True) * b1
    b2_raw = a2 - a2_proj
    b2 = nn.functional.normalize(b2_raw, dim=1)
    b3 = torch.cross(b1, b2, dim=1)

    R = torch.stack([b1, b2, b3], dim=2)  # (N,3,3)

    # Fix reflections: if det < 0 flip second column
    det = torch.det(R)
    if (det < 0).any():
        mask = det < 0
        b2_flipped = b2.clone()
        b2_flipped[mask] = -b2_flipped[mask]
        b3 = torch.cross(b1, b2_flipped, dim=1)
        R = torch.stack([b1, b2_flipped, b3], dim=2)
    return R

def blended_rotation(weights, rotations, method: str = "svd"):
    """
    Project weighted rotation blend to SO(3).
    method: 'svd' or 'gs'. (Kept for backward compatibility.)
    """
    M = torch.einsum("nk,kij->nij", weights, rotations)  # (N,3,3)
    if method == "gs":
        return _gram_schmidt_ortho(M)

    # Standard SVD path
    U, S, Vh = torch.linalg.svd(M)
    R = U @ Vh
    det = torch.det(R)
    if (det < 0).any():
        D = torch.ones(R.shape[0], 3, 3, device=R.device, dtype=R.dtype)
        D[:, 2, 2] = torch.where(det < 0, -1.0, 1.0)
        R = U @ D @ Vh

    finite_mask = torch.isfinite(R).view(R.shape[0], -1).all(dim=1)
    bad = ~finite_mask
    if bad.any():
        R_bad = _gram_schmidt_ortho(M[bad])
        R = R.clone()
        R[bad] = R_bad
    return R

def safe_blended_rotation(weights, rotations, train_mode: bool):
    """
    Degeneracy-aware rotation blending:
      - During training (when upstream grads needed) avoid SVD if singular values are (near) repeated.
      - Fallback to Gram-Schmidt for stability.
    """
    M = torch.einsum("nk,kij->nij", weights, rotations)  # (N,3,3)
    if not train_mode:
        return blended_rotation(weights, rotations, method="svd")

    # Quick degeneracy test on a small subsample (avoid full SVD on all if large):
    with torch.no_grad():
        sample_n = min(256, M.shape[0])
        idx = torch.arange(sample_n, device=M.device)
        U_s, S_s, Vh_s = torch.linalg.svd(M[idx])
        # Gaps between singular values
        gaps = torch.stack([ (S_s[:,0]-S_s[:,1]).abs(),
                             (S_s[:,1]-S_s[:,2]).abs() ], dim=1)  # (sample_n,2)
        degenerate_frac = (gaps < 1e-4).any(dim=1).float().mean().item()
    if degenerate_frac > 0.05:
        # Too many near-degenerate -> use GS for all
        return _gram_schmidt_ortho(M)
    # Otherwise use SVD then post-fix like original
    return blended_rotation(weights, rotations, method="svd")


# --- Loss Functions ---
# (ARAP, Seam, and Laplacian losses remain the same as before)
def arap_loss(vertices_canonical, vertices_posed, knn_graph, cluster_labels):
    edges = torch.from_numpy(np.array(np.where(knn_graph == 1)).T).to(
        vertices_canonical.device
    )
    if isinstance(cluster_labels, np.ndarray):
        cluster_labels = torch.from_numpy(cluster_labels).to(vertices_canonical.device)
    edge_labels = cluster_labels[edges]
    mask = edge_labels[:, 0] == edge_labels[:, 1]
    intra_cluster_edges = edges[mask]
    if intra_cluster_edges.shape[0] == 0:
        return torch.tensor(0.0, device=vertices_canonical.device)
    d_canonical = torch.norm(
        vertices_canonical[intra_cluster_edges[:, 0]]
        - vertices_canonical[intra_cluster_edges[:, 1]],
        dim=1,
    )
    d_posed = torch.norm(
        vertices_posed[intra_cluster_edges[:, 0]]
        - vertices_posed[intra_cluster_edges[:, 1]],
        dim=1,
    )
    return torch.abs(d_canonical - d_posed).mean()


def seam_loss(posed_vertices, knn_graph, cluster_labels):
    edges = torch.from_numpy(np.array(np.where(knn_graph == 1)).T).to(
        posed_vertices.device
    )
    if isinstance(cluster_labels, np.ndarray):
        cluster_labels = torch.from_numpy(cluster_labels).to(posed_vertices.device)
    edge_labels = cluster_labels[edges]
    mask = edge_labels[:, 0] != edge_labels[:, 1]
    seam_edges = edges[mask]
    if seam_edges.shape[0] == 0:
        return torch.tensor(0.0, device=posed_vertices.device)
    return (
        torch.norm(
            posed_vertices[seam_edges[:, 0]] - posed_vertices[seam_edges[:, 1]], dim=1
        )
        ** 2
    ).mean()


def laplacian_smoothness_loss(field, knn_graph):
    L = torch.from_numpy(knn_graph).float().to(field.device)
    L_norm = L / (L.sum(dim=1, keepdim=True) + 1e-8)
    laplacian = field - torch.matmul(L_norm, field)
    return torch.norm(laplacian, dim=1).mean()


# --- Corrective MLP ---
class CorrectiveMLP(nn.Module):
    def __init__(self, in_features, out_features=3, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_features),
        )
        self.net[-1].weight.data.fill_(0.0)
        self.net[-1].bias.data.fill_(0.0)
        self.scale = 0.02  # limit max corrective magnitude (~2cm if units are meters)

    def forward(self, x):
        raw = self.net(x)
        return self.scale * torch.tanh(raw)


def normal_consistency_loss(verts, knn_graph):
    # Encourage neighboring vertex normals to be similar
    with torch.no_grad():
        normals = compute_normals(verts)
    A = torch.from_numpy(knn_graph).to(verts.device)
    idx_i, idx_j = A.nonzero(as_tuple=True)
    if idx_i.numel() == 0:
        return torch.tensor(0.0, device=verts.device)
    ni = normals[idx_i]
    nj = normals[idx_j]
    return (1.0 - ( (ni * nj).sum(-1).clamp(-1,1) )).mean()


# --- Main Pipeline ---
class DeformationModel(nn.Module):
    def __init__(self, source_vertices, vertex_segmentation, knn_graph):
        super().__init__()
        self.source_vertices = torch.from_numpy(source_vertices).float()
        self.knn_graph = knn_graph
        self.num_vertices = source_vertices.shape[0]
        self.num_clusters = len(np.unique(vertex_segmentation))

        self.rotations_6d = nn.Parameter(torch.zeros(self.num_clusters, 6))
        self.rotations_6d.data[:, 0] = 1.0
        self.rotations_6d.data[:, 4] = 1.0
        self.translations = nn.Parameter(torch.zeros(self.num_clusters, 3))

        one_hot = nn.functional.one_hot(
            torch.from_numpy(vertex_segmentation), self.num_clusters
        ).float()

        # Keep original hard segmentation labels for ARAP / seam forever.
        self.register_buffer("orig_dominant_clusters",
                             torch.from_numpy(vertex_segmentation).long())

        self.weight_logits = nn.Parameter(torch.log(one_hot + 1e-8))
        self.weight_logits.requires_grad = False

        mlp_input_dim = 3 + self.num_clusters
        self.mlp = CorrectiveMLP(in_features=mlp_input_dim)
        for p in self.mlp.parameters():
            p.requires_grad = False

        diffused_weights = heat_diffusion(one_hot, self.knn_graph)
        self.register_buffer("initial_weights_prior", diffused_weights)

        # Runtime label tensor used in forward for ARAP/seam (initially original)
        self.register_buffer("dominant_clusters",
                             self.orig_dominant_clusters.clone())

    def use_diffused_weights(self, update_labels=False):
        print("Switching to diffused skinning weights for refinement.")
        self.weight_logits.data = torch.log(self.initial_weights_prior.data + 1e-8)
        if update_labels:
            # Optional: usually keep False to preserve rigidity
            new_labels = torch.argmax(self.initial_weights_prior, dim=1)
            changed = (new_labels != self.orig_dominant_clusters).sum().item()
            print(f"[DEBUG] Diffusion would change {changed} / {self.num_vertices} labels.")
            self.dominant_clusters = new_labels
        else:
            # Keep original segmentation for structural losses
            self.dominant_clusters = self.orig_dominant_clusters.clone()

    def forward(self):
        rotations = rotation_6d_to_matrix(self.rotations_6d)
        if self.weight_logits.requires_grad:
            weights = torch.softmax(self.weight_logits, dim=1)
        else:
            if self.weight_logits.grad is None:
                weights = nn.functional.one_hot(self.orig_dominant_clusters,
                                                self.num_clusters).float()
            else:
                weights = torch.softmax(self.weight_logits, dim=1)

        skinned_vertices = apply_skinning(
            self.source_vertices, weights, rotations, self.translations
        )

        if any(p.requires_grad for p in self.mlp.parameters()):
            # Use degeneracy-safe blending when training full model
            use_safe = self.training and (self.rotations_6d.requires_grad or self.weight_logits.requires_grad)
            if use_safe:
                R_b = safe_blended_rotation(weights, rotations, train_mode=True)
            else:
                R_b = blended_rotation(weights, rotations, method="svd")
            if not torch.isfinite(R_b).all():
                R_b = _gram_schmidt_ortho(torch.einsum("nk,kij->nij", weights, rotations))

            t_b = torch.einsum("nk,kj->nj", weights, self.translations)
            local_p = torch.einsum("nij,nj->ni", R_b.transpose(1,2), (skinned_vertices - t_b))
            mlp_input = torch.cat([local_p, weights], dim=1)
            delta_local = self.mlp(mlp_input)
            delta_world = torch.einsum("nij,nj->ni", R_b, delta_local)
            posed = skinned_vertices + delta_world
            return posed, skinned_vertices, delta_world
        return skinned_vertices, skinned_vertices, None


def run_pipeline(
    source_vertices,
    source_faces,
    target_vertices,
    vertex_segmentation,
    log_dir,
    visual_scale=1.0,
):
    """
    Executes the full 4-stage deformation pipeline using OT loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 0. Pre-processing ---
    print("--- Stage 0: Pre-processing ---")
    knn_graph = build_knn_graph(source_vertices, k=16)
    target_vertices_tensor = torch.from_numpy(target_vertices).float().to(device)

    # Pre-compute target normals and segmentation (as they are static)
    target_normals = compute_normals(target_vertices_tensor).float().to(device)
    num_clusters = len(np.unique(vertex_segmentation))
    target_seg_onehot = (
        nn.functional.one_hot(torch.from_numpy(vertex_segmentation), num_clusters)
        .float()
        .to(device)
    )

    model = DeformationModel(source_vertices, vertex_segmentation, knn_graph).to(device)
    model.source_vertices = model.source_vertices.to(device)

    # --- Visualization setup ---
    img_dir = os.path.join(log_dir, "plots")
    os.makedirs(img_dir, exist_ok=True)

    # --- OT Match-Then-Optimize Loop ---
    def run_stage(
        stage_name,
        total_steps,
        optimizer,
        target_vertices_tensor,
        matching_interval=10,          # int OR (init, final) tuple
        lambda_seg_schedule=(0.8, 0.1),
        num_keypoints=None,
        adaptive_matching=True,        # enable interval growth
        early_stop=False,              # enable early stopping
        early_metric="data_fit",       # "data_fit" or "total"
        early_patience=80,             # steps with no improvement
        early_min_delta=1e-4,          # required improvement
        early_warmup=50,               # do not early-stop before this
        min_fraction=0.4               # run at least this fraction of steps
    ):
        """
        matching_interval:
            - int: fixed interval
            - (init, final): linearly increase interval from init -> final across the stage
        """
        print(f"\n--- {stage_name} (Keypoints: {num_keypoints if num_keypoints else 'All'}) ---")

        pbar = tqdm(range(total_steps), desc=stage_name)

        # Cached correspondence (indices + fixed targets)
        source_indices = None
        fixed_targets = None
        cached_source_normals = None
        cached_target_subset = None
        cached_target_normals = None

        # Early stopping bookkeeping
        best_metric = float("inf")
        since_improve = 0
        min_required_steps = int(total_steps * min_fraction)

        # Resolve interval mode
        if isinstance(matching_interval, tuple) or isinstance(matching_interval, list):
            base_interval_init, base_interval_final = matching_interval
        else:
            base_interval_init = base_interval_final = int(matching_interval)

        last_used_interval = None

        for step in pbar:
            progress = step / max(1, total_steps - 1)

            # Adaptive interval computation
            if adaptive_matching and base_interval_final > base_interval_init:
                current_interval = int(
                    round(base_interval_init + (base_interval_final - base_interval_init) * progress)
                )
            else:
                current_interval = base_interval_init

            if current_interval < 1:
                current_interval = 1

            if current_interval != last_used_interval and step % 25 == 0:
                print(f"[{stage_name}] Adjust match interval -> {current_interval}")
                last_used_interval = current_interval

            model.train()
            optimizer.zero_grad()

            posed_verts, skinned_verts, displacements = model()
            final_verts = posed_verts  # alias used below (must keep grad, do NOT detach)

            # NaN early check
            if not torch.isfinite(posed_verts).all():
                print(f"[NaN Guard] posed_verts contains NaN/Inf at step {step}. Falling back / aborting.")
                break

            # Recompute correspondence on (dynamic) interval
            if (step % current_interval == 0) or fixed_targets is None:
                with torch.no_grad():
                    if num_keypoints:
                        source_indices = fps(final_verts.detach(), num_keypoints)
                        target_indices = fps(
                            target_vertices_tensor.detach(), num_keypoints
                        )
                        src_subset_now = final_verts[source_indices]
                        cached_target_subset = target_vertices_tensor[target_indices]
                    else:
                        source_indices = torch.arange(
                            final_verts.shape[0], device=final_verts.device
                        )
                        src_subset_now = final_verts.detach()
                        cached_target_subset = target_vertices_tensor

                    cached_source_normals = compute_normals(src_subset_now).detach()
                    cached_target_normals = compute_normals(
                        cached_target_subset
                    ).detach()

                    source_labels = model.dominant_clusters[source_indices]
                    if num_keypoints:
                        target_labels = torch.argmax(target_seg_onehot, dim=1)[
                            target_indices
                        ]
                    else:
                        target_labels = torch.argmax(target_seg_onehot, dim=1)

                    # --- Stage 1 (per-cluster refined matching without segmentation cost) ---
                    if "Stage 1" in stage_name:
                        # Flag to know if we already have persisted matches
                        if 'stage1_have_matches' not in locals() or stage1_stable_targets is None \
                                or (stage1_stable_targets.shape[0] != src_subset_now.shape[0]):
                            stage1_have_matches = False
                        else:
                            stage1_have_matches = True

                        # Allocate container (either previous stable targets or placeholder)
                        if stage1_have_matches:
                            new_fixed_targets = stage1_stable_targets.clone()
                        else:
                            new_fixed_targets = torch.empty_like(src_subset_now)  # will be fully written

                        for k in range(model.num_clusters):
                            part_mask_src = (source_labels == k)
                            part_mask_tgt = (target_labels == k)
                            if part_mask_src.sum() == 0:
                                continue
                            if part_mask_tgt.sum() == 0:
                                # No target points in this cluster; if first time, fall back to copying current src
                                if not stage1_have_matches:
                                    new_fixed_targets[part_mask_src] = src_subset_now[part_mask_src]
                                continue

                            src_part = src_subset_now[part_mask_src]
                            tgt_part = cached_target_subset[part_mask_tgt]
                            src_norm_part = cached_source_normals[part_mask_src]
                            tgt_norm_part = cached_target_normals[part_mask_tgt]

                            icp_result = multi_run_icp(src_part, tgt_part, num_inits=24, preselect_k=24)
                            if icp_result is None:
                                fixed_targets[part_mask_src] = src_part
                                continue

                            src_part = icp_result.Xt.squeeze(0)
                            src_norm_part = compute_normals(src_part)

                            # # Descriptors (normals + scale)
                            # src_scale_part = local_scale(src_part)
                            # tgt_scale_part = local_scale(tgt_part)
                            # src_desc = torch.cat([src_norm_part, src_scale_part], dim=1)
                            # tgt_desc = torch.cat([tgt_norm_part, tgt_scale_part], dim=1)

                            # # Costs
                            # cost_pos    = torch.cdist(src_part, tgt_part)
                            # cost_normal = torch.cdist(src_norm_part, tgt_norm_part) ** 2
                            # cost_desc   = torch.cdist(src_desc, tgt_desc) ** 2

                            # def _nr(M):
                            #     return M / (M.mean(dim=1, keepdim=True) + 1e-8)

                            # C = (
                            #     _nr(cost_pos)
                            #     + 0.1 * _nr(cost_normal)
                            #     + 0.05 * _nr(cost_desc)
                            # )

                            # C_full = _nr(cost_pos) * 0.75 + 0.25 * _nr(cost_normal) + 0.05 * _nr(cost_desc)

                            # # Candidate pruning
                            # k_prune = min(64, tgt_part.shape[0])
                            # if k_prune < 1:
                            #     if not stage1_have_matches:
                            #         new_fixed_targets[part_mask_src] = src_part
                            #     continue
                            # _, nn_idx = torch.topk(cost_pos, k_prune, largest=False)
                            # large = C_full.max().detach() + 10.0
                            # C = torch.full_like(C_full, large)
                            # row_ids = torch.arange(C.shape[0], device=C.device).unsqueeze(1).expand(-1, k_prune)
                            # C[row_ids, nn_idx] = C_full[row_ids, nn_idx]

                            # P = sinkhorn_transport_plan(
                            #     C, eps=0.01, tau=0.05, n_iters=50
                            # )
                            # prelim_idx = greedy_unique_assignment(P, nn_idx)
                            cost_pos_part = torch.cdist(src_part, tgt_part) ** 2
                            cost_normal_part = (
                                torch.cdist(src_norm_part, tgt_norm_part) ** 2
                            )
                            C_part = 0.75 * cost_pos_part + 0.25 * cost_normal_part

                            P = sinkhorn_transport_plan(
                                C_part, eps=0.01, tau=0.05, n_iters=40
                                # C_part,
                                # eps=0.003,
                                # tau=10.0,
                                # n_iters=50,
                            )
                            matched_indices_part = torch.argmax(P, dim=1)
                            new_fixed_targets[part_mask_src] = tgt_part[
                                matched_indices_part
                            ]

                            # # Optional mutual refinement
                            # unique_tgt, inverse = torch.unique(prelim_idx, return_inverse=True)
                            # tgt_subset = tgt_part[unique_tgt]
                            # dist_t2s = torch.cdist(tgt_subset, src_part)
                            # k_back = min(4, src_part.shape[0])
                            # _, src_rank = torch.topk(dist_t2s, k_back, largest=False)
                            # mutual_mask = torch.zeros_like(prelim_idx, dtype=torch.bool)
                            # for t_i in range(unique_tgt.shape[0]):
                            #     allowed_sources = src_rank[t_i]
                            #     src_indices_part = torch.nonzero(inverse == t_i).squeeze(1)
                            #     for s_local in src_indices_part:
                            #         if (allowed_sources == s_local).any():
                            #             mutual_mask[s_local] = True
                            # # Fallback to nearest positional candidate
                            # fallback_idx = nn_idx[torch.arange(nn_idx.shape[0]), 0]
                            # final_idx_part = torch.where(mutual_mask, prelim_idx, fallback_idx)
                            # new_matches = tgt_part[final_idx_part]

                            # if stage1_have_matches:
                            #     # Persistence (compare to previous)
                            #     prev_part = stage1_stable_targets[part_mask_src]
                            #     prev_cost = (src_part - prev_part).pow(2).sum(1)
                            #     new_cost  = (src_part - new_matches).pow(2).sum(1)
                            #     # Allow replacement if strictly better OR previous was identity/self (very low prev_cost)
                            #     # Treat tiny prev_cost as non-locked
                            #     tiny = (prev_cost < 1e-8)
                            #     improve_mask = (new_cost + 1e-7 < prev_cost) | tiny
                            #     updated_part = torch.where(improve_mask.unsqueeze(1), new_matches, prev_part)
                            #     new_fixed_targets[part_mask_src] = updated_part
                            # else:
                            #     # First-ever matching pass: accept all matches directly
                            #     new_fixed_targets[part_mask_src] = new_matches

                        stage1_stable_targets = new_fixed_targets
                        fixed_targets = stage1_stable_targets
                        stage1_have_matches = True
                    else:
                        # (Unchanged global OT branch)
                        progress = step / total_steps if total_steps > 0 else 1
                        lambda_seg = lambda_seg_schedule[0] * (1 - progress) + lambda_seg_schedule[1] * progress

                        # 1. Hybrid segmentation weights (smooth transition)
                        warmup_steps = 50 if "Skin-weight" in stage_name else 0
                        soft_w = torch.softmax(model.weight_logits, dim=1)
                        hard_w = nn.functional.one_hot(
                            model.orig_dominant_clusters[source_indices], model.num_clusters
                        ).float()
                        if step < warmup_steps:
                            blend = 0.0
                        else:
                            # Increase blend toward 1.0 gradually (e.g. over 100 steps)
                            blend = min(1.0, (step - warmup_steps) / 100.0)
                        source_seg_for_cost = (1 - blend) * hard_w + blend * soft_w[source_indices]

                        target_seg_subset = target_seg_onehot[target_indices] if num_keypoints else target_seg_onehot

                        src_scale = local_scale(src_subset_now)
                        tgt_scale = local_scale(cached_target_subset)
                        src_desc = torch.cat([cached_source_normals, src_scale], dim=1)
                        tgt_desc = torch.cat([cached_target_normals, tgt_scale], dim=1)

                        cost_pos = torch.cdist(src_subset_now, cached_target_subset)
                        cost_normal = (
                            torch.cdist(cached_source_normals, cached_target_normals)
                            ** 2
                        )
                        cost_seg = (
                            torch.cdist(source_seg_for_cost, target_seg_subset) ** 2
                        )
                        cost_desc = torch.cdist(src_desc, tgt_desc) ** 2

                        def normalize_rows(M):
                            return M / (M.mean(dim=1, keepdim=True) + 1e-8)

                        cost_pos_n = normalize_rows(cost_pos)
                        cost_normal_n = normalize_rows(cost_normal)
                        cost_seg_n = normalize_rows(cost_seg)
                        cost_desc_n = normalize_rows(cost_desc)

                        C_full = (
                            cost_pos_n
                            + 0.1 * cost_normal_n
                            + lambda_seg * cost_seg_n
                            + 0.05 * cost_desc_n
                        )

                        # 3. Candidate pruning (k nearest in Euclidean space)
                        k_prune = 64
                        with torch.no_grad():
                            _, nn_idx = torch.topk(cost_pos, k_prune, largest=False)  # indices of nearest targets
                        # Build masked large cost
                        large = C_full.max().detach() + 10.0
                        C = torch.full_like(C_full, large)
                        row_ids = torch.arange(C.shape[0], device=C.device).unsqueeze(1).expand(-1, k_prune)
                        C[row_ids, nn_idx] = C_full[row_ids, nn_idx]

                        # 4. Sharper Sinkhorn on pruned matrix
                        P = sinkhorn_transport_plan(C, eps=0.003, tau=0.15, n_iters=60)

                        # 5. Greedy uniqueness-aware assignment instead of plain argmax
                        prelim_idx = greedy_unique_assignment(P, nn_idx)

                        # 6. Mutual nearest consistency (position based within candidate set)
                        # For each chosen target, check if source is among that target's top-k_back sources
                        k_back = 4
                        # Compute reverse: for efficiency approximate by ensuring chosen target is within source's k_prune set AND
                        # source is within its k_back in that target's perspective.
                        # Build target->source distance matrix on-the-fly for subset of chosen targets
                        unique_tgt, inverse = torch.unique(prelim_idx, return_inverse=True)
                        tgt_pos = cached_target_subset[unique_tgt]
                        # Dist from each chosen target to all sources
                        dist_t2s = torch.cdist(tgt_pos, src_subset_now)
                        _, src_rank = torch.topk(dist_t2s, k_back, largest=False)
                        # For each source row, check if its index is in the k_back list of its matched target
                        mutual_mask = torch.zeros_like(prelim_idx, dtype=torch.bool)
                        for t_i, tgt_id in enumerate(unique_tgt):
                            allowed_sources = src_rank[t_i]
                            mutual_mask[inverse == t_i] = torch.isin(
                                torch.nonzero(inverse == t_i).squeeze(1), allowed_sources
                            )
                        # Fallback: if not mutual, pick nearest (position) among its candidate list
                        fallback_idx = nn_idx[torch.arange(nn_idx.shape[0]), 0]
                        final_idx = torch.where(mutual_mask, prelim_idx, fallback_idx)

                        new_fixed_targets = cached_target_subset[final_idx]

                        # 7. Match persistence
                        if 'stable_fixed_targets' not in locals():
                            stable_fixed_targets = new_fixed_targets.clone()
                        else:
                            prev_cost = (src_subset_now - stable_fixed_targets).pow(2).sum(1)
                            new_cost  = (src_subset_now - new_fixed_targets).pow(2).sum(1)
                            improve_mask = new_cost < (prev_cost - 1e-5)
                            stable_fixed_targets = torch.where(
                                improve_mask.unsqueeze(1), new_fixed_targets, stable_fixed_targets
                            )
                        fixed_targets = stable_fixed_targets

                        if (step % 50 == 0) and ("Skin-weight" in stage_name):
                            with torch.no_grad():
                                bad = (final_idx == fallback_idx).float().mean().item()
                                print(f"[DEBUG] step {step} fallback_frac={bad:.3f} blend={blend:.2f}")

            current_subset = final_verts[source_indices]
            loss_data_fit = torch.abs(current_subset - fixed_targets).mean()

            progress = step / total_steps
            _lambda_seg = lambda_seg_schedule[0] * (1 - progress) + lambda_seg_schedule[1] * progress

            loss_arap_val = arap_loss(
                model.source_vertices, posed_verts, knn_graph, model.dominant_clusters
            )
            loss_seam_val = seam_loss(posed_verts, knn_graph, model.dominant_clusters)

            seam_weight = 0.5
            arap_weight = 0.15
            total_loss = loss_data_fit + arap_weight * loss_arap_val + seam_weight * loss_seam_val

            loss_vert_lap = laplacian_smoothness_loss(posed_verts, knn_graph)
            total_loss += 0.05 * loss_vert_lap

            if "Skin-weight" in stage_name or "Fine-tune" in stage_name:
                weights = torch.softmax(model.weight_logits, dim=1)
                loss_seg_prior = nn.functional.mse_loss(weights, model.initial_weights_prior)
                loss_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
                total_loss += 0.1 * loss_seg_prior + 0.01 * loss_entropy * _lambda_seg

            if "MLP" in stage_name or "Fine-tune" in stage_name:
                if displacements is not None:
                    loss_delta_smooth = laplacian_smoothness_loss(displacements, knn_graph)
                    loss_delta_mag = torch.norm(displacements, dim=1).mean()
                    total_loss += 0.2 * loss_delta_smooth + 5e-4 * loss_delta_mag
                    loss_norm_consistency = normal_consistency_loss(posed_verts.detach(), knn_graph)
                    total_loss += 0.02 * loss_norm_consistency

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            pbar.set_postfix(
                loss=f"{total_loss.item():.6f}",
                L1=f"{loss_data_fit.item():.6f}",
                match_int=current_interval
            )

            # Early stopping check
            if early_stop:
                metric_val = loss_data_fit.item() if early_metric == "data_fit" else total_loss.item()
                if best_metric - metric_val > early_min_delta:
                    best_metric = metric_val
                    since_improve = 0
                else:
                    since_improve += 1
                if (step >= early_warmup and step >= min_required_steps
                        and since_improve >= early_patience):
                    print(f"[{stage_name}] Early stop at step {step} "
                          f"(best {early_metric}: {best_metric:.6f})")
                    break

            if step % 100 == 0:
                matches_to_plot = torch.stack([final_verts[source_indices], fixed_targets], dim=1)
                plot_pointcloud(
                    final_verts.detach().cpu(),
                    img_dir,
                    title=f"{stage_name.replace(' ', '_')}_{step:04d}",
                    extra_points=target_vertices_tensor.cpu(),
                    matches=matches_to_plot.detach().cpu(),
                    scale=visual_scale,
                )

        posed, _, _ = model()
        if source_faces is not None:
            mesh = trimesh.Trimesh(vertices=posed.detach().cpu().numpy(), faces=source_faces)
            mesh.export(os.path.join(log_dir, f"{stage_name.replace(' ', '_')}_mesh.obj"))
            # --- New: per-target error colormap export ---
            error_mesh_path = os.path.join(
                log_dir, f"{stage_name.replace(' ', '_')}_error_map.ply"
            )
            _export_error_colormap_mesh(
                target_vertices_tensor=target_vertices_tensor,
                posed_vertices_tensor=posed,
                out_path=error_mesh_path,
                faces=source_faces if source_faces is not None else None,
                colormap="plasma",
            )
        else:
            np.save(
                os.path.join(log_dir, f"{stage_name.replace(' ', '_')}_points.npy"),
                posed.detach().cpu().numpy(),
            )

    # --- 1. Rigid Core Fitting ---
    optimizer = optim.Adam([model.rotations_6d, model.translations], lr=5e-3)
    run_stage(
        "Stage 1: Rigid Core Fitting",
        500,
        optimizer,
        target_vertices_tensor,
        matching_interval=(10, 60),  # start every 10 steps, relax to every 60
        adaptive_matching=True,
        early_stop=True,
        early_metric="data_fit",
        early_patience=70,
        # early_min_delta=5e-5,
        early_min_delta=1e-5,
        early_warmup=300,
        num_keypoints=4096,
        lambda_seg_schedule=(0.0, 0.0),
    )

    # --- 2. Skin-weight Refinement ---
    model.use_diffused_weights(update_labels=False)  # keep original labels
    # Debug: how many labels would change?
    with torch.no_grad():
        proposed_labels = torch.argmax(model.initial_weights_prior, dim=1)
        changed = (proposed_labels != model.orig_dominant_clusters).sum().item()
        print(f"[DEBUG] Vertices whose argmax would change after diffusion: {changed}")

    # More detail needed. Increase the number of points.
    model.weight_logits.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    run_stage(
        "Stage 2: Skin-weight Refinement",
        500,
        optimizer,
        target_vertices_tensor,
        matching_interval=(10, 50),   # still adapt but tighter
        adaptive_matching=True,
        early_stop=True,
        early_metric="data_fit",
        early_patience=80,
        early_min_delta=3e-5,
        early_warmup=80,
        num_keypoints=4096,
        lambda_seg_schedule=(1.0, 0.1),
    )

    # --- 3. Per-point Corrective MLP ---
    # High-frequency details. Use the full point cloud for accuracy.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.mlp.parameters():
        param.requires_grad = True
    optimizer_mlp = optim.Adam(model.mlp.parameters(), lr=1e-3)
    run_stage(
        "Stage 3a: MLP Training (Frozen Core)", 200, optimizer_mlp, target_vertices_tensor,
        num_keypoints=None, lambda_seg_schedule=(0.1, 0.05)
    )

    for param in model.parameters():
        param.requires_grad = True
    # Enable targeted debug for Stage 3b
    model.debug_stage3b = True
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    run_stage(
        "Stage 3b: MLP Joint Fine-tuning", 200, optimizer, target_vertices_tensor,
        num_keypoints=None, lambda_seg_schedule=(0.1, 0.05)
    )
    # Disable after stage
    model.debug_stage3b = False

    # --- 4. Global Fine-tune ---
    # Final polish. Must use the full cloud for the best result.
    optimizer = optim.Adam(model.parameters(), lr=5e-5)
    run_stage(
        "Stage 4: Global Fine-tune", 200, optimizer, target_vertices_tensor,
        num_keypoints=None, lambda_seg_schedule=(0.05, 0.01)
    )

    print("\n--- Pipeline Finished ---")
    final_vertices, _, _ = model()
    return final_vertices.detach().cpu().numpy()


# --- Example Usage ---
if __name__ == "__main__":
    os.chdir("/NAS/spa176/papr-retarget")

    part_names = [
        "root",
        "head_neck",
        "left_upper_leg",
        "left_lower_leg",
        "left_lower_arm",
        "left_upper_arm",
        "right_upper_leg",
        "right_lower_leg",
        "right_lower_arm",
        "right_upper_arm",
    ]
    total_part_num = len(part_names)

    part_indices = {
        name: np.load(
            f"/NAS/spa176/smplx/smpl_{total_part_num}_parts/{name}_indices.npy"
        )
        for name in part_names
    }

    sample_name = "sample_5"
    ref_obj_mesh = trimesh.load(
        f"/NAS/spa176/skeleton-free-pose-transfer/demo/smpl_pose_0/{sample_name}.obj",
        process=False,
    )

    deforming_obj_mesh = trimesh.load(
        "/NAS/spa176/skeleton-free-pose-transfer/demo/smpl_pose_0/rest.obj",
        process=False,
    )

    source_verts = deforming_obj_mesh.vertices
    target_verts = ref_obj_mesh.vertices
    source_faces = deforming_obj_mesh.faces

    vert_seg = np.zeros(source_verts.shape[0], dtype=int)
    for i, name in enumerate(part_names):
        vert_seg[part_indices[name]] = i

    log_dir = "fit_pointcloud_logs"
    exp_dir = f"smpl_skinning_ot"
    exp_id = sample_name
    exp_sub_dir = f"exp_{exp_id}_8"
    log_dir = os.path.join(log_dir, exp_dir, exp_sub_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    shutil.copy(__file__, log_dir)

    VISUAL_SCALE = 1.0
    plot_pointcloud(
        torch.from_numpy(source_verts),
        log_dir,
        title="Source Point Cloud",
        scale=VISUAL_SCALE,
    )
    plot_pointcloud(
        torch.from_numpy(target_verts),
        log_dir,
        title="Target Point Cloud",
        scale=VISUAL_SCALE,
    )

    print(f"Source shape: {source_verts.shape}")
    print(f"Target shape: {target_verts.shape}")
    print(f"Segmentation shape: {vert_seg.shape}, Clusters: {len(np.unique(vert_seg))}")

    # # Run the full pipeline
    # final_posed_vertices = run_pipeline(
    #     source_verts, source_faces, target_verts, vert_seg, log_dir, visual_scale=VISUAL_SCALE
    # )
    # from rig_reg_deform_graph import run_pipeline_deform_graph

    # final_posed_vertices = run_pipeline_deform_graph(
    #     source_vertices=source_verts,
    #     source_faces=source_faces,
    #     target_vertices=target_verts,
    #     target_segmentation=vert_seg,
    #     log_dir=log_dir,
    #     visual_scale=VISUAL_SCALE,
    #     # Optional knobs:
    #     nodes_per_part="auto",      # "auto" | "1" | "2"
    #     prior_mode="hinge",         # "hinge" | "ball"
    #     beta_norm=0.2,              # weight for normal term in OT cost
    #     sinkhorn_eps=0.05,
    #     sinkhorn_iters=60,
    #     lambda_seam=0.1,
    #     lambda_arap=0.05,
    #     lambda_prior=0.01,
    #     iters=600,
    #     lr=5e-3,
    # )

    # from rig_reg_deform_graph_adapted import run_pipeline_deform_graph

    # final_posed_vertices = run_pipeline_deform_graph(
    #     source_vertices=source_verts,
    #     source_faces=source_faces,          # kept for mesh export
    #     target_vertices=target_verts,
    #     target_segmentation=vert_seg,
    #     log_dir=log_dir,                    # HTML plots + meshes land here
    #     visual_scale=1.0,
    #     source_segmentation=vert_seg,     # optional; enables per-part balancing in Stage 1
    #     # your existing knobs still work:
    #     nodes_per_part="auto",
    #     k_skin=8,
    #     sinkhorn_eps=0.05,                  # base value (stage schedule overrides internally)
    #     sinkhorn_iters=60,
    #     beta_norm=0.2,
    #     lambda_seam=0.1,
    #     lambda_arap=0.05,
    #     lambda_prior=0.01,
    #     prior_mode="hinge",
    #     iters=600,
    #     lr=5e-3,
    # )

    from rig_reg_deform_graph_merged import run_merged_pipeline

    up_hint = np.array([0, 1, 0], dtype=np.float64)
    front_hint = np.array([0, 0, 1], dtype=np.float64)

    final_posed_vertices = run_merged_pipeline(
        source_vertices=source_verts,
        source_faces=source_faces,
        target_vertices=target_verts,
        target_segmentation=vert_seg,
        # source_segmentation=vert_seg,
        part_indices=part_indices,
        log_dir=log_dir,
        # your existing knobs still work:
        deform_kwargs=dict(
            visual_scale=1.0,
            nodes_per_part="auto",
            k_skin=8,
            sinkhorn_eps=0.05,  # base value (stage schedule overrides internally)
            sinkhorn_iters=60,
            beta_norm=0.2,
            lambda_seam=0.1,
            lambda_arap=0.05,
            lambda_prior=0.01,
            prior_mode="hinge",
            iters=600,
            lr=5e-3,
        ),
        rigid_params=dict(
            outer_iters=15,
            keep_ratio_schedule=[0.7, 0.8, 0.85, 0.9],
            # smooth_lambda=20.0,
            # boundary_gamma=80.0,
            # prior_mu=10.0,
            smooth_lambda=0.0,
            boundary_gamma=1.0,
            prior_mu=1.0,
            # smooth_lambda=0.0,
            # boundary_gamma=0.0,
            # prior_mu=0.0,
            lm_damp=1e-6,
            verbose=True,
            export_meshes=True,
            log_dir=log_dir,
            source_faces=source_faces,
            part_indices=part_indices,
            init_mode="graph",
            root_part_id="root",
            # root_part_id=0,
            front_hint=front_hint,
            up_hint=up_hint,
            z_mode="seam_centroid_to_centroid_direction",
            # p2p_parts=["left_lower_arm", "right_lower_arm"],
            # debug_parts=["left_lower_arm", "right_lower_arm"],
            # debug_parts=["head_neck"],
            # debug_plot=True,
            # flip_normals=["left_lower_arm", "right_lower_arm"],
        ),  # use defaults
    )

    plot_pointcloud(
        torch.from_numpy(final_posed_vertices),
        log_dir,
        title="Final Posed Point Cloud",
        extra_points=torch.from_numpy(target_verts),
        scale=VISUAL_SCALE,
    )

    # You can now save `final_posed_vertices` as a mesh file (e.g., .obj)
    # to visualize the result.
    final_mesh = trimesh.Trimesh(
        vertices=final_posed_vertices, faces=deforming_obj_mesh.faces
    )
    final_mesh.export(os.path.join(log_dir, "final_posed_mesh.obj"))

    # Calculate final error
    final_error = np.linalg.norm(final_posed_vertices - target_verts, axis=1).mean()
    print(f"\nFinal Mean Per-Vertex Error: {final_error:.6f}")
