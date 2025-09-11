import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm  # added
import time  # added
import plotly.graph_objs as go

from typing import Optional, Tuple, Dict

# =========================
# Utils: SO(3) operations
# =========================
def hat(v: torch.Tensor) -> torch.Tensor:
    """
    v: (...,3)
    return: (...,3,3) skew-symmetric matrix
    """
    x, y, z = v[...,0], v[...,1], v[...,2]
    O = torch.zeros_like(x)
    M = torch.stack([
        torch.stack([ O, -z,  y], dim=-1),
        torch.stack([ z,  O, -x], dim=-1),
        torch.stack([-y,  x,  O], dim=-1)
    ], dim=-2)
    return M

def exp_so3(omega: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues for batched axis-angle in R^3 -> SO(3)
    omega: (...,3)
    return: (...,3,3)
    """
    theta = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp_min(1e-12)
    k = omega / theta
    K = hat(k)
    I = torch.eye(3, device=omega.device, dtype=omega.dtype).expand(omega.shape[:-1] + (3,3))
    s = torch.sin(theta)[...,None]
    c = torch.cos(theta)[...,None]
    R = I + s*K + (1 - c)*K@K
    # when theta~0, fall back to I + hat(omega)
    small = (theta.squeeze(-1) < 1e-4)[...,None,None]
    R = torch.where(small, I + hat(omega), R)
    return R

def log_so3(R: torch.Tensor) -> torch.Tensor:
    """
    Map SO(3)->R^3 via matrix log (axis-angle vector).
    R: (...,3,3)
    return: (...,3)
    """
    cos_theta = ((R.diagonal(offset=0, dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1+1e-7, 1-1e-7)
    theta = torch.arccos(cos_theta)
    # Handle small angles
    small = (theta < 1e-4)[...,None,None]
    # For general case:
    w = torch.stack([R[...,2,1] - R[...,1,2],
                     R[...,0,2] - R[...,2,0],
                     R[...,1,0] - R[...,0,1]], dim=-1) / (2*torch.sin(theta)[...,None])
    w = w * theta[...,None]
    # Small angle approx:
    w_small = torch.stack([R[...,2,1] - R[...,1,2],
                           R[...,0,2] - R[...,2,0],
                           R[...,1,0] - R[...,0,1]], dim=-1) / 2.0
    return torch.where(small.squeeze(-2).squeeze(-2)[...,None], w_small, w)


# =========================
# Geometry helpers
# =========================
def pca_principal_axis(points: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """
    points: (N,3) torch
    returns: axis (3,), s1, s2 (top two singular values)
    """
    X = points - points.mean(dim=0, keepdim=True)
    # SVD on covariance
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    axis = Vh[0]
    s1, s2 = S[0].item(), (S[1].item() if S.shape[0] > 1 else 0.0)
    return axis, s1, s2

def compute_knn(points: torch.Tensor, k: int) -> torch.Tensor:
    """
    Return knn indices (N,k) excluding self.
    """
    # (N,N) squared distances via (a-b)^2 = a^2 + b^2 -2ab
    with torch.no_grad():
        x2 = (points**2).sum(-1, keepdim=True)          # (N,1)
        d2 = x2 + x2.t() - 2 * (points @ points.t())    # (N,N)
        d2.fill_diagonal_(float('inf'))
        knn = d2.topk(k=k, largest=False).indices       # (N,k)
    return knn

def estimate_normals_pca_knn(points: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
    """
    Vectorized PCA normal estimation using provided knn indices.
    Safe under AMP (forces float32 for eigh).
    points: (N,3)
    knn_idx: (N,k)
    returns (N,3)
    """
    with torch.no_grad():
        # Disable autocast so we stay in float32 for linalg.eigh (not implemented for float16 on CUDA)
        with torch.cuda.amp.autocast(enabled=False):
            pts32 = points.float()
            nb = pts32[knn_idx]              # (N,k,3)
            mu = nb.mean(dim=1, keepdim=True)
            X = nb - mu
            C = X.transpose(1,2) @ X / (X.shape[1]-1)   # (N,3,3) float32
            eigvals, eigvecs = torch.linalg.eigh(C)     # float32
            n = eigvecs[:,:,0]
            n = n / (n.norm(dim=-1, keepdim=True)+1e-12)
    return n.to(points.dtype)

def estimate_normals_pca(points: torch.Tensor, k: int = 16) -> torch.Tensor:
    """
    AMP-safe scalar version.
    """
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=False):
            pts32 = points.float()
            dists = torch.cdist(pts32, pts32)
            knn_idx = dists.topk(k=k+1, largest=False).indices[:,1:]
            normals = []
            for i in range(pts32.shape[0]):
                nb = pts32[knn_idx[i]]
                mu = nb.mean(dim=0, keepdim=True)
                X = nb - mu
                C = X.t().mm(X) / max(1, nb.shape[0]-1)
                eigvals, eigvecs = torch.linalg.eigh(C)
                n = eigvecs[:,0]
                normals.append(n / (n.norm()+1e-12))
            normals = torch.stack(normals, dim=0)
    return normals.to(points.dtype)

def infer_part_adjacency(points: torch.Tensor, labels: torch.Tensor, tau: Optional[float] = None) -> torch.Tensor:
    """
    Infer a simple undirected adjacency over parts by proximity.
    Returns adjacency matrix A of shape (P,P) with {0,1}.
    """
    parts = torch.unique(labels).tolist()
    P = len(parts)
    part_to_idx = {p:i for i,p in enumerate(parts)}
    # compute per-part centroids and radii
    centroids, radii = [], []
    for p in parts:
        pts = points[labels==p]
        c = pts.mean(dim=0)
        r = torch.quantile(torch.linalg.norm(pts - c, dim=1), 0.9)
        centroids.append(c)
        radii.append(r)
    C = torch.stack(centroids)  # (P,3)
    R = torch.stack(radii)      # (P,)
    if tau is None:
        bbox = (points.max(0).values - points.min(0).values).norm().item()
        tau = 0.04 * bbox
    # adjacency if spheres overlap or are close
    A = torch.zeros((P,P), dtype=torch.bool, device=points.device)
    for i in range(P):
        for j in range(i+1, P):
            d = torch.norm(C[i]-C[j]).item()
            if d < (R[i]+R[j]).item() + tau:
                A[i,j] = A[j,i] = True
    return A, parts


# =========================
# Deformation Graph
# =========================
def build_deformation_graph(points: torch.Tensor,
                            labels: torch.Tensor,
                            nodes_per_part: str = "auto",
                            elong_thresh: float = 1.7) -> Dict[str, torch.Tensor]:
    """
    Build nodes: 1 or 2 per part using PCA elongation.
    Returns dict with:
    - 'G': (M,3) node positions
    - 'G_part': (M,) node part labels
    - 'E': (E,2) undirected edges over nodes (k-NN in node space + same-part links)
    """
    device = points.device
    parts = torch.unique(labels).tolist()
    G_list, G_part_list = [], []
    for p in parts:
        idx = (labels==p)
        pts = points[idx]
        axis, s1, s2 = pca_principal_axis(pts)
        cent = pts.mean(dim=0)
        if nodes_per_part == "2" or (nodes_per_part=="auto" and s2>1e-12 and (s1/s2) > elong_thresh):
            # two nodes along principal axis at 25% and 75% quantiles
            proj = (pts - cent) @ axis
            q25 = torch.quantile(proj, 0.25)
            q75 = torch.quantile(proj, 0.75)
            g1 = cent + q25*axis
            g2 = cent + q75*axis
            G_list += [g1, g2]
            G_part_list += [p, p]
        else:
            G_list.append(cent)
            G_part_list.append(p)
    G = torch.stack(G_list, dim=0).to(device)   # (M,3)
    G_part = torch.tensor(G_part_list, device=device, dtype=labels.dtype)

    # Build node edges by kNN in node space and fully connect nodes within same part
    M = G.shape[0]
    with torch.no_grad():
        d = torch.cdist(G, G)
        k = min(6, max(2, M-1))
        knn_idx = d.topk(k=k+1, largest=False).indices[:,1:]
    edges = set()
    for i in range(M):
        for j in knn_idx[i].tolist():
            a, b = (i, j) if i<j else (j, i)
            edges.add((a, b))
    # fully connect same-part nodes
    for i in range(M):
        for j in range(i+1, M):
            if G_part[i]==G_part[j]:
                edges.add((i,j))
    E = torch.tensor(list(edges), dtype=torch.long, device=device) if edges else torch.empty((0,2), dtype=torch.long, device=device)
    return {"G": G, "G_part": G_part, "E": E}


def gaussian_skinning(points: torch.Tensor, G: torch.Tensor, k:int=8, sigma: Optional[float]=None) -> torch.Tensor:
    """
    Compute per-point skinning weights to graph nodes using Gaussian kernel.
    returns: W (N,M), sparse via top-k.
    """
    N, M = points.shape[0], G.shape[0]
    d = torch.cdist(points, G)  # (N,M)
    if sigma is None:
        # set sigma as mean k-th NN distance across points
        kth = min(k, M-1)
        sigma = (d.topk(k=kth+1, largest=False).values[:, kth]).mean().item() + 1e-6
    K = torch.exp(- (d**2) / (2*sigma**2))
    # sparsify: keep top-k
    kval = min(k, M)
    topv, topi = K.topk(k=kval, dim=1)
    W = torch.zeros_like(K)
    W.scatter_(1, topi, topv)
    W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
    return W

def estimate_seam_distance_knn(points: torch.Tensor, labels: torch.Tensor, k: int = 32) -> torch.Tensor:
    """
    Approximate distance from each point to the nearest point of a different part label,
    using k-NN in the point cloud (k small). Returns (N,) distances.
    """
    with torch.no_grad():
        d = torch.cdist(points, points)  # (N,N) -- for large N you may want an approx kNN
        knn_idx = d.topk(k=min(k+1, points.shape[0]), largest=False).indices[:,1:]
        # For each point, find min distance among neighbors with different label
        N = points.shape[0]
        seam_dist = torch.full((N,), float("inf"), device=points.device, dtype=points.dtype)
        for i in range(N):
            nb = knn_idx[i]
            mask = (labels[nb] != labels[i])
            if mask.any():
                seam_dist[i] = d[i, nb[mask]].min()
        # Replace inf (no other-part neighbors within k) with a large value
        inf_mask = torch.isinf(seam_dist)
        if inf_mask.any():
            seam_dist[inf_mask] = d.max()
    return seam_dist

def build_weight_mask(points: torch.Tensor,
                      point_labels: torch.Tensor,
                      node_part: torch.Tensor,
                      seam_delta: float = 0.02) -> torch.Tensor:
    """
    Build a (N,M) binary mask for skinning weights:
    - Interior points (farther than seam_delta * bbox_diag from other-part points) may only
      take weights from nodes of the same part.
    - Seam points may blend with any nodes.
    """
    N = points.shape[0]
    M = node_part.shape[0]
    # Distance-based seam classification
    bbox_diag = (points.max(0).values - points.min(0).values).norm()
    seam_dist = estimate_seam_distance_knn(points, point_labels, k=32)
    interior = seam_dist > (seam_delta * bbox_diag)
    # Node part matrix
    # For each node j, allow for point i if (interior_i and node_part_j == label_i) or not interior
    same_part = (point_labels[:, None] == node_part[None, :])  # (N,M) bool
    allow_all = torch.ones((N, M), dtype=torch.bool, device=points.device)
    mask = torch.where(interior[:, None], same_part, allow_all)
    return mask.to(points.dtype)  # use 0/1 float for easy multiply



# =========================
# Model: node SE(3) + residual
# =========================
class RigRegDeformGraph(nn.Module):
    def __init__(self,
                 points_init: torch.Tensor,
                 node_pos: torch.Tensor,
                 skinning: torch.Tensor,
                 enable_residual: bool = True,
                 residual_hidden: int = 64):
        super().__init__()
        self.register_buffer("X0", points_init)        # (N,3)
        self.register_buffer("G0", node_pos)           # (M,3)
        self.register_buffer("W", skinning)            # (N,M) (base / hard weights snapshot)
        self.use_soft_weights = False                  # runtime toggle for soft/learnable weights
        self.weight_logits = None  # created lazily when enabling soft weights
        # Optional (N,M) mask where 0 disables a node for a given point
        self.register_buffer("W_mask", torch.ones_like(skinning))

        M = node_pos.shape[0]
        # SE(3) params per node
        self.so3_vec = nn.Parameter(torch.zeros(M,3))   # small at init
        self.trans = nn.Parameter(torch.zeros(M,3))

        # Soft hinge axis per edge will be handled in loss (learned axes parameters)
        self.enable_residual = enable_residual  # compile-time (MLP constructed or not)
        self.use_residual = enable_residual     # runtime flag whether to apply residual term
        if enable_residual:
            # residual MLP: input [blended-local-coord (3) , W (M)] -> delta (3)
            # use a bottleneck by projecting W to 16 via a small linear
            self.w_proj = nn.Linear(M, 16)
            self.mlp = nn.Sequential(
                nn.Linear(3+16, residual_hidden),
                nn.ReLU(),
                nn.Linear(residual_hidden, residual_hidden),
                nn.ReLU(),
                nn.Linear(residual_hidden, 3),
            )
        else:
            self.w_proj = None
            self.mlp = None

    def forward(self) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns: posed_points (N,3), extras dict
        """
        N, M = self.W.shape
        # Decide weights to use
        if self.use_soft_weights and self.weight_logits is not None:
            W_current = torch.softmax(self.weight_logits, dim=1)  # (N,M)
        else:
            W_current = self.W
        # Apply mask (0/1); renormalize rows; if any row sums to 0, fall back to unmasked
        Wm = W_current * self.W_mask
        row_sums = Wm.sum(dim=1, keepdim=True)
        fallback = (row_sums <= 1e-12).any()
        if fallback:
            W_current = W_current
        else:
            W_current = Wm / (row_sums + 1e-12)
        R = exp_so3(self.so3_vec)                  # (M,3,3)
        t = self.trans                             # (M,3)
        # Blend as classic deformation graph:
        # p' = sum_k w_ik [ R_k (p - g_k) + g_k + t_k ]
        p = self.X0[:,None,:].expand(N,M,3)        # (N,M,3)
        g = self.G0[None,:,:].expand(N,M,3)        # (N,M,3)
        Rp = torch.einsum("mkj,nmj->nmk", R, (p - g))
        q = Rp + g + t[None,:,:]                   # (N,M,3)
        W = W_current[:,:,None]                    # (N,M,1)
        skinned = (W * q).sum(dim=1)               # (N,3)

        extras = {"R":R, "t":t, "skinned": skinned, "W_used": W_current}

        if self.enable_residual and self.use_residual:
            # build blended rotation and translation per point for local coords
            # blended rotation: R_b = sum_k w_ik R_k (not strictly orthonormal; OK for feature)
            Rb = torch.einsum("nm,nmkj->nkj", W_current, R[None,:,:,:])  # (N,3,3)
            tb = torch.einsum("nm,mj->nj", W_current, t)                 # (N,3)
            local = torch.einsum("nij,nj->ni", Rb.transpose(1,2), (skinned - tb))
            wfeat = self.w_proj(W_current)
            mlp_in = torch.cat([local, wfeat], dim=1)
            delta_local = self.mlp(mlp_in)
            delta_world = torch.einsum("nij,nj->ni", Rb, delta_local)
            posed = skinned + delta_world
            extras.update({"delta": delta_world, "local": local})
            return posed, extras
        else:
            return skinned, extras

    # ---------- runtime control helpers ----------
    def enable_soft_weights(self):
        if self.weight_logits is None:
            with torch.no_grad():
                init = (self.W + 1e-8).log()  # log-prob init
            self.weight_logits = nn.Parameter(init.clone())
        self.use_soft_weights = True

    def disable_soft_weights(self, bake: bool = False, hard: bool = False):
        if bake and self.weight_logits is not None:
            with torch.no_grad():
                W_soft = torch.softmax(self.weight_logits, dim=1)
                if hard:
                    idx = W_soft.argmax(dim=1)
                    W_hard = torch.zeros_like(W_soft)
                    W_hard[torch.arange(W_hard.shape[0]), idx] = 1.0
                    self.W.copy_(W_hard)
                else:
                    self.W.copy_(W_soft)
        self.use_soft_weights = False
        # keep logits for later stages if needed

    def set_weight_mask(self, mask: torch.Tensor):
        """mask: (N,M) with 0/1 entries"""
        if mask.shape != self.W.shape:
            raise ValueError("mask shape must match W")
        self.W_mask = mask.detach().clone()


    def set_residual_usage(self, flag: bool):
        if not self.enable_residual:
            return
        self.use_residual = flag


# =========================
# Losses: OT with normals, hinge/ball priors, seams, ARAP-residual
# =========================
def sinkhorn_transport(a, b, C, eps=0.05, nits=100):
    """
    Stabilized Sinkhorn (scales only; NOT entropy duals) for uniform (or given) a,b.
    Operates in float32 to avoid half underflow. Returns P,u,v (float32).
    """
    C32 = C.float()
    # Clamp exponent argument to avoid exp under/overflow
    K = torch.exp((-C32 / eps).clamp(min=-60.0, max=60.0))
    # Add tiny floor to avoid exactly zero rows/cols
    K = K + 1e-12
    a32 = a.float()
    b32 = b.float()
    # Normalize a,b (safety)
    a32 = a32 / a32.sum()
    b32 = b32 / b32.sum()
    # Initialize scalings
    u = torch.ones_like(a32)
    v = torch.ones_like(b32)
    for _ in range(nits):
        Kv = K @ v
        Kv = Kv + 1e-12
        u = a32 / Kv
        KTu = K.t() @ u
        KTu = KTu + 1e-12
        v = b32 / KTu
    # Transport plan
    P = (u[:,None] * K) * v[None,:]
    return P, u, v

def ot_normal_cost(Xs, Ns, Xt, Nt, alpha_pos=1.0, beta_norm=0.1):
    """
    Build cost matrix combining position and normal alignment (sign-invariant).
    Xs: (Ns,3) source points
    Ns: (Ns,3) source normals
    Xt: (Nt,3) target points (posed)
    Nt: (Nt,3) target normals
    """
    # position cost
    D = torch.cdist(Xs, Xt)  # (Ns,Nt)
    Cpos = (D**2)
    # normal cost: 1 - |n_s · n_t|
    dot = (Ns[:,None,:] * Nt[None,:,:]).sum(-1).abs().clamp(0,1)
    Cn = 1.0 - dot
    C = alpha_pos * Cpos + beta_norm * Cn
    return C

def ot_loss_with_normals(source_pts: torch.Tensor,
                         source_normals: torch.Tensor,
                         posed_pts: torch.Tensor,
                         posed_normals: torch.Tensor,
                         eps: float = 0.05,
                         iters: int = 60,
                         alpha_pos: float = 1.0,
                         beta_norm: float = 0.1) -> torch.Tensor:
    Ns = source_pts.shape[0]
    Nt = posed_pts.shape[0]
    a = torch.full((Ns,), 1.0/Ns, device=source_pts.device)
    b = torch.full((Nt,), 1.0/Nt, device=posed_pts.device)
    C = ot_normal_cost(source_pts, source_normals, posed_pts, posed_normals,
                       alpha_pos=alpha_pos, beta_norm=beta_norm)
    P, _, _ = sinkhorn_transport(a, b, C, eps=eps, nits=iters)
    return (P * C).sum()

def seam_loss(points0: torch.Tensor, labels: torch.Tensor, k:int=8) -> torch.Tensor:
    """
    Encourage seam continuity by penalizing jumps across different part labels
    for nearest neighbors. Simple smoothness across labels.
    """
    d = torch.cdist(points0, points0)
    knn_idx = d.topk(k=k+1, largest=False).indices[:,1:]
    loss = 0.0
    for i in range(points0.shape[0]):
        pi = points0[i]
        li = labels[i]
        nb = knn_idx[i]
        pj = points0[nb]
        lj = labels[nb]
        mask = (lj != li)
        if mask.any():
            loss = loss + torch.norm(pi - pj[mask], dim=1).mean()
    return loss / points0.shape[0]

def arap_residual_loss(model: RigRegDeformGraph, k:int=8) -> torch.Tensor:
    """
    ARAP on residual only: compare (posed - skinned) over neighbors to be small / smooth.
    """
    posed, extras = model()
    delta = posed - extras["skinned"]
    d = torch.cdist(model.X0, model.X0)
    knn_idx = d.topk(k=k+1, largest=False).indices[:,1:]
    loss = 0.0
    for i in range(model.X0.shape[0]):
        nb = knn_idx[i]
        diff = delta[i] - delta[nb]
        loss = loss + (diff.norm(dim=1).mean())
    return loss / model.X0.shape[0]


def hinge_ball_prior(R: torch.Tensor,
                     E: torch.Tensor,
                     mode: str = "hinge",
                     axes: Optional[torch.Tensor] = None,
                     lam: float = 0.01) -> torch.Tensor:
    """
    R: (M,3,3) rotations per node
    E: (E,2) edges over nodes
    mode: 'hinge' or 'ball'; for hinge, encourage relative log-rot to align with axis
    axes: (E,3) learned unit axes per edge (if None, we infer from node positions outside)
    """
    if E.numel()==0:
        return torch.tensor(0.0, device=R.device)
    Rrel = torch.einsum("eij,ejk->eik", R[E[:,1]], R[E[:,0]].transpose(1,2))  # R_j R_i^T
    w = log_so3(Rrel)  # (E,3)
    if mode == "hinge":
        if axes is None:
            # fallback: encourage sparsity of rotation around any axis by L1 on orthogonal components
            # (this is weak; ideally provide axes)
            loss = (w[:,1:].abs().mean())  # arbitrary pick
        else:
            axes = axes / (axes.norm(dim=-1, keepdim=True)+1e-12)
            # penalize component of w orthogonal to axis
            proj = (w*axes).sum(-1, keepdim=True) * axes
            ortho = w - proj
            loss = (ortho.norm(dim=-1).mean())
    else:  # ball
        loss = (w.norm(dim=-1).mean())
    return lam * loss


# =========================
# Main entry
# =========================
def run_pipeline_deform_graph(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_segmentation: np.ndarray,
    log_dir: str,
    visual_scale: float = 1.0,
    *,  # keyword-only
    source_segmentation: Optional[np.ndarray] = None,
    nodes_per_part: str = "auto",
    elong_thresh: float = 1.7,
    k_skin: int = 8,
    sinkhorn_eps: float = 0.05,
    sinkhorn_iters: int = 60,
    alpha_pos: float = 1.0,
    beta_norm: float = 0.2,
    lambda_seam: float = 0.1,
    lambda_arap: float = 0.05,
    lambda_prior: float = 0.01,
    prior_mode: str = "hinge",
    iters: int = 600,
    lr: float = 5e-3,
    normal_k: int = 16,
    knn_recompute_every: int = 1,   # set >1 for speed (approximate)
    seam_arap_k: int = 8,
) -> np.ndarray:
    """
    Deformation-graph pipeline with normal-aware OT, designed to be called from the user's main.
    Returns posed target vertices as numpy (N,3).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xs = torch.from_numpy(source_vertices).float().to(device)   # (Ns,3)
    Xt0 = torch.from_numpy(target_vertices).float().to(device)  # (Nt,3)
    tgt_labels = torch.from_numpy(target_segmentation).long().to(device)

    # Build def graph on target (deformable) shape
    graph = build_deformation_graph(Xt0, tgt_labels, nodes_per_part=nodes_per_part, elong_thresh=elong_thresh)
    G = graph["G"]
    E = graph["E"]

    # Skinning weights
    W = gaussian_skinning(Xt0, G, k=k_skin)
    # Build part-aware weight mask (seam-aware)
    W_mask = build_weight_mask(Xt0, tgt_labels, graph['G_part'], seam_delta=0.02)


    # Model (residual network constructed but disabled initially)
    model = RigRegDeformGraph(Xt0, G, W, enable_residual=True).to(device)
    model.set_weight_mask(W_mask)
    model.set_residual_usage(False)

    # Normals (recomputed each time for posed points; but source normals once)
    Ns = estimate_normals_pca(Xs, k=16)
    # ---------- Per-part rigid ICP init (simplified) ----------
    def rigid_align(A: torch.Tensor, B: torch.Tensor):
        # find R,t minimizing ||R*A + t - B|| (A,B: (n,3))
        muA = A.mean(0)
        muB = B.mean(0)
        A0 = A - muA
        B0 = B - muB
        H = A0.t() @ B0
        U,S,Vt = torch.linalg.svd(H)
        R = Vt.t() @ U.t()
        if torch.det(R) < 0:
            Vt[-1] *= -1
            R = Vt.t() @ U.t()
        t = muB - (R @ muA)
        return R, t
    with torch.no_grad():
        parts = torch.unique(tgt_labels).tolist()
        for p in parts:
            tgt_mask = (tgt_labels==p)
            if source_segmentation is not None:
                src_labels = torch.from_numpy(source_segmentation).to(device)
                src_mask = (src_labels==p)
            else:
                src_mask = tgt_mask  # assume label spaces match
            if tgt_mask.sum()<4 or src_mask.sum()<4:
                continue
            R_p, t_p = rigid_align(Xt0[tgt_mask], Xs[src_mask])
            # apply to nodes of this part
            node_mask = (graph['G_part']==p)
            if node_mask.any():
                # set so3_vec via log map and trans = t_p
                so3 = log_so3(R_p[None,:,:])[0]
                model.so3_vec.data[node_mask] = so3
                model.trans.data[node_mask] = t_p

    # ---------- Visualization helpers ----------
    def _plot_pointcloud(
        points: torch.Tensor,
        save_dir: str,
        title: str,
        extra_points: Optional[torch.Tensor] = None,
        matches: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        transport_plan: Optional[torch.Tensor] = None,
        topk: int = 1,
        plan_threshold: Optional[float] = None,
        max_pairs: Optional[int] = 2000,
        plan_src_points: Optional[torch.Tensor] = None,
        plan_tgt_points: Optional[torch.Tensor] = None,
    ):
        pts = points.detach().cpu()
        fig = go.Figure()
        fig.add_trace(go.Scatter3d(x=pts[:,0], y=pts[:,1], z=pts[:,2], mode='markers', marker=dict(size=2,color='blue'), name='posed'))
        if extra_points is not None:
            ep = extra_points.detach().cpu()
            fig.add_trace(go.Scatter3d(x=ep[:,0], y=ep[:,1], z=ep[:,2], mode='markers', marker=dict(size=2,color='red'), name='source'))
        # Build correspondence pairs either from explicit matches or transport plan
        pairs = None
        if transport_plan is not None and extra_points is not None:
            P = transport_plan.detach()
            Ns_, Nt_ = P.shape
            if topk <= 1:
                j = P.argmax(dim=1)
                i = torch.arange(Ns_, device=P.device)
                pairs = torch.stack([i, j], dim=1).cpu()
            else:
                topv, topi = P.topk(k=topk, dim=1)
                if plan_threshold is not None:
                    mask = topv >= plan_threshold
                else:
                    mask = torch.ones_like(topv, dtype=torch.bool)
                idx_src = torch.arange(Ns_, device=P.device)[:, None].expand_as(topi)
                sel_i = idx_src[mask]
                sel_j = topi[mask]
                pairs = torch.stack([sel_i, sel_j], dim=1).cpu()
            if max_pairs is not None and pairs.shape[0] > max_pairs:
                perm = torch.randperm(pairs.shape[0])[:max_pairs]
                pairs = pairs[perm]
        elif matches is not None and extra_points is not None:
            try:
                pairs = matches.detach().cpu()
            except Exception:
                pairs = matches
            if hasattr(pairs, 'ndim') and pairs.ndim == 2 and pairs.shape[1] >= 2:
                pairs = pairs[:, :2]
            else:
                pairs = None
        # Draw correspondences if available
        if pairs is not None and extra_points is not None:
            # Use plan-specific point sets if provided (to align with P), else default to full clouds
            src_cloud = (plan_src_points if plan_src_points is not None else extra_points).detach().cpu()
            tgt_cloud = (plan_tgt_points if plan_tgt_points is not None else points).detach().cpu()
            Ns_tot, Nt_tot = src_cloud.shape[0], tgt_cloud.shape[0]
            lines_x=[];lines_y=[];lines_z=[]
            for ij in np.asarray(pairs):
                i, j = int(ij[0]), int(ij[1])
                if i < 0 or i >= Ns_tot or j < 0 or j >= Nt_tot:
                    continue
                lines_x += [src_cloud[i,0].item(), tgt_cloud[j,0].item(), None]
                lines_y += [src_cloud[i,1].item(), tgt_cloud[j,1].item(), None]
                lines_z += [src_cloud[i,2].item(), tgt_cloud[j,2].item(), None]
            if lines_x:
                fig.add_trace(go.Scatter3d(x=lines_x,y=lines_y,z=lines_z,mode='lines', line=dict(color='green',width=1), name='corr', opacity=0.7))
        fig.update_layout(title=title, scene=dict(aspectmode='data'))
        os.makedirs(save_dir, exist_ok=True)
        fig.write_html(os.path.join(save_dir, f"{title}.html"))

    def _export_mesh(points: torch.Tensor, faces: np.ndarray, out_path: str):
        try:
            import trimesh
            mesh = trimesh.Trimesh(vertices=points.detach().cpu().numpy(), faces=faces, process=False)
            mesh.export(out_path)
        except Exception as e:
            print(f"[mesh export] skipped: {e}")

    # ---------- Stage training loop template ----------
    def train_stage(stage_idx: int,
                    steps: int,
                    optimize_transforms: bool,
                    optimize_weights: bool,
                    enable_residual_flag: bool,
                    optimize_residual: bool,
                    stage_lr: float):
        nonlocal model
        stage_name = f"stage{stage_idx}"
        print(f"\n=== {stage_name} | steps={steps} ===")
        # Configure runtime flags
        model.set_residual_usage(enable_residual_flag)
        # Freeze / un-freeze main params
        model.so3_vec.requires_grad_(optimize_transforms)
        model.trans.requires_grad_(optimize_transforms)
        if optimize_weights:
            model.enable_soft_weights()
            model.weight_logits.requires_grad_(True)
        else:
            if model.use_soft_weights:
                model.disable_soft_weights(bake=True, hard=False)
            if model.weight_logits is not None:
                model.weight_logits.requires_grad_(False)
        if model.enable_residual:
            for p in list(model.mlp.parameters()) + [model.w_proj]:
                p.requires_grad_(optimize_residual)
        # Collect params
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(params, lr=stage_lr)
        axes = None
        if prior_mode == "hinge" and E.numel()>0 and optimize_transforms:
            axes = torch.randn(E.shape[0], 3, device=device, requires_grad=True)
            optimizer.add_param_group({"params": [axes], "lr": stage_lr})
        best_loss = float('inf')
        best_posed = None
        knn_idx = compute_knn(Xt0, normal_k)
        pbar = tqdm(range(steps), desc=stage_name, dynamic_ncols=True)
        last_P = None
        for it in pbar:
            if (it % knn_recompute_every) == 0:
                with torch.no_grad():
                    posed_tmp, _ = model()
                knn_idx = compute_knn(posed_tmp.detach(), max(normal_k, seam_arap_k))
            optimizer.zero_grad(set_to_none=True)
            posed, extras = model()
            normals_knn = knn_idx[:, :normal_k]
            Nt_est = estimate_normals_pca_knn(posed.detach(), normals_knn)
            # Stage-specific OT schedule
            if stage_idx == 1:
                eps_stage = max(0.10, float(sinkhorn_eps))
                beta_stage = 0.0
            elif stage_idx == 2:
                eps_stage = max(0.06, float(sinkhorn_eps))
                beta_stage = float(beta_norm)
            else:
                eps_stage = float(min(0.04, sinkhorn_eps))
                beta_stage = float(beta_norm)
            # Optional balanced downsampling for cost (especially Stage 1)
            use_ds = (stage_idx == 1)
            if use_ds:
                # If source_segmentation is provided, sample evenly per part; else uniform
                if source_segmentation is not None:
                    src_labels_t = torch.from_numpy(source_segmentation).long().to(device)
                else:
                    src_labels_t = None
                # pick up to 500 per part or 4000 total
                max_per_part = 500
                max_total = 4000
                idx_s = []
                if src_labels_t is not None:
                    for p in torch.unique(tgt_labels):
                        src_idx = torch.nonzero(src_labels_t==p, as_tuple=False).squeeze(-1)
                        if src_idx.numel() == 0:
                            continue
                        take = min(max_per_part, src_idx.numel())
                        perm = torch.randperm(src_idx.numel(), device=device)[:take]
                        idx_s.append(src_idx[perm])
                    idx_s = torch.cat(idx_s) if len(idx_s)>0 else torch.arange(Xs.shape[0], device=device)
                else:
                    take = min(max_total, Xs.shape[0])
                    idx_s = torch.randperm(Xs.shape[0], device=device)[:take]
                # target side: downsample posed to similar size
                take_t = min(idx_s.shape[0], posed.shape[0])
                idx_t = torch.randperm(posed.shape[0], device=device)[:take_t]
                Xs_cost = Xs[idx_s]
                posed_cost = posed[idx_t]
                Ns_cost = Ns[idx_s]
                Nt_cost = Nt_est[idx_t]
            else:
                Xs_cost = Xs
                posed_cost = posed
                Ns_cost = Ns
                Nt_cost = Nt_est
            # OT cost (positions + normals)
            x2s = (Xs_cost ** 2).sum(-1, keepdim=True)
            x2t = (posed_cost ** 2).sum(-1, keepdim=True).t()
            D2 = (x2s + x2t - 2 * Xs_cost @ posed_cost.t()).clamp_min(0)
            Cpos = D2
            dot = (Ns_cost[:, None, :] * Nt_cost[None, :, :]).sum(-1).abs().clamp(0, 1)
            Cn = 1.0 - dot
            C = alpha_pos * Cpos + beta_stage * Cn
            a = torch.full((Xs_cost.shape[0],), 1.0 / Xs_cost.shape[0], device=device)
            b = torch.full((posed_cost.shape[0],), 1.0 / posed_cost.shape[0], device=device)
            P, _, _ = sinkhorn_transport(a, b, C, eps=eps_stage, nits=sinkhorn_iters)
            last_P = P.detach()
            L_ot = (P * C.float()).sum()
            seam_arap_knn = knn_idx[:, :seam_arap_k]
            L_seam, L_arap = fused_seam_arap_losses(
                posed, extras['skinned'], tgt_labels, seam_arap_knn,
                lambda_seam=lambda_seam, lambda_arap=lambda_arap
            )
            L_prior = hinge_ball_prior(
                extras['R'], E, mode=prior_mode,
                axes=axes if (prior_mode == 'hinge' and axes is not None) else None,
                lam=lambda_prior
            )
            loss = L_ot + L_seam + L_arap + L_prior
            if not torch.isfinite(loss):
                print("Non-finite loss encountered; skipping step")
                continue
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_posed = posed.detach().cpu().numpy()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})
            # Periodic visualization
            if (it % 100) == 0:
                inter_dir = os.path.join(log_dir, f"{stage_name}_intermediate")
                os.makedirs(inter_dir, exist_ok=True)
                _plot_pointcloud(
                    posed, inter_dir,
                    f"{stage_name}_it{it:04d}", extra_points=Xs, matches=None,
                    transport_plan=last_P, topk=1, max_pairs=2000,
                    plan_src_points=Xs_cost, plan_tgt_points=posed_cost,
                )
        # End-of-stage artifacts
        with torch.no_grad():
            posed_final, _ = model()
        out_dir = os.path.join(log_dir, stage_name)
        os.makedirs(out_dir, exist_ok=True)
        posed_np = posed_final.detach().cpu().numpy()
        if source_faces is not None and len(source_faces) > 0:
            _export_mesh(posed_final, source_faces, os.path.join(out_dir, f"posed_{stage_name}.obj"))
        else:
            np.save(os.path.join(out_dir, f"posed_{stage_name}.npy"), posed_np)
        try:
            match_cols = last_P.argmax(dim=1)
            matches = torch.stack([torch.arange(match_cols.shape[0], device=device), match_cols], dim=1)
        except Exception:
            matches = None
        _plot_pointcloud(
            posed_final,
            out_dir,
            f"posed_{stage_name}",
            extra_points=Xs,
            matches=matches,
            transport_plan=last_P,
            topk=1,
            max_pairs=4000,
            plan_src_points=Xs_cost,
            plan_tgt_points=posed_cost,
        )
        print(f"{stage_name} finished | best loss {best_loss:.4f}")
        return best_posed if best_posed is not None else posed_np

    # ----------------- STAGES -----------------
    total_start = time.time()
    # Stage 1: rigid only (hard weights, no residual)
    model.disable_soft_weights(bake=True, hard=True)
    stage1_posed = train_stage(1, iters//4, optimize_transforms=True, optimize_weights=False, enable_residual_flag=False, optimize_residual=False, stage_lr=lr)
    # Stage 2: optimize skinning weights (soft), freeze transforms, still no residual
    stage2_posed = train_stage(2, iters//4, optimize_transforms=False, optimize_weights=True, enable_residual_flag=False, optimize_residual=False, stage_lr=lr*0.5)
    # Stage 3: residual MLP only
    stage3_posed = train_stage(3, iters//4, optimize_transforms=False, optimize_weights=False, enable_residual_flag=True, optimize_residual=True, stage_lr=lr*0.5)
    # Stage 4: joint fine-tune
    stage4_posed = train_stage(4, iters - 3*(iters//4), optimize_transforms=True, optimize_weights=True, enable_residual_flag=True, optimize_residual=True, stage_lr=lr*0.25)
    total_time = time.time() - total_start
    print(f"All stages complete in {total_time:.2f}s")
    return stage4_posed

def fused_seam_arap_losses(posed: torch.Tensor,
                           skinned: torch.Tensor,
                           labels: torch.Tensor,
                           knn_idx: torch.Tensor,
                           lambda_seam: float,
                           lambda_arap: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized seam continuity + ARAP residual losses sharing the same KNN.
    posed: (N,3) current posed points
    skinned: (N,3) purely skinned points (before residual)
    labels: (N,) part labels
    knn_idx: (N,k) neighbor indices
    """
    # Seam loss (cross-label neighbor distance)
    pj = posed[knn_idx]                 # (N,k,3)
    pi = posed[:, None, :]              # (N,1,3)
    diff = pi - pj                      # (N,k,3)
    seam_mask = (labels[knn_idx] != labels[:, None])  # (N,k) bool
    seam_dist = diff.norm(dim=-1)       # (N,k)
    seam_num = (seam_dist * seam_mask).sum(-1)
    seam_den = seam_mask.sum(-1).clamp_min(1)
    seam_row = seam_num / seam_den
    L_seam = seam_row.mean() * lambda_seam

    # ARAP residual on delta
    delta = posed - skinned             # (N,3)
    delta_nb = delta[knn_idx]           # (N,k,3)
    arap_row = ( (delta[:, None, :] - delta_nb).norm(dim=-1) ).mean(-1)  # (N,)
    L_arap = arap_row.mean() * lambda_arap
    return L_seam, L_arap
