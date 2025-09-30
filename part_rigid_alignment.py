"""
part_rigid_alignment.py
-----------------------
Graph-regularized per-part rigid alignment (one SE(3) per part) with joint ICP.

Key entrypoint:
    estimate_part_rigid_transforms(source_parts, target_parts, part_graph, anchors=None, params=None) -> dict

Where:
- source_parts, target_parts: dict[str or int] -> dict with fields:
    {
        "points": (N,3) float32/64 numpy array (required),
        "normals": (N,3) numpy array (optional but recommended for point-to-plane),
        # optional precomputed local frame 3x3 under key "frame"
        # optional precomputed centroid under key "centroid"
    }
- part_graph: list of (i, j, w) undirected edges (i, j labels drawn from keys of parts dicts; weight w>0)
- anchors (optional): dict of (i, j) -> (M,3) array of boundary anchor points sampled on the SOURCE near the interface of parts i and j
- params: dict to override defaults (see DEFAULTS below)

Returns:
- transforms: dict[part_id] -> {"R": (3,3) rotation matrix, "t": (3,) translation vector}

Notes:
- This module keeps *your* plotting/mesh export untouched: just call the returned transforms on your data.
- KD-tree optional: if sklearn is present, we use it; else we do a brute-force NN (works for prototypes, slower).
"""

import copy
import itertools
import os
import shutil
import warnings
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import trimesh

# -------------------------------
# Numerics: SO(3)/SE(3) utilities
# -------------------------------

EPS = 1e-9

def skew(v):
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=v.dtype)

def so3_exp(omega):
    """Exponential map from so(3) to SO(3)."""
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        # Use first-order Taylor
        K = skew(omega)
        return np.eye(3, dtype=omega.dtype) + K
    axis = omega / theta
    K = skew(axis)
    s, c = np.sin(theta), np.cos(theta)
    return c*np.eye(3) + s*K + (1-c)*np.outer(axis, axis)

def so3_log(R):
    """Log map from SO(3) to so(3) (as a 3-vector)."""
    # Clamp trace for numerical stability
    tr = np.clip(np.trace(R), -1.0, 3.0)
    cos_theta = 0.5*(tr - 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3, dtype=R.dtype)
    # Robust axis extraction
    omega_hat = (1.0/(2.0*np.sin(theta))) * np.array([
        R[2,1] - R[1,2],
        R[0,2] - R[2,0],
        R[1,0] - R[0,1]
    ], dtype=R.dtype)
    return theta * omega_hat

def se3_exp(xi):
    """Exponential map from se(3) to SE(3). xi = [omega(3), v(3)]"""
    omega = xi[:3]
    v = xi[3:]
    R = so3_exp(omega)
    theta = np.linalg.norm(omega)
    if theta < 1e-12:
        V = np.eye(3, dtype=xi.dtype) + 0.5*skew(omega)
    else:
        K = skew(omega/theta)
        s, c = np.sin(theta), np.cos(theta)
        V = (np.eye(3) + (1-c)/theta * K + (theta - s)/(theta) * (K @ K))
    t = V @ v
    T = np.eye(4, dtype=xi.dtype)
    T[:3,:3] = R
    T[:3, 3] = t
    return T

def se3_apply(R, t, X, center=None):
    """Apply SE(3) to points: Y = R (X - center) + center + t.
    If center is None, acts as standard Y = R X + t.
    """
    if center is None:
        return (X @ R.T) + t[None, :]
    else:
        return ((X - center[None,:]) @ R.T) + center[None,:] + t[None,:]


def _clone_states(states):
    """Deep-clone states dict while preserving numpy arrays."""
    cloned = {}
    for k, state in states.items():
        cloned_state = {}
        for field, value in state.items():
            if isinstance(value, np.ndarray):
                cloned_state[field] = value.copy()
            else:
                cloned_state[field] = copy.deepcopy(value)
        cloned[k] = cloned_state
    return cloned


def _axis_flip_matrix(axis: str) -> np.ndarray:
    axis = axis.lower()
    if axis == "x":
        return np.diag([1.0, -1.0, -1.0])  # keep +X, flip Y/Z
    if axis == "y":
        return np.diag([-1.0, 1.0, -1.0])  # keep +Y, flip X/Z
    if axis == "z":
        return np.diag([-1.0, -1.0, 1.0])  # keep +Z, flip X/Y
    raise ValueError(f"Unsupported axis '{axis}' for flip test")


def _enumerate_flip_mats(axes):
    axes = [a.lower() for a in axes if a]
    mats = [np.eye(3)]
    seen = {tuple(np.eye(3).reshape(-1))}
    for axis in axes:
        M = _axis_flip_matrix(axis)
        key = tuple(np.round(M, decimals=8).reshape(-1))
        if key not in seen:
            mats.append(M)
            seen.add(key)
    return mats

# -------------------------------
# Correspondence utilities
# -------------------------------

def build_kdtree(points):
    try:
        from sklearn.neighbors import KDTree
        return KDTree(points), "sklearn"
    except Exception:
        return points.copy(), "bruteforce"

def nn_query(tree, backend, Q, k=1):
    if backend == "sklearn":
        ind = tree.query(Q, k=k, return_distance=False)
        return ind.squeeze(-1)
    else:
        # brute force
        # Q: (M,3), tree: (N,3)
        diffs = Q[:,None,:] - tree[None,:,:]  # (M,N,3)
        d2 = np.sum(diffs*diffs, axis=2)      # (M,N)
        return np.argmin(d2, axis=1)

def trimmed_matches(src_pts_t, tgt_pts, tgt_nrm=None, keep_ratio=0.8):
    """Nearest-neighbor correspondences with trimming by distance quantile.
    Returns indices (match_tgt), distances, and normals (if available).
    """
    tree, be = build_kdtree(tgt_pts)
    nn = nn_query(tree, be, src_pts_t, k=1)  # (M,)
    matched = tgt_pts[nn]
    diffs = src_pts_t - matched
    d = np.linalg.norm(diffs, axis=1)
    q = np.quantile(d, keep_ratio)
    mask = d <= q
    if tgt_nrm is not None:
        return mask, nn, d, tgt_nrm[nn]
    else:
        return mask, nn, d, None


def sinkhorn_matches(
    src_pts_t: np.ndarray,
    tgt_pts: np.ndarray,
    *,
    keep_ratio: float = 0.8,
    src_normals: Optional[np.ndarray] = None,
    tgt_normals: Optional[np.ndarray] = None,
    sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
):
    """Optimal-transport correspondences via Sinkhorn plan (argmax per source).

    Returns mask, nn indices, distances. Normals are intentionally omitted so the caller
    can build point-to-point directions by default.
    """
    if src_pts_t.size == 0 or tgt_pts.size == 0:
        mask = np.zeros(src_pts_t.shape[0], dtype=bool)
        nn = np.zeros(src_pts_t.shape[0], dtype=np.int64)
        d = np.zeros(src_pts_t.shape[0], dtype=np.float64)
        return mask, nn, d

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Sinkhorn-based matching requires PyTorch.") from exc

    try:
        from smpl_skinning_ot_local import sinkhorn_transport_plan as _sinkhorn_transport_plan
    except ImportError as exc:
        raise RuntimeError(
            "sinkhorn_transport_plan not available; ensure smpl_skinning_ot_local.py is importable"
        ) from exc

    params = dict(sinkhorn_kwargs or {})
    eps = params.pop("eps", 0.01)
    tau = params.pop("tau", 0.05)
    n_iters = params.pop("n_iters", 40)
    pos_weight = params.pop("pos_weight", 0.75)
    normal_weight = params.pop("normal_weight", 0.25)
    device = params.pop("device", "cpu")

    if params:
        unknown = ", ".join(sorted(params.keys()))
        raise ValueError(f"Unsupported sinkhorn_kwargs entries: {unknown}")

    device = torch.device(device)

    src_tensor = torch.as_tensor(src_pts_t, dtype=torch.float32)
    tgt_tensor = torch.as_tensor(tgt_pts, dtype=torch.float32)
    if src_tensor.device != device:
        src_tensor = src_tensor.to(device)
    if tgt_tensor.device != device:
        tgt_tensor = tgt_tensor.to(device)

    cost_pos = torch.cdist(src_tensor, tgt_tensor) ** 2
    C = pos_weight * cost_pos

    if src_normals is not None and tgt_normals is not None:
        src_norm_tensor = torch.as_tensor(src_normals, dtype=torch.float32)
        tgt_norm_tensor = torch.as_tensor(tgt_normals, dtype=torch.float32)
        if src_norm_tensor.device != device:
            src_norm_tensor = src_norm_tensor.to(device)
        if tgt_norm_tensor.device != device:
            tgt_norm_tensor = tgt_norm_tensor.to(device)
        cost_normal = torch.cdist(src_norm_tensor, tgt_norm_tensor) ** 2
        C = C + normal_weight * cost_normal

    P = _sinkhorn_transport_plan(C, eps=eps, tau=tau, n_iters=n_iters)
    nn_indices = torch.argmax(P, dim=1)

    nn = nn_indices.detach().cpu().numpy().astype(np.int64)
    diffs = src_pts_t - tgt_pts[nn]
    d = np.linalg.norm(diffs, axis=1)

    keep_ratio_clamped = float(np.clip(keep_ratio, 0.0, 1.0))
    if keep_ratio_clamped <= 0.0:
        q = d.min() if d.size > 0 else 0.0
    else:
        q = np.quantile(d, keep_ratio_clamped)
    mask = d <= q
    return mask, nn, d

# -------------------------------
# System assembly (Gauss-Newton)
# -------------------------------

def assemble_system(parts_keys, states, data_terms, smooth_terms, boundary_terms, prior_terms, lm_damp=1e-6):
    """Assemble normal equations A^T A delta = A^T b across all residuals.
    Unknown ordering: for key k at index idx -> block slice [6*idx:6*idx+6]
    Each residual is linearized as J delta = -r.
    Returns H (6P, 6P) and g (6P,).
    """
    P = len(parts_keys)
    index_of = {k:i for i,k in enumerate(parts_keys)}
    H = np.zeros((6*P, 6*P), dtype=np.float64)
    g = np.zeros((6*P,), dtype=np.float64)

    def add_block(i, j, J_i, J_j, r, w=1.0):
        """Add residual with Jacobians wrt part i and j (either may be None).
        r is shape (m,), J_* are (m,6).
        """
        if w <= 0: return
        if r.ndim == 0:
            r = r[None]
        # Weighting per residual vector: sqrt(w)
        sw = np.sqrt(w)
        r = sw * r
        if J_i is not None:
            J_i = sw * J_i
        if J_j is not None:
            J_j = sw * J_j

        if J_i is not None:
            ii = 6*index_of[i]; iis = slice(ii, ii+6)
            H[iis, iis] += J_i.T @ J_i
            g[iis]      += J_i.T @ (-r)
        if J_j is not None:
            jj = 6*index_of[j]; jjs = slice(jj, jj+6)
            H[jjs, jjs] += J_j.T @ J_j
            g[jjs]      += J_j.T @ (-r)
        if J_i is not None and J_j is not None and i != j:
            ii = 6*index_of[i]; iis = slice(ii, ii+6)
            jj = 6*index_of[j]; jjs = slice(jj, jj+6)
            H[iis, jjs] += J_i.T @ J_j
            H[jjs, iis] += J_j.T @ J_i

    # Data terms: list of entries per part: each entry is (x, y, n, center, weight)
    for k, entries in data_terms.items():
        R = states[k]["R"]; t = states[k]["t"]; c = states[k]["center"]
        for (x, y, n, w) in entries:
            # residual: r = n^T ( (R*(x-c)+c + t) - y )
            # Rxmc = R @ (x - c)
            # p = Rxmc + c + t
            # use se3_apply for clarity
            p = se3_apply(R, t, x[None,:], center=c)[0]
            r = n.dot(p - y)  # scalar

            # Jacobians wrt δω, δt (for this part k only)
            # dp/dω = R * [x-c]_x * δω  (note: derivative is R * (ω x (x-c)), equivalently -R*[x-c]_x*δω if using right perturbation)
            # For standard left-multiplicative update exp(δω^) R, dp/dω ≈ - R [x-c]_x δω
            Jw = - (n @ (R @ skew(x - c)))  # shape (3,3) -> (1,3) after n@
            # Above n@(R@S) yields a row vector: (1,3) mapping δω -> scalar
            Jw = Jw.reshape(1,3)
            Jt = n.reshape(1,3)
            J = np.concatenate([Jw, Jt], axis=1)  # (1,6)
            add_block(k, None, J, None, np.array([r]), w=w)

    # Smooth terms: list of (i, j, w, c_i, c_j)
    for (i, j, w_s, c_i, c_j) in smooth_terms:
        Ri = states[i]["R"]; ti = states[i]["t"]
        Rj = states[j]["R"]; tj = states[j]["t"]
        # ED residual (3):
        # r = (Ri (c_j - c_i) + c_i + ti) - (Rj (c_j - c_j) + c_j + tj)
        #   = Ri (c_j - c_i) + ti + c_i - (tj + c_j)
        v_ij = (c_j - c_i)
        r = (Ri @ v_ij + ti + c_i) - (tj + c_j)  # (3,)

        # Jacobians
        # wrt i: dr/dω_i = Ri * [v_ij]_x * δω_i (with same sign convention as above -> -Ri [v_ij]_x)
        Jw_i = - (Ri @ skew(v_ij))  # (3,3)
        Jt_i = np.eye(3)
        J_i = np.concatenate([Jw_i, Jt_i], axis=1)  # (3,6)

        # wrt j:
        # r = ... - (tj + c_j), no rotation term of j appears (with this simplified ED variant)
        # If you prefer the symmetric ED, you could include Rj (c_i - c_j) + c_j + tj on RHS.
        Jw_j = np.zeros((3,3))
        Jt_j = -np.eye(3)
        J_j = np.concatenate([Jw_j, Jt_j], axis=1)  # (3,6)

        add_block(i, j, J_i, J_j, r, w=w_s)

    # Boundary terms: list of entries ((i,j), [b_k], w_b)
    for (i, j, b_list, w_b) in boundary_terms:
        Ri = states[i]["R"]; ti = states[i]["t"]; ci = states[i]["center"]
        Rj = states[j]["R"]; tj = states[j]["t"]; cj = states[j]["center"]
        for b in b_list:
            # r = (Ri (b-ci) + ci + ti) - (Rj (b-cj) + cj + tj)
            r = (Ri @ (b - ci) + ci + ti) - (Rj @ (b - cj) + cj + tj)

            Jw_i = - (Ri @ skew(b - ci))
            Jt_i = np.eye(3)
            J_i  = np.concatenate([Jw_i, Jt_i], axis=1)

            Jw_j = + (Rj @ skew(b - cj))  # sign flips
            Jt_j = -np.eye(3)
            J_j  = np.concatenate([Jw_j, Jt_j], axis=1)

            add_block(i, j, J_i, J_j, r, w=w_b)

    # Prior terms: list of (k, w_p, R0)
    for (k, w_p, R0) in prior_terms:
        R = states[k]["R"]
        r_geo = so3_log(R @ R0.T)  # (3,)
        # Jacobian ≈ identity near zero (sufficient as a proximal term)
        Jw = np.eye(3)
        Jt = np.zeros((3,3))
        J = np.concatenate([Jw, Jt], axis=1)  # (3,6)
        add_block(k, None, J, None, r_geo, w=w_p)

    # Levenberg damping (on all dofs)
    H += lm_damp * np.eye(H.shape[0])
    return H, g

def solve_system(H, g):
    # Solve H delta = g in least-squares (H is SPD-ish)
    try:
        delta = np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        delta, *_ = np.linalg.lstsq(H, g, rcond=None)
    return delta

# -------------------------------
# Initialization helpers
# -------------------------------

def robust_centroid(X):
    # geometric median would be better; use trimmed mean as simple robust proxy
    if X.shape[0] <= 8:
        return X.mean(axis=0)
    d = np.linalg.norm(X - X.mean(axis=0), axis=1)
    mask = d <= np.quantile(d, 0.9)
    return X[mask].mean(axis=0)

def pca_frame(X, normals=None, up_hint=None, frontal_hint=None, part_name=None):
    """Build a (z, x, y) right-handed frame from points.
    z: longest PCA axis (optionally flipped to align with up_hint), 
    x: smallest variance axis (optionally guided by frontal_hint),
    y = z x x.
    Return 3x3 rotation with columns [x, y, z].
    """
    Xc = X - X.mean(axis=0)
    C = Xc.T @ Xc
    eigvals, eigvecs = np.linalg.eigh(C)  # ascending
    axes = eigvecs  # columns are eigenvectors
    x = axes[:,0]
    z = axes[:,-1]
    
    # Handle up_hint: just flip z if pointing in opposite direction
    if up_hint is not None:
        if part_name is not None:
            print(f"Part {part_name} Using up hint for PCA frame.")
        # Ensure z and up_hint have positive dot product
        if np.dot(z, up_hint) < 0:
            z = -z
    
    # Handle frontal_hint: use it to guide x direction
    if frontal_hint is not None:
        if part_name is not None:
            print(f"Part {part_name} Using frontal hint for PCA frame.")
        # Project frontal_hint onto the plane perpendicular to z
        frontal_proj = frontal_hint - np.dot(frontal_hint, z) * z
        frontal_proj_norm = np.linalg.norm(frontal_proj)
        
        # Check if projection is degenerate (frontal_hint is nearly parallel to z)
        if frontal_proj_norm > 1e-6:
            x = frontal_proj / frontal_proj_norm
        else:
            # Use frontal_hint directly as x (and orthogonalize later)
            x = frontal_hint / (np.linalg.norm(frontal_hint) + 1e-12)
    
    # Compute y = z x x and re-orthogonalize
    y = np.cross(z, x); y /= (np.linalg.norm(y) + 1e-12)
    x = np.cross(y, z); x /= (np.linalg.norm(x) + 1e-12)
    
    # outward sign for x via normals, if provided and no frontal_hint override
    if normals is not None and normals.shape[0] > 0 and frontal_hint is None:
        s = np.sign(np.mean(normals @ x))
        if s < 0: x = -x; y = -y  # keep right-handed
    
    R = np.stack([x, y, z], axis=1)
    # Orthonormalize
    u, _, vt = np.linalg.svd(R)
    R = u @ vt
    return R

# NEW: build per-part up vectors from adjacency graph
def build_up_hints_from_graph(source_parts, target_parts, part_graph, strategy="lexi"):
    """
    Returns two dicts:
      up_src[k] -> vector from source centroid of k to chosen neighbor centroid
      up_tgt[k] -> vector from target centroid of k to chosen neighbor centroid
    The chosen neighbor is the same id for source and target, picked deterministically.
    strategy: "farthest" (by source centroid distance) or "lexi" (lexicographically smallest neighbor id)
    """
    # Precompute centroids
    cs = {k: (np.asarray(v.get("centroid")) if "centroid" in v else robust_centroid(np.asarray(v["points"])))
          for k, v in source_parts.items()}
    ct = {k: (np.asarray(v.get("centroid")) if "centroid" in v else robust_centroid(np.asarray(v["points"])))
          for k, v in target_parts.items()}

    # Build adjacency
    neighbors = {k: [] for k in source_parts.keys()}
    for (i, j, _w) in part_graph:
        if i in neighbors: neighbors[i].append(j)
        if j in neighbors: neighbors[j].append(i)

    # Choose neighbor per part
    neighbor_of = {}
    for k, nbrs in neighbors.items():
        if not nbrs:
            continue
        if strategy == "lexi":
            # deterministic order by string/int representation
            neighbor_of[k] = sorted(nbrs, key=lambda x: str(x))[0]
        else:
            # farthest by source centroid distance
            ck = cs[k]
            best = max(nbrs, key=lambda j: float(np.linalg.norm(cs[j] - ck)))
            neighbor_of[k] = best

    # Build up vectors
    up_src = {}
    up_tgt = {}
    for k, j in neighbor_of.items():
        vs = cs[j] - cs[k]
        vt = ct[j] - ct[k]
        # guard tiny vectors
        if np.linalg.norm(vs) > 1e-12:
            up_src[k] = vs
        if np.linalg.norm(vt) > 1e-12:
            up_tgt[k] = vt
    return up_src, up_tgt

def init_transforms_from_frames(source_parts, target_parts, part_graph=None, up_hint_strategy="lexi", use_up_hints=False):
    """Return dict k -> {'R','t','center','R0'} using PCA frames and centroids.
       If part_graph is provided and use_up_hints is True, use centroid-to-neighbor vectors as up hints.
    """
    up_src, up_tgt = ({}, {})
    if use_up_hints and part_graph is not None:
        up_src, up_tgt = build_up_hints_from_graph(source_parts, target_parts, part_graph, strategy=up_hint_strategy)

    states = {}
    for k in source_parts.keys():
        Xs = source_parts[k]["points"]
        Xt = target_parts[k]["points"]
        ns = source_parts[k].get("normals", None)
        nt = target_parts[k].get("normals", None)
        cs = source_parts[k].get("centroid", None)
        ct = target_parts[k].get("centroid", None)
        if cs is None: cs = robust_centroid(Xs)
        if ct is None: ct = robust_centroid(Xt)

        Fs = source_parts[k].get("frame", None)
        Ft = target_parts[k].get("frame", None)
        if Fs is None:
            Fs = pca_frame(Xs, ns, up_hint=up_src.get(k, None), part_name=k)
        if Ft is None:
            Ft = pca_frame(Xt, nt, up_hint=up_tgt.get(k, None), part_name=k)

        R0 = Ft @ Fs.T
        t0 = ct - cs
        states[k] = {"R": R0.copy(), "t": t0.copy(), "center": cs.copy(), "R0": R0.copy()}
    return states

# -------------------------------
# Seam-aware helpers for robust child axis
# -------------------------------


def _seam_center(B):
    return robust_centroid(B)


def _nn_distances_to_set(X, Y):
    try:
        from sklearn.neighbors import KDTree

        tree = KDTree(Y)
        d, _ = tree.query(X, k=1, return_distance=True)
        return d.reshape(-1)
    except Exception:
        dif = X[:, None, :] - Y[None, :, :]
        d2 = (dif**2).sum(axis=2)
        return (d2.min(axis=1)) ** 0.5


def _weighted_pca_axis(X, w):
    w = w.reshape(-1, 1)
    sw = float(w.sum()) + 1e-12
    mu = (w * X).sum(axis=0) / sw
    Xc = X - mu
    C = (Xc * w).T @ Xc / sw
    vals, vecs = np.linalg.eigh(C)
    v = vecs[:, -1]
    v /= np.linalg.norm(v) + 1e-12
    return v


# -------------------------------
# Graph-consistent frame construction
# -------------------------------

def _fit_plane_normal(P):
    """Return unit normal of best-fit plane through P (PCA smallest eigenvector)."""
    if P.shape[0] < 3:
        return np.array([0.0, 0.0, 1.0])
    Pc = P - P.mean(axis=0)
    C = Pc.T @ Pc
    vals, vecs = np.linalg.eigh(C)  # ascending
    n = vecs[:, 0]
    n /= np.linalg.norm(n) + 1e-12
    return n


def _dominant_axis_in_plane(X, z, w=None, center=None):
    """Return dominant direction in plane orthogonal to z along with planar eigenvalues."""
    X = np.asarray(X)
    if center is None:
        center = robust_centroid(X)
    P = np.eye(3) - np.outer(z, z)  # projector onto plane orthogonal to z
    Xc = X - center
    Xp = Xc @ P.T
    if w is None:
        C = Xp.T @ Xp
    else:
        w = w.reshape(-1, 1)
        sw = float(w.sum()) + 1e-12
        C = (Xp * w).T @ Xp / sw
    vals, vecs = np.linalg.eigh(C)
    x = vecs[:, -1]
    x /= np.linalg.norm(x) + 1e-12
    lam1 = vals[-1]
    lam2 = vals[-2] if vals.shape[0] >= 2 else 0.0
    return x, (lam1, lam2)



# --------------------------------------------------------------
# Robust blended frame recipe (shape-first with graph anchoring)
# --------------------------------------------------------------
def recipe_blended_frames(
        parts, 
        part_graph, 
        root=None,
        up_hint=None,
        frontal_hint=None, 
        anchors=None, 
        params=None
    ) -> dict:
    """
    Build per-part frames using a robust *shape + graph* recipe.

    Goal:
      - Pose-invariant for identical shapes under different poses (use PCA when anisotropy is strong).
      - Flip-resistant across symmetric/round parts by using graph seam cues to fix axis *sign* and in-plane *spin*.

    Inputs:
      parts: dict[part_id] -> { "points": (N,3), optional "normals": (N,3), optional "centroid": (3,) }
      part_graph: iterable of (i, j, w) edges (w is unused here, but allowed)
      root: optional root part id; if None, pick the highest-degree node
      anchors: optional dict[(i,j)] -> (M,3) seam points (in *part i* coordinates) used to orient child w.r.t parent
      params: optional dict of hyper-parameters:
        - tau_aniso: float, anisotropy threshold for trusting PCA long axis (default 1.3)
        - prefer_parent_for_spin: bool, if True align x with parent seam projection when available (default True)
        - blend_spin_cues: bool, if True blend seam/parent/PCA cues for the in-plane axis instead of serial fallbacks (default False)
        - blend_spin_cues_signless: bool, if True fuse spin cues via signless second-moment blending (default True)
        - blend_include_neighbors: bool, if True add non-parent neighbor seams to blended spin cues (default True)
        - neighbor_consensus_sign: bool, if True use neighbor seams to pick spin sign on non-leaf parts (default True)
        - spin_flip_quality_min: float, minimum seam quality required before applying seam-based spin flips (default 5.0)
        - spin_ambiguous_ratio: float, anisotropy ratio below which planar PCA is treated as ambiguous for seam flips (default 1.15)
        - verbose: bool

    Returns:
      F: dict[part_id] -> 3x3 rotation matrix with columns [x, y, z]
    """
    import numpy as _np
    from collections import deque as _deque

    if params is None: params = {}
    tau_aniso = float(params.get("tau_aniso", 1.3))
    prefer_parent_for_spin = bool(params.get("prefer_parent_for_spin", True))
    blend_spin_cues = bool(params.get("blend_spin_cues", False))
    blend_spin_cues_signless = bool(params.get("blend_spin_cues_signless", True))
    blend_include_neighbors = bool(params.get("blend_include_neighbors", True))
    neighbor_consensus_sign = bool(params.get("neighbor_consensus_sign", True))
    spin_flip_quality_min = float(params.get("spin_flip_quality_min", 5.0))
    spin_ambiguous_ratio = float(params.get("spin_ambiguous_ratio", 1.15))
    verbose = bool(params.get("verbose", False))

    keys = list(parts.keys())
    if len(keys) == 0:
        return {}

    # Build adjacency (undirected)
    nbrs = {k: [] for k in keys}
    deg = {k: 0 for k in keys}
    for (i, j, *rest) in part_graph:
        if i in nbrs and j in nbrs:
            nbrs[i].append(j); nbrs[j].append(i)
            deg[i] += 1; deg[j] += 1

    # Pick a root if none given: highest degree (fallback to first key)
    if root is None:
        root = max(deg.items(), key=lambda kv: (kv[1], str(kv[0])))[0] if len(deg)>0 else keys[0]
    if verbose:
        print(f"[recipe_blended_frames] root = {root}")

    # Centroids
    cent = {k: (parts[k].get("centroid") if "centroid" in parts[k] else robust_centroid(_np.asarray(parts[k]["points"])))
            for k in keys}

    # Precompute PCA eigens for each part
    evals = {}
    evecs = {}
    for k in keys:
        X = _np.asarray(parts[k]["points"], dtype=_np.float64)
        Xc = X - cent[k]
        C = Xc.T @ Xc
        w, V = _np.linalg.eigh(C)  # ascending
        # if k == root:
        #     if up_hint is not None:
        #         if np.dot(V[:, -1], up_hint) < 0:
        #             V[:, -1] = -V[:, -1]

        #     # Handle frontal_hint: use it to guide x direction
        #     if frontal_hint is not None:
        #         # Project frontal_hint onto the plane perpendicular to z
        #         frontal_proj = frontal_hint - np.dot(frontal_hint, V[:, -1]) * V[:, -1]
        #         frontal_proj_norm = np.linalg.norm(frontal_proj)
                
        #         # Check if projection is degenerate (frontal_hint is nearly parallel to z)
        #         if frontal_proj_norm > 1e-6:
        #             V[:, 0] = frontal_proj / frontal_proj_norm
        #         else:
        #             # Use frontal_hint directly as x (and orthogonalize later)
        #             V[:, 0] = frontal_hint / (np.linalg.norm(frontal_hint) + 1e-12)

        evals[k] = w
        evecs[k] = V  # columns
    def _anisotropy(vals):
        # vals ascending
        l1, l2, l3 = float(vals[-1]), float(vals[-2]), float(vals[-3]) if vals.shape[0]>=3 else (float(vals[-1]), float(vals[-2]), 0.0)
        return (l1/(l2+1e-12), l2/(l3+1e-12))

    def _normalize(v):
        n = _np.linalg.norm(v)
        return v / (n + 1e-12)

    def _proj_plane(u, z):
        return u - (u @ z) * z

    def _seam_vec(i, j):
        """Return vector at part i pointing toward neighbor j (using anchors if available, else centroid-to-centroid)."""
        if anchors is not None and (i, j) in anchors:
            s_ij = _seam_center(anchors[(i, j)])
            return _normalize(s_ij - cent[i])
        else:
            v = cent[j] - cent[i]
            return _normalize(v) if _np.linalg.norm(v) > 1e-12 else v

    def _seam_quality(i, j):
        """A simple seam quality score for weighting; higher is better."""
        if anchors is not None and (i, j) in anchors:
            Bij = _np.asarray(anchors[(i, j)])
            # compactness via planar spread around seam plane
            if Bij.shape[0] >= 8:
                Pc = Bij - Bij.mean(axis=0)
                C = Pc.T @ Pc
                vals, _ = _np.linalg.eigh(C)  # ascending
                # smaller smallest eigenvalue => tighter seam -> higher quality
                return float(Bij.shape[0]) / (1.0 + float(vals[0]))
            else:
                return float(Bij.shape[0])
        else:
            # fallback: inverse distance weight from centroids
            v = cent[j] - cent[i]
            d = float(_np.linalg.norm(v))
            return 1.0 / (1e-6 + d)

    # Output frames
    F = {}

    # Root: choose z then x using local rules (no parent yet)
    a1_root, a2_root = _anisotropy(evals[root])
    if a1_root > tau_aniso:
        # if verbose:
        #     print(f"+++ [recipe_blended_frames] root {root} trusting PCA long axis (lam1/lam2={a1_root:.3f})")
        z_root = evecs[root][:, -1]
    else:
        if verbose:
            print(f"--- [recipe_blended_frames] root {root} weak PCA anisotropy (lam1/lam2={a1_root:.3f}), using seam cues")
        # use average of neighbor seams
        acc = _np.zeros(3, dtype=_np.float64)
        wsum = 0.0
        for j in nbrs[root]:
            wq = _seam_quality(root, j)
            acc += wq * _seam_vec(root, j); wsum += wq
        z_root = _normalize(acc) if wsum>1e-12 else evecs[root][:, -1]
    if up_hint is not None:
        if np.dot(z_root, up_hint) < 0:
            z_root = -z_root
    # x: prefer dominant in-plane PCA unless ambiguous; else best neighbor seam
    x_root, (lam1p, lam2p) = _dominant_axis_in_plane(parts[root]["points"], z_root, center=cent[root])
    if lam1p - lam2p < 1e-8:  # ambiguous
        if verbose:
            print(f"### [recipe_blended_frames] root {root} ambiguous planar axis (lam1={lam1p:.6f}, lam2={lam2p:.6f}), using seam cues")
        # use best-quality neighbor seam projected
        best = None; best_w = -_np.inf
        for j in nbrs[root]:
            wq = _seam_quality(root, j)
            cand = _proj_plane(_seam_vec(root, j), z_root)
            if _np.linalg.norm(cand) < 1e-8: continue
            if wq > best_w:
                best_w = wq; best = cand
        if best is not None and _np.linalg.norm(best) > 1e-8:
            x_root = _normalize(best)
    y_root = _normalize(_np.cross(z_root, x_root))
    x_root = _normalize(_np.cross(y_root, z_root))
    F[root] = _np.stack([x_root, y_root, z_root], axis=1)

    # BFS outwards, fixing sign and spin using parent seams
    q = _deque([root])
    visited = {root}
    while q:
        i = q.popleft()
        for j in nbrs[i]:
            if j in visited:
                continue

            # Step 1) primary axis z_j
            a1, a2 = _anisotropy(evals[j])
            if a1 > tau_aniso:
                # if verbose:
                #     print(f"+++++ [recipe_blended_frames] part {j} trusting PCA long axis (lam1/lam2={a1:.3f})")
                z_j = evecs[j][:, -1]
            else:
                if verbose:
                    print(f"----- [recipe_blended_frames] part {j} weak PCA anisotropy (lam1/lam2={a1:.3f}), using seam cues")
                # weighted sum of seams
                acc = _np.zeros(3, dtype=_np.float64)
                wsum = 0.0
                for k2 in nbrs[j]:
                    wq = _seam_quality(j, k2)
                    acc += wq * _seam_vec(j, k2); wsum += wq
                z_j = _normalize(acc) if wsum>1e-12 else evecs[j][:, -1]

            # Step 2) build in-plane cues by combining seam, parent, and geometric hints
            sp = _seam_vec(j, i)
            sp_proj = _proj_plane(sp, z_j)
            sp_proj_norm = _np.linalg.norm(sp_proj)
            sp_proj_norm_threshold = 0.1
            seam_quality = _seam_quality(j, i)

            x_parent = F[i][:, 0]
            x_parent_proj = _proj_plane(x_parent, z_j)
            n_xpp = _np.linalg.norm(x_parent_proj)
            x_parent_proj_unit = x_parent_proj / (n_xpp + 1e-12)

            is_leaf = len(nbrs.get(j, [])) <= 1
            w_plane = None
            if is_leaf and anchors is not None and (i, j) in anchors:
                Xj = parts[j]["points"]
                d = _nn_distances_to_set(Xj, anchors[(i, j)])
                q50, q90 = np.quantile(d, 0.5), np.quantile(d, 0.9)
                # q50, q90 = np.quantile(d, 0.2), np.quantile(d, 0.9)
                denom = max(q90 - q50, 1e-6)
                w_plane = np.clip((d - q50) / denom, 0.0, 1.0) ** 2
            x_e2, (lam1p, lam2p) = _dominant_axis_in_plane(
                parts[j]["points"], z_j, w=w_plane, center=cent[j]
            )
            lam_gap = lam1p - lam2p
            anis_ratio = lam1p / (lam2p + 1e-12) if lam2p > 0 else _np.inf

            u = None
            if blend_spin_cues:
                geo_weight = float(max(min(anis_ratio - 1.0, 4.0), 0.0))
                candidate_info = []
                if lam_gap > 1e-8 and geo_weight > 0.0:
                    candidate_info.append((x_e2, geo_weight))
                if n_xpp > 1e-6:
                    parent_weight = float(max(n_xpp, 1e-3))
                    candidate_info.append((x_parent_proj_unit, parent_weight))
                seam_weight = 0.0
                if prefer_parent_for_spin and sp_proj_norm > 1e-6:
                    seam_weight = seam_quality * sp_proj_norm
                    if anchors is None or (j, i) not in anchors:
                        seam_weight *= 0.5
                    seam_weight = float(min(max(seam_weight, 0.0), 20.0))
                    # if verbose:
                    #     print(f"~~~ part {j} seam weight = {seam_weight:.3f} (quality={seam_quality:.3f}, len={sp_proj_norm:.3f})")
                    if seam_weight > 0.0:
                        sp_unit = sp_proj / (sp_proj_norm + 1e-12)
                        candidate_info.append((sp_unit, seam_weight))
                if blend_include_neighbors and len(nbrs.get(j, [])) >= 2:
                    for k2 in nbrs[j]:
                        if k2 == i:
                            continue
                        neigh = _proj_plane(_seam_vec(j, k2), z_j)
                        neigh_norm = _np.linalg.norm(neigh)
                        if neigh_norm < 1e-6:
                            continue
                        wq = _seam_quality(j, k2)
                        if wq <= 0.0:
                            continue
                        candidate_info.append((neigh / (neigh_norm + 1e-12), float(wq)))
                if blend_spin_cues_signless:
                    S = _np.zeros((3, 3), dtype=_np.float64)
                    wsum = 0.0
                    for vec, weight in candidate_info:
                        if weight <= 0.0:
                            continue
                        S += float(weight) * _np.outer(vec, vec)
                        wsum += float(weight)
                    if wsum > 0.0:
                        vals_S, vecs_S = _np.linalg.eigh(S)
                        u = vecs_S[:, -1]
                if u is None and candidate_info:
                    acc = _np.zeros(3, dtype=_np.float64)
                    for vec, weight in candidate_info:
                        acc += float(weight) * vec
                    if _np.linalg.norm(acc) > 1e-8:
                        u = acc
            else:
                if prefer_parent_for_spin and sp_proj_norm > 1e-8:
                    u = sp_proj

            if u is None or _np.linalg.norm(u) < 1e-8:
                if lam_gap > 1e-8:
                    u = x_e2
                else:
                    if verbose:
                        print(f"~~~ [recipe_blended_frames] part {j} planar PCA axis also ambiguous (lam1={lam1p:.6f}, lam2={lam2p:.6f})")

            if u is None or _np.linalg.norm(u) < 1e-8:
                if verbose:
                    print(f"@@@ [recipe_blended_frames] part {j} fallback to best neighbor seam axis")
                best = None; best_w = -_np.inf
                for k2 in nbrs[j]:
                    if k2 == i: continue
                    wq = _seam_quality(j, k2)
                    cand = _proj_plane(_seam_vec(j, k2), z_j)
                    if _np.linalg.norm(cand) < 1e-8: continue
                    if wq > best_w: best_w = wq; best = cand
                if best is not None:
                    u = best

            if u is None or _np.linalg.norm(u) < 1e-8:
                if verbose:
                    print(f"!!! [recipe_blended_frames] part {j} falling back to parent's x axis projection")
                u = x_parent_proj

            x_j = _normalize(u)
            y_j = _normalize(_np.cross(z_j, x_j))
            x_j = _normalize(_np.cross(y_j, z_j))

            # Step 3) sign disambiguation w.r.t parent seam
            # Ensure +Z roughly points away from parent toward the rest of j
            if _np.dot(z_j, _seam_vec(j, i)) < 0.0:
                if verbose:
                    print(f"*** [recipe_blended_frames] part {j} flipping z to point away from parent {i}")
                z_j = -z_j; x_j = -x_j  # keep right-handed; y stays
                y_j = _normalize(_np.cross(z_j, x_j))
                x_j = _normalize(_np.cross(y_j, z_j))

            if neighbor_consensus_sign and len(nbrs.get(j, [])) >= 2:
                cons = _np.zeros(3, dtype=_np.float64)
                wsum = 0.0
                for k2 in nbrs[j]:
                    if k2 == i:
                        continue
                    wq = _seam_quality(j, k2)
                    if wq <= 0.0:
                        continue
                    c = _proj_plane(_seam_vec(j, k2), z_j)
                    cn = _np.linalg.norm(c)
                    if cn < 1e-6:
                        continue
                    cons += float(wq) * (c / (cn + 1e-12))
                    wsum += float(wq)
                if wsum > 0.0 and _np.linalg.norm(cons) > 1e-6:
                    c_hat = cons / (_np.linalg.norm(cons) + 1e-12)
                    if _np.dot(x_j, c_hat) < 0.0:
                        print(f"111 [recipe_blended_frames] part {j} flipping x to align with neighbor {k2}")
                        x_j = -x_j; y_j = -y_j

            # Optionally align x spin so that its projection is concordant with parent seam projection
            if prefer_parent_for_spin and anchors is not None and (j, i) in anchors:
                ok_len = sp_proj_norm > 0.1
                dot_x_sp = float(_np.dot(x_j, sp_proj))
                ok_dot = dot_x_sp < -0.1
                ok_quality = seam_quality >= spin_flip_quality_min
                ambiguous = anis_ratio < spin_ambiguous_ratio
                parent_agrees = (n_xpp > 0.2 and _np.dot(x_j, x_parent_proj) > 0)
                if verbose and j == "right_upper_leg":
                    print(f"$$$ part {j} seam_quality = {seam_quality:.3f}, dot(x, seam_proj)={dot_x_sp:.3f}, anis_ratio={anis_ratio:.3f}, n_xpp={n_xpp:.3f}, parent_agrees={parent_agrees}")
                if ok_len and ok_dot and ok_quality and ambiguous and not parent_agrees:
                    if verbose:
                        print(f"222 [recipe_blended_frames] part {j} flipping x (seam-gated)")
                    x_j = -x_j; y_j = -y_j

            # Alternative: always align with parent seam direction if available
            # sp_proj = _proj_plane(_seam_vec(j, i), z_j)
            # print(f"(((((Seam projected norm for part {j} toward parent {i}: {_np.linalg.norm(sp_proj):.6f})")
            # if _np.linalg.norm(sp_proj) > 1e-1 and _np.dot(x_j, sp_proj) < 0.0:
            #     if verbose:
            #         print(f"$$$ [recipe_blended_frames] part {j} flipping x to align spin with parent seam")
            #     x_j = -x_j; y_j = -y_j

            F[j] = _np.stack([x_j, y_j, z_j], axis=1)
            visited.add(j)
            q.append(j)

    return F


def graph_consistent_frames(
    parts,
    part_graph,
    root=None,
    anchors=None,
    up_hint=None,
    front_hint=None,
    z_mode="centroid",
    x_mode="distal_centroid",
):
    """
    Build per-part frames that are consistent across the adjacency graph by propagating
    orientation from a root and resolving twist using graph-aware cues.

    z_mode controls the outward axis choice.
    x_mode controls the in-plane ("front") axis. Supported options:
        "distal_centroid" -> direction to distal neighbor centroids.
        "distal_seam"     -> direction between seam centers (requires anchors).
        other values fall back to shape and parent continuity only.

    Returns: dict part_id -> 3x3 rotation (columns [x,y,z]).
    """
    keys = list(parts.keys())
    if root is None:
        root = keys[0]

    # Build adjacency
    nbrs = {k: [] for k in keys}
    for i, j, w in part_graph:
        if i in nbrs:
            nbrs[i].append(j)
        if j in nbrs:
            nbrs[j].append(i)

    # Precompute raw PCA frames and centroids
    cent = {
        k: parts[k].get("centroid", robust_centroid(parts[k]["points"])) for k in keys
    }
    rawF = {}
    for k in keys:
        if k == root:
            up_hint = up_hint
            front_hint = front_hint
        else:
            up_hint = None
            front_hint = None
        rawF[k] = pca_frame(parts[k]["points"], parts[k].get("normals", None), up_hint=up_hint, frontal_hint=front_hint)

    # Precompute seam normals (optional) from anchors if given
    seam_normal = {}
    if anchors is not None:
        for (i, j), Bij in anchors.items():
            n = _fit_plane_normal(Bij)
            v = cent[j] - cent[i]
            if np.dot(n, v) < 0:
                n = -n
            seam_normal[(i, j)] = n
            seam_normal[(j, i)] = -n

    from collections import deque

    F = {}
    visited = set([root])

    # Root frame: use rawF[root], optional up_hint already folded in
    F[root] = rawF[root]
    q = deque([root])

    while q:
        i = q.popleft()
        Xi = F[i][:, 0]  # parent's x

        for j in nbrs.get(i, []):
            if j in visited:
                continue
            # z_j: point from i to j (centroid direction)
            if z_mode == "centroid":
                v = cent[j] - cent[i]
                z_j = v / (np.linalg.norm(v) + 1e-12)
            elif (
                z_mode == "seam_to_distal_pca"
                and anchors is not None
                and (i, j) in anchors
            ):
                Bij = anchors[(i, j)]
                s_ij = _seam_center(Bij)
                Xj = parts[j]["points"]
                d = _nn_distances_to_set(Xj, Bij)
                q50, q90 = np.quantile(d, 0.5), np.quantile(d, 0.9)
                denom = max(q90 - q50, 1e-6)
                w = np.clip((d - q50) / denom, 0.0, 1.0) ** 2
                if w.sum() < 1e-6:
                    z_j = rawF[j][:, 2]
                else:
                    z_j = _weighted_pca_axis(Xj, w)
                    if np.dot(z_j, cent[j] - s_ij) < 0:
                        z_j = -z_j
            elif z_mode == "seam_centroid_to_distal_seam" and anchors is not None:
                distal = []
                for k2 in nbrs.get(j, []):
                    if k2 == i:
                        continue
                    if (j, k2) in anchors:
                        distal.append(_seam_center(anchors[(j, k2)]))
                if len(distal) > 0 and (i, j) in anchors:
                    s_ij = _seam_center(anchors[(i, j)])
                    s_dst = np.mean(np.stack(distal, axis=0), axis=0)
                    v = s_dst - s_ij
                    z_j = v / (np.linalg.norm(v) + 1e-12)
                else:
                    v = cent[j] - cent[i]
                    z_j = v / (np.linalg.norm(v) + 1e-12)
            elif z_mode == "seam_centroid_to_centroid" and anchors is not None:
                if (i, j) in anchors:
                    s_ij = _seam_center(anchors[(i, j)])
                    v = cent[j] - s_ij
                    z_j = v / (np.linalg.norm(v) + 1e-12)
            elif (
                z_mode == "seam_centroid_to_centroid_direction" and anchors is not None
            ):
                if (i, j) in anchors:
                    s_ij = _seam_center(anchors[(i, j)])
                    v = cent[j] - s_ij
                    if np.dot(v, rawF[j][:, 2]) < 0:
                        rawF[j][:, 2] = -rawF[j][:, 2]
                    z_j = rawF[j][:, 2]
            else:
                z_j = rawF[j][:, 2]

            # x_j: blend graph-aware, geometric, and continuity cues in the plane orthogonal to z_j
            Pperp = np.eye(3) - np.outer(z_j, z_j)
            x_parent_proj = Pperp @ Xi
            n_parent = np.linalg.norm(x_parent_proj)
            if n_parent > 1e-12:
                x_parent_proj /= n_parent

            x_candidates = []  # list of (vector, weight)

            # Distal neighbor cues (graph information)
            # if x_mode in ("distal_centroid", "distal_seam"):
            #     distal_dirs = []
            #     if x_mode == "distal_centroid":
            #         for k2 in nbrs.get(j, []):
            #             if k2 == i:
            #                 continue
            #             vdst = cent[k2] - cent[j]
            #             if np.linalg.norm(vdst) > 1e-12:
            #                 distal_dirs.append(vdst)
            #     elif x_mode == "distal_seam" and anchors is not None:
            #         s_parent = _seam_center(anchors[(i, j)]) if (i, j) in anchors else cent[i]
            #         distal_seams = []
            #         for k2 in nbrs.get(j, []):
            #             if k2 == i:
            #                 continue
            #             if (j, k2) in anchors:
            #                 distal_seams.append(_seam_center(anchors[(j, k2)]))
            #         if len(distal_seams) > 0:
            #             s_dst = np.mean(np.stack(distal_seams, axis=0), axis=0)
            #             vdst = s_dst - s_parent
            #             if np.linalg.norm(vdst) > 1e-12:
            #                 distal_dirs.append(vdst)
            #     if len(distal_dirs) > 0:
            #         vbar = np.mean(np.stack(distal_dirs, axis=0), axis=0)
            #         x_d = Pperp @ vbar
            #         if np.linalg.norm(x_d) > 1e-6:
            #             x_d /= np.linalg.norm(x_d)
            #             x_candidates.append((x_d, 1.0))

            # Geometric cue: dominant in-plane axis (optionally seam-weighted for leaves)
            is_leaf = len(nbrs.get(j, [])) <= 1
            w_plane = None
            if is_leaf and anchors is not None and (i, j) in anchors:
                Xj = parts[j]["points"]
                d = _nn_distances_to_set(Xj, anchors[(i, j)])
                q50, q90 = np.quantile(d, 0.5), np.quantile(d, 0.9)
                denom = max(q90 - q50, 1e-6)
                w_plane = np.clip((d - q50) / denom, 0.0, 1.0) ** 2
            x_geo, (lam1, lam2) = _dominant_axis_in_plane(
                parts[j]["points"], z_j, w=w_plane, center=cent[j]
            )
            anisotropy = lam1 / (lam2 + 1e-12)
            thr = 1.01 if is_leaf else 1.05
            if np.isfinite(anisotropy) and anisotropy > thr:
                # x_candidates.append((x_geo, 1.25 if is_leaf else 1.0))
                x_candidates.append((x_geo, 1.0 if is_leaf else 1.0))
            else:
                print(f"### Part {j} low anisotropy {anisotropy:.3f}, skipping geometric x cue.")
                x_candidates.append((x_parent_proj, 1.0))

            # # Continuity cue: parent's x projection
            # if n_parent > 1e-12:
            #     x_candidates.append((x_parent_proj, 0.5))

            # Blend candidates
            if len(x_candidates) == 0:
                if anchors is not None and (i, j) in seam_normal:
                    x_tmp = Pperp @ seam_normal[(i, j)]
                    if np.linalg.norm(x_tmp) > 1e-6:
                        x_j = x_tmp
                    else:
                        x_j = Pperp @ rawF[j][:, 0]
                else:
                    x_j = Pperp @ rawF[j][:, 0]
            else:
                acc = np.zeros(3, dtype=np.float64)
                for vec, weight in x_candidates:
                    acc += float(weight) * vec
                x_j = acc

            if np.linalg.norm(x_j) < 1e-8:
                x_j = Pperp @ rawF[j][:, 0]
            x_j /= np.linalg.norm(x_j) + 1e-12

            # if n_parent > 1e-12 and np.dot(x_j, x_parent_proj) < 0:
            #     x_j = -x_j

            y_j = np.cross(z_j, x_j)
            y_j /= np.linalg.norm(y_j) + 1e-12
            x_j = np.cross(y_j, z_j)
            x_j /= np.linalg.norm(x_j) + 1e-12

            Rj = np.stack([x_j, y_j, z_j], axis=1)
            u, _, vt = np.linalg.svd(Rj)
            F[j] = u @ vt

            visited.add(j)
            q.append(j)

    for k in keys:
        if k not in F:
            F[k] = rawF[k]
    return F


# -------------------------------
# Main algorithm
# -------------------------------

DEFAULTS = dict(
    outer_iters=15,
    keep_ratio_schedule=[0.7, 0.8, 0.85, 0.9],
    huber_delta=0.02,  # not used directly (kept for future robust weighting)
    smooth_lambda=20.0,
    boundary_gamma=80.0,
    prior_mu=10.0,
    lm_damp=1e-6,
    verbose=True,
    # NEW:
    use_up_hints=True,
    up_hint_strategy="farthest",  # or "lexi"
    init_mode="graph",  # or "graph"
    root_part_id="root",
    z_mode="centroid",
    x_mode="distal_centroid",
    initial_axis_flip_enabled=False,
    initial_axis_flip_axes=["z"],
    initial_axis_flip_include_z=False,
    initial_axis_flip_use_data_terms=False,
    initial_axis_flip_keep_ratio=1.0,
    initial_axis_flip_parts=None,
    initial_axis_flip_debug_plotly=False,
    double_sided_data_terms=False,
    use_sinkhorn_matches=False,
    sinkhorn_kwargs=None,
    compute_normals_on_the_fly=False,
)

def build_data_terms(
    source_parts,
    target_parts,
    states,
    keep_ratio,
    p2p_parts=None,
    flip_normals=None,
    *,
    double_sided: bool = False,
    use_sinkhorn_matches: bool = False,
    sinkhorn_kwargs: Optional[Dict[str, Any]] = None,
    compute_normals_on_the_fly: bool = False,
):
    """Build per-part point-to-plane correspondences and return a dict k -> list[(x,y,n,w)].
    p2p_parts: optional set/list of part ids to force point-to-point (ignores target normals).
    flip_normals: optional set/list of part ids to negate target normals (for debugging).
    double_sided: when True, also adds target->source correspondences for a symmetric Chamfer.
    use_sinkhorn_matches: toggle to compute correspondences via Sinkhorn transport (argmax per source).
    sinkhorn_kwargs: optional dict of parameters for sinkhorn_matches (e.g. eps, tau, n_iters).
    compute_normals_on_the_fly: when True, estimate normals for transformed source points per part.
    """
    if p2p_parts is None: p2p_parts = set()
    if flip_normals is None: flip_normals = set()
    data = {}
    for k in source_parts.keys():
        Xs = source_parts[k]["points"]
        Xt = target_parts[k]["points"]
        ns = source_parts[k].get("normals", None)
        nt_full = target_parts[k].get("normals", None)
        if ns is not None and ns.shape[0] != Xs.shape[0]:
            ns = None
        if nt_full is not None and nt_full.shape[0] != Xt.shape[0]:
            nt_full = None
        R = states[k]["R"]; t = states[k]["t"]; c = states[k]["center"]
        ns_world = ns @ R.T if ns is not None else None
        nt = None if (k in p2p_parts) else nt_full
        Xs_t = se3_apply(R, t, Xs, center=c)
        ns_estimated = None
        if compute_normals_on_the_fly and Xs_t.shape[0] >= 3:
            try:
                ns_estimated = compute_normals(Xs_t)
            except ImportError:
                warnings.warn(
                    "compute_normals_on_the_fly requires scikit-learn; falling back to default normals.",
                    RuntimeWarning,
                )
            except Exception as exc:
                warnings.warn(
                    f"Failed to compute normals on-the-fly for part '{k}': {exc}.",
                    RuntimeWarning,
                )
                ns_estimated = None
        entries: List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

        if use_sinkhorn_matches:
            mask, nn, _ = sinkhorn_matches(
                Xs_t,
                Xt,
                keep_ratio=keep_ratio,
                src_normals=None if (k in p2p_parts) else (ns_estimated if ns_estimated is not None else ns_world),
                tgt_normals=nt,
                sinkhorn_kwargs=sinkhorn_kwargs,
            )
            n = None
        else:
            mask, nn, _, n = trimmed_matches(Xs_t, Xt, nt, keep_ratio=keep_ratio)
        idx = np.where(mask)[0]
        if idx.size > 0:
            xs = Xs[idx]
            ys = Xt[nn[idx]]
            if n is None or use_sinkhorn_matches:
                candidate_normals = None
                if ns_estimated is not None:
                    candidate_normals = ns_estimated[idx]
                elif ns_world is not None:
                    candidate_normals = ns_world[idx]
                if candidate_normals is not None:
                    nrm = candidate_normals.copy()
                    norms = np.linalg.norm(nrm, axis=1, keepdims=True)
                    valid = norms[:, 0] > 1e-12
                    if np.any(valid):
                        nrm[valid] /= norms[valid]
                    if not np.all(valid):
                        v = ys - Xs_t[idx]
                        v_norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
                        nrm[~valid] = v[~valid] / v_norm[~valid]
                else:
                    # point-to-point fallback: align along connection vector
                    v = ys - Xs_t[idx]
                    nrm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
            else:
                nrm = n[idx]
            if k in flip_normals:
                nrm = -nrm
            entries.extend((xs[i], ys[i], nrm[i], 1.0) for i in range(idx.size))

        if double_sided:
            if use_sinkhorn_matches:
                mask_rev, nn_rev, _ = sinkhorn_matches(
                    Xt,
                    Xs_t,
                    keep_ratio=keep_ratio,
                    src_normals=None if (k in p2p_parts) else nt_full,
                    tgt_normals=None if (k in p2p_parts) else (ns_estimated if ns_estimated is not None else ns_world),
                    sinkhorn_kwargs=sinkhorn_kwargs,
                )
            else:
                mask_rev, nn_rev, _, _ = trimmed_matches(Xt, Xs_t, None, keep_ratio=keep_ratio)
            idx_rev = np.where(mask_rev)[0]
            if idx_rev.size > 0:
                xs_rev = Xs[nn_rev[idx_rev]]
                ys_rev = Xt[idx_rev]
                if use_sinkhorn_matches or nt is None:
                    candidate_normals_rev = None
                    if ns_estimated is not None:
                        candidate_normals_rev = ns_estimated[nn_rev[idx_rev]]
                    elif ns_world is not None:
                        candidate_normals_rev = ns_world[nn_rev[idx_rev]]
                    if candidate_normals_rev is not None:
                        nrm_rev = candidate_normals_rev.copy()
                        norms_rev = np.linalg.norm(nrm_rev, axis=1, keepdims=True)
                        valid_rev = norms_rev[:, 0] > 1e-12
                        if np.any(valid_rev):
                            nrm_rev[valid_rev] /= norms_rev[valid_rev]
                        if not np.all(valid_rev):
                            v_rev = ys_rev - Xs_t[nn_rev[idx_rev]]
                            v_rev_norm = np.linalg.norm(v_rev, axis=1, keepdims=True) + 1e-12
                            nrm_rev[~valid_rev] = v_rev[~valid_rev] / v_rev_norm[~valid_rev]
                    else:
                        v_rev = ys_rev - Xs_t[nn_rev[idx_rev]]
                        nrm_rev = v_rev / (np.linalg.norm(v_rev, axis=1, keepdims=True) + 1e-12)
                else:
                    nrm_rev = nt[idx_rev]
                if k in flip_normals:
                    nrm_rev = -nrm_rev
                entries.extend((xs_rev[i], ys_rev[i], nrm_rev[i], 1.0) for i in range(idx_rev.size))

        data[k] = entries
    return data

def build_smooth_terms(part_graph, states, smooth_lambda):
    terms = []
    # part_graph entries: (i, j, w)
    for (i, j, w) in part_graph:
        ci = states[i]["center"]; cj = states[j]["center"]
        terms.append((i, j, smooth_lambda * float(w), ci, cj))
    return terms

def build_boundary_terms(anchors, boundary_gamma):
    """anchors: dict (i,j) or (j,i) -> (M,3)"""
    terms = []
    if anchors is None: return terms
    for key, b in anchors.items():
        i, j = key
        terms.append((i, j, b, boundary_gamma))
    return terms

def build_prior_terms(states, prior_mu):
    return [(k, prior_mu, states[k]["R0"]) for k in states.keys()]

def apply_delta(states, parts_keys, delta):
    for idx, k in enumerate(parts_keys):
        d = delta[6*idx:6*idx+6]
        T = se3_exp(d)
        R = states[k]["R"]; t = states[k]["t"]; c = states[k]["center"]
        # right-multiplicative update to match Jacobians in assemble_system
        R_new = R @ T[:3,:3]
        t_new = t + T[:3,3]
        # re-orthonormalize to combat drift
        u, _, vt = np.linalg.svd(R_new)
        R_new = u @ vt
        states[k]["R"] = R_new
        states[k]["t"] = t_new
    return states

def total_cost(parts_keys, states, data_terms, smooth_terms, boundary_terms, prior_terms):
    cost = 0.0
    # data
    for k, entries in data_terms.items():
        R = states[k]["R"]; t = states[k]["t"]; c = states[k]["center"]
        for (x, y, n, w) in entries:
            r = n.dot(se3_apply(R,t,x[None,:],center=c)[0] - y)
            cost += w * (r*r)
    print(f"333 Data cost: {cost:.6f}")
    # smooth
    for (i, j, w_s, ci, cj) in smooth_terms:
        Ri = states[i]["R"]; ti = states[i]["t"]
        Rj = states[j]["R"]; tj = states[j]["t"]
        r = (Ri @ (cj-ci) + ti + ci) - (tj + cj)
        cost += w_s * float(r @ r)
    # boundary
    for (i, j, b_list, w_b) in boundary_terms:
        Ri = states[i]["R"]; ti = states[i]["t"]; ci = states[i]["center"]
        Rj = states[j]["R"]; tj = states[j]["t"]; cj = states[j]["center"]
        for b in b_list:
            r = (Ri @ (b - ci) + ci + ti) - (Rj @ (b - cj) + cj + tj)
            cost += w_b * float(r @ r)
    # prior
    for (k, w_p, R0) in prior_terms:
        r = so3_log(states[k]["R"] @ R0.T)
        cost += w_p * float(r @ r)
    return cost


def _search_initial_axis_flips(
    states,
    parts_keys,
    part_graph,
    source_parts,
    target_parts,
    source_anchors,
    cfg,
    keep_ratio=None,
    axes=None,
    parts=None,
    include_data_terms=False,
    p2p_parts=None,
    flip_normals=None,
    verbose=False,
):
    axes = list(axes or [])
    if cfg.get("initial_axis_flip_include_z", False) and "z" not in axes:
        axes.append("z")
    flip_mats = _enumerate_flip_mats(axes)
    if len(flip_mats) <= 1:
        return _clone_states(states), dict(improvement=0.0, original_cost=0.0, final_cost=0.0, changed_parts=[])

    debug_plotly = bool(cfg.get("initial_axis_flip_debug_plotly", False))
    go = None
    debug_plot_dir = None
    if debug_plotly:
        try:
            import plotly.graph_objects as go  # type: ignore
        except Exception as exc:
            warnings.warn(
                f"Plotly not available for initial axis flip debug visualization: {exc}",
                RuntimeWarning,
            )
            debug_plotly = False
        else:
            log_dir = cfg.get("log_dir", ".")
            debug_plot_dir = os.path.join(log_dir, "debug_init_flip")
            os.makedirs(debug_plot_dir, exist_ok=True)

    boundary_terms = build_boundary_terms(source_anchors, cfg["boundary_gamma"])
    use_boundary = len(boundary_terms) > 0
    if not include_data_terms and not use_boundary:
        return _clone_states(states), dict(improvement=0.0, original_cost=0.0, final_cost=0.0, changed_parts=[])

    keep_schedule = cfg.get("keep_ratio_schedule", [])
    if keep_ratio is None:
        keep_ratio_eval = keep_schedule[0] if len(keep_schedule) > 0 else 0.9
    else:
        keep_ratio_eval = float(keep_ratio)

    p2p_parts = set(p2p_parts or [])
    flip_normals = set(flip_normals or [])

    smooth_terms = []
    prior_terms = []

    def evaluate(candidate_states):
        data_terms = build_data_terms(
            source_parts,
            target_parts,
            candidate_states,
            keep_ratio_eval,
            p2p_parts=p2p_parts,
            flip_normals=flip_normals,
            double_sided=cfg.get("double_sided_data_terms", False),
            use_sinkhorn_matches=cfg.get("use_sinkhorn_matches_initial", False),
            sinkhorn_kwargs=cfg.get("sinkhorn_kwargs", None),
            # compute_normals_on_the_fly=cfg.get("compute_normals_on_the_fly", False),
            compute_normals_on_the_fly=False,
        ) if include_data_terms else {}
        return total_cost(parts_keys, candidate_states, data_terms, smooth_terms, boundary_terms, prior_terms)

    def _debug_visualize_candidate(part_id, cand_idx, mat, best_state, cand_state, best_cost_val, cand_cost_val):
        if not debug_plotly or go is None or debug_plot_dir is None:
            return
        part_data = source_parts.get(part_id)
        if part_data is None or "points" not in part_data:
            return
        Xs = np.asarray(part_data["points"])
        if Xs.size == 0:
            return
        target_data = target_parts.get(part_id)
        Xt = None
        if target_data is not None and "points" in target_data:
            Xt = np.asarray(target_data["points"])

        center_best = best_state.get("center", None)
        center_cand = cand_state.get("center", center_best)
        pts_best = se3_apply(best_state["R"], best_state["t"], Xs, center=center_best)
        pts_cand = se3_apply(cand_state["R"], cand_state["t"], Xs, center=center_cand)

        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=pts_best[:, 0],
                y=pts_best[:, 1],
                z=pts_best[:, 2],
                mode="markers",
                marker=dict(size=2.5, color="#1f77b4"),
                name=f"best (cost={best_cost_val:.3e})",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=pts_cand[:, 0],
                y=pts_cand[:, 1],
                z=pts_cand[:, 2],
                mode="markers",
                marker=dict(size=2.5, color="#d62728"),
                name=f"candidate {cand_idx} (cost={cand_cost_val:.3e})",
            )
        )
        if Xt is not None and Xt.size > 0:
            fig.add_trace(
                go.Scatter3d(
                    x=Xt[:, 0],
                    y=Xt[:, 1],
                    z=Xt[:, 2],
                    mode="markers",
                    marker=dict(size=2.0, color="#7f7f7f", opacity=0.5),
                    name="target",
                )
            )

        fig.update_layout(
            title=f"Part {part_id} candidate {cand_idx} axis flip",
            scene=dict(aspectmode="data"),
            legend=dict(itemsizing="constant"),
        )
        fig.add_annotation(
            text="\n".join(
                [
                    "Candidate flip matrix:",
                    str(np.array_str(mat, precision=3)),
                ]
            ),
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.02,
            showarrow=False,
            align="left",
        )

        part_dir = os.path.join(debug_plot_dir, str(part_id))
        os.makedirs(part_dir, exist_ok=True)
        out_path = os.path.join(part_dir, f"candidate_{cand_idx:02d}.html")
        fig.write_html(out_path)

    working_states = _clone_states(states)
    base_cost = evaluate(working_states)

    parts_filter = None if parts is None else {p for p in parts if p in working_states}

    # Build adjacency map and traverse in BFS order across the part graph, mirroring recipe_blended_frames
    neighbors = {k: [] for k in parts_keys}
    if part_graph is not None:
        for (i, j, *_) in part_graph:
            if i in neighbors and j in neighbors:
                neighbors[i].append(j)
                neighbors[j].append(i)

    from collections import deque

    traversal = []
    visited = set()

    def enqueue_component(start):
        if start in visited or start not in neighbors:
            return
        queue = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            if parts_filter is None or node in parts_filter:
                traversal.append(node)
            for nbr in neighbors.get(node, []):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)

    root = cfg.get("root_part_id", None)
    if root is not None:
        enqueue_component(root)
    for k in parts_keys:
        enqueue_component(k)

    # Ensure isolated or filtered parts are still considered
    if parts_filter is not None:
        for p in parts_filter:
            if p not in traversal and p in working_states:
                traversal.append(p)
    else:
        for k in parts_keys:
            if k not in traversal and k in working_states:
                traversal.append(k)

    if not traversal:
        return working_states, dict(improvement=0.0, original_cost=base_cost, final_cost=base_cost, changed_parts=[])

    best_cost = base_cost
    changed_parts = []

    for part in traversal:
        # print_details = (part == "left_upper_leg") and verbose
        print_details = verbose
        part_best_cost = best_cost
        part_best_states = working_states
        part_changed = False
        for cand_idx, mat in enumerate(flip_mats[1:], start=1):
            candidate_states = _clone_states(working_states)
            R_current = candidate_states[part]["R"]
            R_candidate = R_current @ mat.T
            candidate_states[part]["R"] = R_candidate
            candidate_states[part]["R0"] = R_candidate.copy()
            best_state_before = part_best_states[part]
            part_best_cost_before = part_best_cost
            candidate_cost = evaluate(candidate_states)
            _debug_visualize_candidate(
                part,
                cand_idx,
                mat,
                best_state_before,
                candidate_states[part],
                part_best_cost_before,
                candidate_cost,
            )
            if print_details:
                print(f"@@@ [init axis flip] part {part} try flip mat:\n{mat}\n  cost = {candidate_cost:.6e} (best {part_best_cost:.6e})")
            if candidate_cost + 1e-9 < part_best_cost:
                part_best_cost = candidate_cost
                part_best_states = candidate_states
                part_changed = True
        if part_changed:
            working_states = part_best_states
            best_cost = part_best_cost
            changed_parts.append(part)

    if verbose and changed_parts and base_cost > 0:
        print(
            "[init axis flip] parts={} reduced seam cost {:.6e} -> {:.6e}".format(
                changed_parts, base_cost, best_cost
            )
        )

    return working_states, dict(
        improvement=max(base_cost - best_cost, 0.0),
        original_cost=base_cost,
        final_cost=best_cost,
        changed_parts=changed_parts,
    )

def estimate_part_rigid_transforms(source_parts, target_parts, part_graph, source_anchors=None, target_anchors=None, params=None, log_dir="."):
    """Main entry: returns dict part_id -> {'R','t'}"""
    if params is None: params = {}
    cfg = DEFAULTS.copy(); cfg.update(params or {})
    cfg.setdefault("log_dir", log_dir)
    parts_keys = list(source_parts.keys())

    # init
    init_mode = cfg.get("init_mode", "graph")
    root = cfg.get("root_part_id", None)
    up_hint = cfg.get("up_hint", None)
    front_hint = cfg.get("front_hint", None)
    z_mode = cfg.get("z_mode", "centroid")
    x_mode = cfg.get("x_mode", "distal_centroid")

    if init_mode == "graph":
        # Fsrc = graph_consistent_frames(
        #     source_parts,
        #     part_graph,
        #     root=root,
        #     anchors=source_anchors,
        #     up_hint=up_hint,
        #     front_hint=front_hint,
        #     z_mode=z_mode,
        #     x_mode=x_mode,
        # )
        # Ftgt = graph_consistent_frames(
        #     target_parts,
        #     part_graph,
        #     root=root,
        #     anchors=target_anchors,
        #     up_hint=up_hint,
        #     front_hint=front_hint,
        #     z_mode=z_mode,
        #     x_mode=x_mode,
        # )
        Fsrc = recipe_blended_frames(
            source_parts, 
            part_graph, 
            root=root,
            up_hint=up_hint,
            frontal_hint=front_hint, 
            anchors=source_anchors,
            params={
                "verbose": True,
                "tau_aniso": 1.25,
                # "prefer_parent_for_spin": False,
                "blend_spin_cues": True,
                "blend_include_neighbors": False,
                "neighbor_consensus_sign": False,
            }
        )
        Ftgt = recipe_blended_frames(
            target_parts,
            part_graph,
            root=root,
            up_hint=up_hint,
            frontal_hint=front_hint,
            anchors=target_anchors,
            params={
                "verbose": True,
                "tau_aniso": 1.25,
                # "prefer_parent_for_spin": False,
                "blend_spin_cues": True,
                "blend_include_neighbors": False,
                "neighbor_consensus_sign": False,
            }
        )
        for k in source_parts.keys():
            source_parts[k]["frame"] = Fsrc[k]
            target_parts[k]["frame"] = Ftgt[k]
        states = init_transforms_from_frames(source_parts, target_parts)
    else:
        states = init_transforms_from_frames(source_parts, target_parts)
    # # init (now uses graph-based up hints if enabled)
    # states = init_transforms_from_frames(
    #     source_parts, target_parts,
    #     part_graph=part_graph,
    #     up_hint_strategy=cfg.get("up_hint_strategy", "farthest"),
    #     use_up_hints=cfg.get("use_up_hints", True)
    # )

    # Debug/diagnostic part options
    debug_parts = set(params.get("debug_parts", []))
    p2p_parts = set(params.get("p2p_parts", []))  # force p2point for selected parts
    flip_normals = set(params.get("flip_normals", []))  # optional normal flip for debug

    if cfg.get("initial_axis_flip_enabled", False):
        print("=== Initial axis flip search ===")
        axes_to_try = list(cfg.get("initial_axis_flip_axes", ["z"]))
        initial_flip_parts = cfg.get("initial_axis_flip_parts", None)
        keep_ratio_eval = cfg.get("initial_axis_flip_keep_ratio", None)
        states_candidate, _ = _search_initial_axis_flips(
            states,
            parts_keys,
            part_graph,
            source_parts,
            target_parts,
            source_anchors,
            cfg,
            keep_ratio=keep_ratio_eval,
            axes=axes_to_try,
            parts=initial_flip_parts,
            include_data_terms=cfg.get("initial_axis_flip_use_data_terms", False),
            p2p_parts=p2p_parts,
            flip_normals=flip_normals,
            verbose=cfg.get("verbose", False),
        )
        states = states_candidate
        print("=== End initial axis flip search ===")

    # NEW: visualize initial frames/centroids before optimization
    visualize_init_frames_plotly(
        source_parts,
        target_parts,
        log_dir,
        part_order=parts_keys,
        # anchors=target_anchors,
    )

    keep_schedule = cfg["keep_ratio_schedule"]
    outer_iters = cfg["outer_iters"]
    if len(keep_schedule) < outer_iters:
        keep_schedule = keep_schedule + [keep_schedule[-1]]*(outer_iters - len(keep_schedule))

    # Export initial mesh (before iterations)
    if cfg.get("export_meshes", False):
        log_dir = cfg.get("log_dir", ".")
        source_faces = cfg.get("source_faces", None)
        part_indices = cfg.get("part_indices", None)

        if source_faces is not None and part_indices is not None:
            initial_transforms = {k: {"R": states[k]["R"].copy(), "t": states[k]["t"].copy()} for k in parts_keys}
            initial_points_by_part = transform_points_by_part(
                {k: v["points"] for k, v in source_parts.items()},
                initial_transforms,
                centers={k: v.get("centroid", v["points"].mean(axis=0)) for k, v in source_parts.items()},
            )
            initial_vertices = compose_mesh_parts(
                initial_points_by_part, part_indices, parts_keys
            )
            initial_mesh = trimesh.Trimesh(vertices=initial_vertices, faces=source_faces, process=False)
            initial_mesh.export(os.path.join(log_dir, "iter_00_initial.obj"))

    def _debug_part_report(it, k, states_k):
        # lightweight stats on matches for one part
        Xs = source_parts[k]["points"]
        Xt = target_parts[k]["points"]
        ns = source_parts[k].get("normals", None)
        nt_full = target_parts[k].get("normals", None)
        if ns is not None and ns.shape[0] != Xs.shape[0]:
            ns = None
        if nt_full is not None and nt_full.shape[0] != Xt.shape[0]:
            nt_full = None
        R = states_k["R"]
        t = states_k["t"]
        c = states_k["center"]
        ns_world = ns @ R.T if ns is not None else None
        nt = None if (k in p2p_parts) else nt_full
        Xs_t = se3_apply(R, t, Xs, center=c)
        ns_estimated = None
        if cfg.get("compute_normals_on_the_fly", False) and Xs_t.shape[0] >= 3:
            try:
                ns_estimated = compute_normals(Xs_t)
            except ImportError:
                warnings.warn(
                    "compute_normals_on_the_fly requires scikit-learn; falling back to default normals.",
                    RuntimeWarning,
                )
            except Exception as exc:
                warnings.warn(
                    f"Failed to compute normals on-the-fly for part '{k}' during debug: {exc}.",
                    RuntimeWarning,
                )
                ns_estimated = None

        if cfg.get("use_sinkhorn_matches", False):
            mask, nn, d = sinkhorn_matches(
                Xs_t,
                Xt,
                keep_ratio=keep_ratio,
                src_normals=None if (k in p2p_parts) else (ns_estimated if ns_estimated is not None else ns_world),
                tgt_normals=nt,
                sinkhorn_kwargs=cfg.get("sinkhorn_kwargs", None),
            )
            n = None
        else:
            mask, nn, d, n = trimmed_matches(Xs_t, Xt, nt, keep_ratio=keep_ratio)
        kept = int(mask.sum())
        total = int(len(mask))
        if kept == 0:
            print(f"[it {it:02d}] [{k}] kept=0/{total}")
            return None
        d_kept = d[mask]
        msgs = f"[it {it:02d}] [{k}] kept={kept}/{total}  d(med/mean/95p)={np.median(d_kept):.4f}/{d_kept.mean():.4f}/{np.quantile(d_kept,0.95):.4f}"
        # directional agreement if normals available or synthetic
        if n is None:
            candidate_normals = None
            if ns_estimated is not None:
                candidate_normals = ns_estimated[mask]
            elif ns_world is not None:
                candidate_normals = ns_world[mask]
            if candidate_normals is not None:
                nrm = candidate_normals.copy()
                norms = np.linalg.norm(nrm, axis=1, keepdims=True)
                valid = norms[:, 0] > 1e-12
                if np.any(valid):
                    nrm[valid] /= norms[valid]
                if not np.all(valid):
                    v = Xt[nn[mask]] - Xs_t[mask]
                    v_norm = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
                    nrm[~valid] = v[~valid] / v_norm[~valid]
            else:
                v = Xt[nn[mask]] - Xs_t[mask]
                nrm = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
        else:
            nrm = n[mask]
        cos = np.sum(
            nrm
            * (Xt[nn[mask]] - Xs_t[mask])
            / (
                np.linalg.norm(Xt[nn[mask]] - Xs_t[mask], axis=1, keepdims=True) + 1e-12
            ),
            axis=1,
        )
        msgs += f"  cos(dir) mean={cos.mean():.3f}"
        print(msgs)
        return (Xs_t, Xt, nn, mask)

    def _debug_plot_matches(k, Xs_t, Xt, nn, mask, out_dir, it):
        try:
            import plotly.graph_objects as go
        except Exception:
            return
        sel = np.where(mask)[0]
        xs = Xs_t[sel]
        ys = Xt[nn[sel]]
        # build line segments
        Xl = []
        Yl = []
        Zl = []
        for a, b in zip(xs, ys):
            Xl.extend([a[0], b[0], None])
            Yl.extend([a[1], b[1], None])
            Zl.extend([a[2], b[2], None])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter3d(
                x=Xs_t[:, 0],
                y=Xs_t[:, 1],
                z=Xs_t[:, 2],
                mode="markers",
                marker=dict(size=2, color="blue"),
                name=f"{k} src_t",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=Xt[:, 0],
                y=Xt[:, 1],
                z=Xt[:, 2],
                mode="markers",
                marker=dict(size=2, color="red"),
                name=f"{k} tgt",
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=Xl,
                y=Yl,
                z=Zl,
                mode="lines",
                line=dict(color="gray", width=2),
                name="matches",
            )
        )
        fig.update_layout(scene=dict(aspectmode="data"))
        os.makedirs(out_dir, exist_ok=True)
        fig.write_html(
            os.path.join(out_dir, f"matches_{k}_it{it:02d}.html"),
            include_plotlyjs="cdn",
        )

    prev_cost = None
    for it in range(outer_iters):
        keep_ratio = keep_schedule[it]

        # Build terms
        # data_terms = build_data_terms(source_parts, target_parts, states, keep_ratio)
        data_terms = build_data_terms(
            source_parts,
            target_parts,
            states,
            keep_ratio,
            p2p_parts=p2p_parts,
            flip_normals=flip_normals,
            double_sided=cfg.get("double_sided_data_terms", False),
            use_sinkhorn_matches=cfg.get("use_sinkhorn_matches", False),
            sinkhorn_kwargs=cfg.get("sinkhorn_kwargs", None),
            compute_normals_on_the_fly=cfg.get("compute_normals_on_the_fly", False),
        )
        # Optional per-part diagnostics
        for k in debug_parts:
            dbg = _debug_part_report(it, k, states[k])
            if dbg is not None and params.get("debug_plot", False):
                Xs_t, Xt, nn, mask = dbg
                _debug_plot_matches(
                    k, Xs_t, Xt, nn, mask, os.path.join(log_dir, "debug_matches"), it
                )

        smooth_terms = build_smooth_terms(part_graph, states, cfg["smooth_lambda"])
        boundary_terms = build_boundary_terms(source_anchors, cfg["boundary_gamma"])
        prior_terms = build_prior_terms(states, cfg["prior_mu"])

        # Assemble & solve
        H, g = assemble_system(parts_keys, states, data_terms, smooth_terms, boundary_terms, prior_terms, lm_damp=cfg["lm_damp"])
        delta = solve_system(H, g)

        # Update
        states = apply_delta(states, parts_keys, delta)

        # Export mesh after iteration
        if cfg.get("export_meshes", False):
            log_dir = cfg.get("log_dir", ".")
            source_faces = cfg.get("source_faces", None)
            part_indices = cfg.get("part_indices", None)

            if source_faces is not None and part_indices is not None:
                post_iter_transforms = {k: {"R": states[k]["R"].copy(), "t": states[k]["t"].copy()} for k in parts_keys}
                post_iter_points_by_part = transform_points_by_part(
                    {k: v["points"] for k, v in source_parts.items()},
                    post_iter_transforms,
                    centers={k: v.get("centroid", v["points"].mean(axis=0)) for k, v in source_parts.items()},
                )
                post_iter_vertices = compose_mesh_parts(
                    post_iter_points_by_part, part_indices, parts_keys
                )
                post_iter_mesh = trimesh.Trimesh(
                    vertices=post_iter_vertices, faces=source_faces, process=False
                )
                post_iter_mesh.export(os.path.join(log_dir, f"iter_{it+1:02d}.obj"))

        # Monitor
        cost = total_cost(parts_keys, states, data_terms, smooth_terms, boundary_terms, prior_terms)
        if cfg["verbose"]:
            step_norm = np.linalg.norm(delta)
            print(f"[Iter {it+1:02d}] keep={keep_ratio:.2f}  cost={cost:.6e}  |delta|={step_norm:.3e}")
        if prev_cost is not None and abs(prev_cost - cost) < 1e-7:
            break
        prev_cost = cost

    # Pack outputs
    out = {k: {"R": states[k]["R"].copy(), "t": states[k]["t"].copy()} for k in parts_keys}
    return out


# -------------------------------
# Seam-anchor generation (for good segmentations)
# -------------------------------


def _rng_choice(rng, n, size, replace=False):
    """Helper: draw indices using either a provided RNG or global numpy."""
    if rng is None:
        return np.random.choice(n, size, replace=replace)
    if hasattr(rng, "choice"):
        return rng.choice(n, size, replace=replace)
    # Fallback for legacy RandomState which also exposes choice
    return np.random.choice(n, size, replace=replace)


def _rng_randint(rng, high):
    """Helper: draw single integer in [0, high)."""
    if rng is None:
        return int(np.random.randint(high))
    if hasattr(rng, "integers"):
        return int(rng.integers(high))
    if hasattr(rng, "randint"):
        return int(rng.randint(high))
    return int(np.random.randint(high))


def _median_nn_spacing(X, max_samples=4096, rng=None):
    """Median distance to the 2nd nearest neighbor (proxy for point spacing)."""
    Xs = X
    if X.shape[0] > max_samples:
        idx = _rng_choice(rng, X.shape[0], max_samples, replace=False)
        Xs = X[idx]
    try:
        from sklearn.neighbors import KDTree

        tree = KDTree(Xs)
        dists, _ = tree.query(Xs, k=2, return_distance=True)
        return float(np.median(dists[:, 1]))
    except Exception:
        D2 = ((Xs[:, None, :] - Xs[None, :, :]) ** 2).sum(axis=2)
        D2 += np.eye(Xs.shape[0]) * 1e9  # mask self
        return float(np.median(np.sqrt(D2.min(axis=1))))


def _nn_distances_all(X, Y):
    """Return per-point nearest-neighbor distance from each x in X to set Y."""
    try:
        from sklearn.neighbors import KDTree

        tree = KDTree(Y)
        d, _ = tree.query(X, k=1, return_distance=True)
        return d.reshape(-1)
    except Exception:
        dif = X[:, None, :] - Y[None, :, :]
        d2 = (dif * dif).sum(axis=2)
        return np.sqrt(d2.min(axis=1))


def _farthest_point_sampling(P, K, rng=None):
    """Simple FPS on point set P (N,3)."""
    N = P.shape[0]
    if N == 0:
        return P
    if N <= K:
        return P.copy()
    sel = [_rng_randint(rng, N)]
    d = np.full(N, np.inf, dtype=float)
    for _ in range(1, K):
        last = P[sel[-1]]
        d = np.minimum(d, np.linalg.norm(P - last, axis=1))
        sel.append(int(np.argmax(d)))
    return P[sel]


def compute_seam_anchors_from_parts(
    parts, part_graph, k_per_edge=64, radius_scale=2.0, symmetric=True, rng=None
):
    """
    Build seam anchors for each adjacent pair (i,j) in part_graph from *source* parts.
    Assumes segmentation is good (labels are reliable).

    Strategy:
      - Estimate point-spacing per part (median NN distance).
      - For each edge (i,j):
          * Compute distances from points of i to set j, and from j to set i.
          * Select points within tau = radius_scale * min(spacing_i, spacing_j) on each side.
          * Union those points, then sample up to k_per_edge with FPS for even coverage.
      - If `symmetric` is True, produce both keys (i,j) and (j,i) in the dict.

    Args:
      rng: optional numpy random number generator for deterministic sampling.

    Returns:
      anchors: dict[(i,j)] -> (K,3) numpy array of anchor points near the seam.
    """
    spacing = {k: _median_nn_spacing(v["points"], rng=rng) for k, v in parts.items()}
    anchors = {}
    for i, j, w in part_graph:
        if i not in parts or j not in parts:
            continue
        Xi = parts[i]["points"]
        Xj = parts[j]["points"]
        tau = radius_scale * float(min(spacing[i], spacing[j]))

        di = _nn_distances_all(Xi, Xj)
        dj = _nn_distances_all(Xj, Xi)

        seam_i = Xi[di <= tau]
        seam_j = Xj[dj <= tau]

        if seam_i.shape[0] + seam_j.shape[0] == 0:
            if di.size > 0:
                qi = np.quantile(di, 0.05)
                seam_i = Xi[di <= qi]
            if dj.size > 0:
                qj = np.quantile(dj, 0.05)
                seam_j = Xj[dj <= qj]

        if seam_i.size and seam_j.size:
            seam_union = np.concatenate([seam_i, seam_j], axis=0)
        else:
            seam_union = seam_i if seam_i.size else seam_j
        seam_union = _farthest_point_sampling(seam_union, k_per_edge, rng=rng)

        anchors[(i, j)] = seam_union
        if symmetric:
            anchors[(j, i)] = seam_union.copy()
    return anchors

# -------------------------------
# Convenience utilities
# -------------------------------

def transform_points_by_part(points_by_part, transforms, centers=None):
    """Apply estimated transforms to a dict of points per part. centers: dict or None."""
    out = {}
    for k, arr in points_by_part.items():
        c = centers[k] if centers is not None else None
        out[k] = se3_apply(transforms[k]["R"], transforms[k]["t"], arr, center=c)
    return out

def compose_mesh_parts(vertices_by_part, part_indices, order):
    """Compose separate part meshes back to a single mesh by placing transformed 
    vertices back into their original positions.
    
    Args:
        vertices_by_part: dict mapping part names to transformed vertices
        part_indices: dict mapping part names to original vertex indices
        order: list of part names (same as vertices_by_part.keys())
    
    Returns:
        vertices: (N, 3) array with vertices in original order
    """
    # Get the total number of vertices from the maximum index
    max_idx = max(np.max(part_indices[k]) for k in order) + 1
    vertices = np.zeros((max_idx, 3), dtype=np.float64)
    
    # Place each part's transformed vertices back into original positions
    for k in order:
        indices = part_indices[k]
        vertices[indices] = vertices_by_part[k]
    
    return vertices


def compute_normals(points, k=10, robust=True):
    """
    Compute surface normals for a set of 3D points using local PCA.
    
    Args:
        points: (N, 3) numpy array of 3D points
        k: number of nearest neighbors to use for normal estimation
        robust: if True, use robust estimation methods
    
    Returns:
        normals: (N, 3) numpy array of unit normal vectors
    """
    from sklearn.neighbors import NearestNeighbors
    
    N = points.shape[0]
    normals = np.zeros((N, 3), dtype=np.float64)
    
    if N < 3:
        # Not enough points for meaningful normals
        return np.tile([0, 0, 1], (N, 1))
    
    # Adjust k for small point sets
    k = min(k, N - 1, 20)  # Cap at 20 for efficiency
    
    # Build KD-tree for nearest neighbor search
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points)
    
    for i in range(N):
        # Find k nearest neighbors (including self)
        distances, indices = nbrs.kneighbors(points[i:i+1])
        neighbor_points = points[indices[0]]  # (k+1, 3)
        
        if robust and k > 6:
            # Remove outliers using distance-based filtering
            dists = distances[0]
            median_dist = np.median(dists[1:])  # exclude self (distance=0)
            threshold = median_dist * 2.5
            valid_mask = dists <= threshold
            neighbor_points = neighbor_points[valid_mask]
        
        if neighbor_points.shape[0] < 3:
            # Fall back to global up vector if not enough neighbors
            normals[i] = np.array([0, 0, 1])
            continue
            
        # Center the neighborhood
        centroid = neighbor_points.mean(axis=0)
        centered = neighbor_points - centroid
        
        # Compute covariance matrix
        cov = centered.T @ centered
        
        # Add small regularization for numerical stability
        cov += 1e-8 * np.eye(3)
        
        # Compute eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        
        # Normal is the eigenvector with smallest eigenvalue
        normal = eigenvecs[:, 0]
        
        # Normalize
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            normal = normal / norm
        else:
            normal = np.array([0, 0, 1])  # fallback
            
        normals[i] = normal
    
    # Optional: orient normals consistently using a simple heuristic
    # Point normals roughly toward the centroid of the point cloud
    if N > 10:
        global_centroid = points.mean(axis=0)
        for i in range(N):
            toward_center = global_centroid - points[i]
            if np.dot(normals[i], toward_center) < 0:
                normals[i] = -normals[i]
    
    return normals


def visualize_init_frames_plotly(source_parts, target_parts, log_dir, part_order=None, axis_scale=0.06,
                                 anchors=None):
    """
    Plot for each part:
      - Source and target point clouds (light gray, semi-transparent)
      - Source/target centroids
      - Local frames Fs (source, solid) and Ft (target, dashed)
      - Vector from source centroid to target centroid
      - (Optional) Seam anchor points per edge if anchors provided
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        print("Plotly is not installed. Install with: pip install plotly")
        return

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Order
    if part_order is None:
        part_order = list(source_parts.keys())

    # Gather all points for scale
    all_pts = []
    for k in part_order:
        if "points" in source_parts[k]:
            all_pts.append(source_parts[k]["points"])
        if "points" in target_parts[k]:
            all_pts.append(target_parts[k]["points"])
    if len(all_pts) == 0:
        print("No points to visualize.")
        return
    all_pts = np.concatenate(all_pts, axis=0)
    bb_min = all_pts.min(axis=0)
    bb_max = all_pts.max(axis=0)
    diag = np.linalg.norm(bb_max - bb_min)
    axis_len = max(1e-8, axis_scale * diag)

    # Part base palette (used for centroids and the cs->ct vector)
    base_palette = [
        "#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
        "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf",
        "#a55194","#393b79","#637939","#8c6d31","#843c39"
    ]

    # Consistent axis colors: X=red, Y=green, Z=blue
    axis_colors = ["#e41a1c", "#4daf4a", "#377eb8"]

    def hex_to_rgb(h):
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0,2,4))

    def lighten_rgb(rgb, alpha=0.5):
        # blend toward white by alpha
        r,g,b = rgb
        r2 = int(round(r + alpha*(255 - r)))
        g2 = int(round(g + alpha*(255 - g)))
        b2 = int(round(b + alpha*(255 - b)))
        return (r2,g2,b2)

    def rgb_str(rgb):
        return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

    def get_centroid(d, default_pts_key="points"):
        if "centroid" in d:
            return np.asarray(d["centroid"])
        return robust_centroid(np.asarray(d[default_pts_key]))

    def get_frame(d, pts_key="points", part_name=None):
        # use provided frame if present; else compute via PCA (use normals if available)
        F = d.get("frame", None)
        if F is not None:
            # print(f"Part {part_name} Using provided frame for part.")
            return np.asarray(F)
        normals = d.get("normals", d.get("normal", None))
        return pca_frame(np.asarray(d[pts_key]), normals, part_name=part_name)

    def axes_traces(center, F, width=5, dash=None, opacity=1.0):
        # Draw 3 axes using the global axis_colors and consistent mapping
        traces = []
        for i in range(3):
            end = center + axis_len * F[:, i]
            traces.append(go.Scatter3d(
                x=[center[0], end[0]],
                y=[center[1], end[1]],
                z=[center[2], end[2]],
                mode="lines",
                line=dict(color=axis_colors[i], width=width, dash=dash),
                opacity=opacity,
                hoverinfo="none",
                showlegend=False
            ))
        return traces

    fig_traces = []

    # Legend stubs for axis colors
    fig_traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                                   line=dict(color=axis_colors[0], width=6), name="X axis"))
    fig_traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                                   line=dict(color=axis_colors[1], width=6), name="Y axis"))
    fig_traces.append(go.Scatter3d(x=[None], y=[None], z=[None], mode="lines",
                                   line=dict(color=axis_colors[2], width=6), name="Z axis"))

    # Centroid markers grouped for cleaner legend
    src_centroid_x = []; src_centroid_y = []; src_centroid_z = []; src_centroid_c = []; src_centroid_text = []
    tgt_centroid_x = []; tgt_centroid_y = []; tgt_centroid_z = []; tgt_centroid_c = []; tgt_centroid_text = []

    # Add per-part elements
    for idx, k in enumerate(part_order):
        base_hex = base_palette[idx % len(base_palette)]
        base_rgb = hex_to_rgb(base_hex)
        src_color = rgb_str(base_rgb)
        tgt_color = rgb_str(lighten_rgb(base_rgb, alpha=0.45))
        vec_color = rgb_str(lighten_rgb(base_rgb, alpha=0.2))

        # Source
        Xs = np.asarray(source_parts[k]["points"])
        cs = get_centroid(source_parts[k])
        Fs = get_frame(source_parts[k], part_name=k)

        # Target
        Xt = np.asarray(target_parts[k]["points"])
        ct = get_centroid(target_parts[k])
        Ft = get_frame(target_parts[k], part_name=k)

        # Light gray scatter clouds (non-interfering)
        fig_traces.append(
            go.Scatter3d(
                x=Xs[:,0], y=Xs[:,1], z=Xs[:,2],
                mode="markers",
                marker=dict(size=1.8, color="rgba(200,200,200,0.25)"),
                name=f"{k} src pts",
                showlegend=False
            )
        )
        fig_traces.append(
            go.Scatter3d(
                x=Xt[:,0], y=Xt[:,1], z=Xt[:,2],
                mode="markers",
                marker=dict(size=1.8, color="rgba(150,150,150,0.25)"),
                name=f"{k} tgt pts",
                showlegend=False
            )
        )

        # Centroid markers (batched for legend clarity)
        src_centroid_x.append(cs[0]); src_centroid_y.append(cs[1]); src_centroid_z.append(cs[2])
        src_centroid_c.append(src_color); src_centroid_text.append(f"{k} (src)")

        tgt_centroid_x.append(ct[0]); tgt_centroid_y.append(ct[1]); tgt_centroid_z.append(ct[2])
        tgt_centroid_c.append(tgt_color); tgt_centroid_text.append(f"{k} (tgt)")

        # Frames: source solid, target dashed (same X/Y/Z colors)
        fig_traces += axes_traces(cs, Fs, width=6, dash=None, opacity=1.0)
        fig_traces += axes_traces(ct, Ft, width=6, dash="dash", opacity=0.95)

        # Vector from source centroid to target centroid (line)
        fig_traces.append(
            go.Scatter3d(
                x=[cs[0], ct[0]],
                y=[cs[1], ct[1]],
                z=[cs[2], ct[2]],
                mode="lines",
                line=dict(color=vec_color, width=6),
                name=f"{k} vector",
                showlegend=False,
                hoverinfo="text",
                text=[f"{k} cs->ct", f"{k} cs->ct"]
            )
        )

    # Add centroid markers as two grouped traces to keep legend short
    fig_traces.append(
        go.Scatter3d(
            x=src_centroid_x, y=src_centroid_y, z=src_centroid_z,
            mode="markers",
            marker=dict(size=5, color=src_centroid_c),
            name="Source centroids",
            text=src_centroid_text,
            hoverinfo="text"
        )
    )
    fig_traces.append(
        go.Scatter3d(
            x=tgt_centroid_x, y=tgt_centroid_y, z=tgt_centroid_z,
            mode="markers",
            marker=dict(size=5, color=tgt_centroid_c, symbol="diamond"),
            name="Target centroids",
            text=tgt_centroid_text,
            hoverinfo="text"
        )
    )

    if anchors is not None and len(anchors):
        # Build a canonical undirected edge set so we only plot once
        canonical_edges = {}
        for (i, j), pts in anchors.items():
            # canonical key by string ordering to stay stable for mixed types
            ci, cj = sorted([str(i), str(j)])
            key = (ci, cj)
            # Skip self or empty
            if i == j or pts is None or pts.shape[0] == 0:
                continue
            # Keep first occurrence
            if key not in canonical_edges:
                canonical_edges[key] = (i, j, pts)

        edge_palette = [
            "#e6194B","#3cb44b","#ffe119","#4363d8","#f58231",
            "#911eb4","#46f0f0","#f032e6","#bcf60c","#fabebe",
            "#008080","#e6beff","#9A6324","#fffac8","#800000",
            "#aaffc3","#808000","#ffd8b1","#000075","#808080"
        ]
        for e_idx, ((ci, cj), (i, j, pts)) in enumerate(canonical_edges.items()):
            col = edge_palette[e_idx % len(edge_palette)]
            fig_traces.append(
                go.Scatter3d(
                    x=pts[:,0], y=pts[:,1], z=pts[:,2],
                    mode="markers",
                    marker=dict(size=4, color=col, symbol="x"),
                    name=f"Seam {i}-{j}",
                    hoverinfo="text",
                    text=[f"anchor {i}-{j} #{k}" for k in range(pts.shape[0])],
                    opacity=0.9
                )
            )

    fig = go.Figure(data=fig_traces)
    fig.update_layout(
        title="Init frames and centroids (source vs target)",
        scene=dict(
            aspectmode="data",
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            xaxis=dict(showgrid=True, zeroline=False),
            yaxis=dict(showgrid=True, zeroline=False),
            zaxis=dict(showgrid=True, zeroline=False),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    out_path = os.path.join(log_dir, "init_frames.html")
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[Plotly] Wrote {out_path}")

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

    sample_name = "sample_0"
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
    export_faces = {k: source_faces[part_indices[k]] for k in part_names}

    vert_seg = np.zeros(source_verts.shape[0], dtype=int)
    source = {}
    target = {}
    for i, name in enumerate(part_names):
        vert_seg[part_indices[name]] = i
        source[name] = {}
        target[name] = {}
        source[name]["points"] = source_verts[part_indices[name]]
        source[name]["centroid"] = robust_centroid(source[name]["points"])
        source[name]["normals"] = compute_normals(source[name]["points"])
        target[name]["points"] = target_verts[part_indices[name]]
        target[name]["centroid"] = robust_centroid(target[name]["points"])
        target[name]["normals"] = compute_normals(target[name]["points"])

    log_dir = "fit_pointcloud_logs"
    exp_dir = f"smpl_rigid"
    exp_id = sample_name
    exp_sub_dir = f"exp_{exp_id}_2"
    log_dir = os.path.join(log_dir, exp_dir, exp_sub_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    shutil.copy(__file__, log_dir)
    graph = [
        ("root", "head_neck", 1.0),
        ("root", "left_upper_arm", 1.0),
        ("root", "right_upper_arm", 1.0),
        ("root", "left_upper_leg", 1.0),
        ("root", "right_upper_leg", 1.0),
        ("left_upper_arm", "left_lower_arm", 1.0),
        ("right_upper_arm", "right_lower_arm", 1.0),
        ("left_lower_leg", "left_upper_leg", 1.0),
        ("right_lower_leg", "right_upper_leg", 1.0),
    ]
    # NEW: build per-part up vectors from adjacency graph
    up_src, up_tgt = build_up_hints_from_graph(
        source, target, graph, strategy="lexi"
    )

    up_hint = np.array([0, 1, 0], dtype=np.float64)
    front_hint = np.array([0, 0, 1], dtype=np.float64)

    # for k in source.keys():
    #     source[k]["frame"] = pca_frame(
    #         source[k]["points"], source[k]["normals"], up_hint=up_src.get(k)
    #     )
    #     target[k]["frame"] = pca_frame(
    #         target[k]["points"], target[k]["normals"], up_hint=up_tgt.get(k)
    #     )
    # Fsrc = graph_consistent_frames(
    #     source, graph, root="root", anchors=None, up_hint=None
    # )
    # Ftgt = graph_consistent_frames(
    #     target, graph, root="root", anchors=None, up_hint=None
    # )
    # for k in source.keys():
    #     source[k]["frame"] = Fsrc[k]
    #     target[k]["frame"] = Ftgt[k]
    random_seed = 42
    seed_base = int(random_seed)
    solver_params = dict(
        outer_iters=15,
        keep_ratio_schedule=[0.7, 0.8, 0.85, 0.9],
        # keep_ratio_schedule=[1.0],
        # smooth_lambda=20.0,
        # boundary_gamma=80.0,
        # prior_mu=10.0,
        smooth_lambda=0.0,
        boundary_gamma=1.0,
        # prior_mu=1.0,
        prior_mu=0.0,
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
        front_hint=front_hint,
        up_hint=up_hint,
        z_mode="seam_centroid_to_centroid_direction",
        random_seed=seed_base,
        p2p_parts=[
            "left_upper_leg", 
            "right_upper_leg", 
            "left_lower_arm", 
            "right_lower_arm",
            "left_upper_arm",
            "right_upper_arm",
            "left_lower_leg",
            "right_lower_leg",
        ],
        # debug_parts=["left_lower_arm", "right_lower_arm"],
        debug_parts=[
            "left_upper_leg",
            "right_upper_leg",
            "left_lower_arm",
            "right_lower_arm",
            "left_upper_arm",
            "right_upper_arm",
            "left_lower_leg",
            "right_lower_leg",
        ],
        # debug_parts=["head_neck"],
        debug_plot=True,
        # flip_normals=["left_lower_arm", "right_lower_arm"],
        initial_axis_flip_enabled=True,
        initial_axis_flip_use_data_terms=True,
        double_sided_data_terms=True,
        use_sinkhorn_matches=True,
        use_sinkhorn_matches_initial=False,
        sinkhorn_kwargs=dict(
            eps=1e-2,
            tau=5e-1,
            n_iters=40,
        ),
        compute_normals_on_the_fly=True,
        initial_axis_flip_debug_plotly=True,
        initial_axis_flip_axes=["x", "y", "z"],
    )

    rng_source = np.random.default_rng(seed_base)
    rng_target = np.random.default_rng(seed_base + 1)

    # NEW: compute seam anchors from source/target parts with deterministic sampling
    source_anchors = compute_seam_anchors_from_parts(
        source,
        graph,
        k_per_edge=8,
        radius_scale=0.5,
        symmetric=True,
        rng=rng_source,
    )
    target_anchors = compute_seam_anchors_from_parts(
        target,
        graph,
        k_per_edge=8,
        radius_scale=0.5,
        symmetric=True,
        rng=rng_target,
    )

    T = estimate_part_rigid_transforms(
        source,
        target,
        graph,
        source_anchors=source_anchors,
        target_anchors=target_anchors,
        params=solver_params,
        log_dir=log_dir,
    )
    print("Keys:", list(T.keys()))

    print(f"Source shape: {source_verts.shape}")
    print(f"Target shape: {target_verts.shape}")
    print(f"Segmentation shape: {vert_seg.shape}, Clusters: {len(np.unique(vert_seg))}")

    deformed_points_by_part = transform_points_by_part(
        {k: v["points"] for k, v in source.items()},
        T,
        centers={
            k: v.get("centroid", v["points"].mean(axis=0)) for k, v in source.items()
        },
    )
    final_posed_vertices = compose_mesh_parts(
        deformed_points_by_part, part_indices, part_names
    )
    # You can now save `final_posed_vertices` as a mesh file (e.g., .obj)
    # to visualize the result.
    final_mesh = trimesh.Trimesh(
        vertices=final_posed_vertices, faces=source_faces, process=False
    )
    final_mesh.export(os.path.join(log_dir, "final_posed_mesh.obj"))

    # Calculate final error
    final_error = np.linalg.norm(final_posed_vertices - target_verts, axis=1).mean()
    print(f"\nFinal Mean Per-Vertex Error: {final_error:.6f}")
