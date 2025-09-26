"""
rig_reg_deform_graph_merged.py
--------------------------------
Merged pipeline:
    Stage 1: per-part rigid alignment (numpy), graph-regularized, using estimate_part_rigid_transforms()
    Stage 2-4: deformation graph optimization (reuse rig_reg_deform_graph_adapted.py helpers)

This file wires stages together and accepts a kNN-style part graph adjacency constructed
from target segmentation. Stage 1 builds per-part dicts (points, centroid, normals), seam
anchors, and calls estimate_part_rigid_transforms from part_rigid_alignment.py, then exports
the posed rigid mesh and continues with stages 2-4.
"""

import os
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional, Union

# Import the original pipeline for stages 2-4
import rig_reg_deform_graph_adapted as dg
from part_rigid_alignment import (
    estimate_part_rigid_transforms,
    compute_seam_anchors_from_parts,
    se3_apply, compute_normals, robust_centroid
)


def build_parts_from_seg(points: np.ndarray, part_indices: Dict[Any, np.ndarray]) -> Dict[Any, dict]:
    parts: Dict[Any, dict] = {}
    for pid, mask in part_indices.items():
        P = points[mask]
        # Prepare like part_rigid_alignment main: points, centroid, normals
        parts[pid] = {
            "points": P,
            "centroid": robust_centroid(P),
            "normals": compute_normals(P),
        }
    return parts



def build_graph_from_knn(
    part_indices: Dict[Any, np.ndarray],
    knn_adj: Union[np.ndarray, List[Tuple[int, int]]],
    weight: float = 1.0,
) -> List[Tuple[Any, Any, float]]:
    """
    Convert kNN-style adjacency into part graph edges (i,j,w).
    - labels: ordered list of part IDs (keys of parts dict). Index maps to id.
    - knn_adj: either a boolean/int adjacency matrix (P,P) or a list of index pairs.
    Returns undirected unique edges using provided label ids.
    """
    parts = list(part_indices.keys())
    P = len(parts)
    edges: set[Tuple[int, int]] = set()
    if isinstance(knn_adj, np.ndarray):
        assert knn_adj.shape[0] == P and knn_adj.shape[1] == P, "adjacency must be (P,P)"
        for i in range(P):
            for j in range(i + 1, P):
                if knn_adj[i, j]:
                    edges.add((i, j))
    else:
        for (i, j) in knn_adj:
            a, b = (int(i), int(j))
            if a == b:
                continue
            if a > b:
                a, b = b, a
            edges.add((a, b))
    return [(parts[i], parts[j], float(weight)) for (i, j) in sorted(edges)]


def run_merged_pipeline(
    source_vertices: np.ndarray,
    source_faces: np.ndarray,
    target_vertices: np.ndarray,
    target_segmentation: np.ndarray,
    part_indices: Dict[Any, np.ndarray],
    *,
    source_segmentation: Optional[np.ndarray] = None,
    log_dir: str = "logs_merged",
    rigid_params: Optional[dict] = None,
    deform_kwargs: Optional[dict] = None,
) -> np.ndarray:
    os.makedirs(log_dir, exist_ok=True)

    # Build per-part dicts
    tgt_parts = build_parts_from_seg(target_vertices, part_indices)
    if source_segmentation is None:
        # assume same labeling as target
        src_parts = build_parts_from_seg(source_vertices, part_indices)
    else:
        src_parts = build_parts_from_seg(source_vertices, part_indices)

    # Build kNN part graph on part centroids (simple proximity)
    # Here reuse adjacency from dg.infer_part_adjacency on torch then convert to numpy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Xt = torch.from_numpy(source_vertices).float()
    # labels = torch.from_numpy(source_segmentation).long().to(device)
    # A, parts_order = dg.infer_part_adjacency_boundary(Xt, part_indices)
    # TODO: fix the following issues for infer_part_adjacency_boundary:
    # 1. left_upper_leg and right_upper_leg are close to each other, causing them to be connected in the graph, this is a case where 
    # they are not connected in the source mesh, but initially close and are in parallel, so we should find a way to avoid this
    # 2. there are duplicate in the graph in the form of (a,b) and (b,a), need to be removed and only keep one of them
    A, parts_order = dg.infer_part_adjacency_boundary(Xt, part_indices)
    # Convert to numpy adjacency matching dict key order (assume keys are sorted by parts_order)
    # ordered_keys = [int(p) for p in parts_order]
    # Remap to dense index order of our dicts
    # key_to_idx = {k: i for i, k in enumerate(ordered_keys)}
    # P = len(ordered_keys)
    A_np = A.detach().cpu().numpy().astype(bool)
    # Rigid stage accepts either adjacency or explicit edges; we'll pass adjacency with matching order
    # Ensure dicts are ordered according to ordered_keys
    # src_parts = {k: src_parts[k] for k in ordered_keys}
    # tgt_parts = {k: tgt_parts[k] for k in ordered_keys}

    # Convert adjacency to edge list (i,j,w) in terms of label IDs
    # TODO: temporaily hardcode a simple tree graph for testing
    # graph_edges = build_graph_from_knn(part_indices, A_np, weight=1.0)
    graph_edges = [
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

    # print out graph edges
    print("Part graph edges (i,j,w):")
    for (i, j, w) in graph_edges:
        print(f"  {i} -- {j} (w={w})")

    # # Build vertex index lists per part for export/composition
    # part_indices = {int(pid): np.where((source_segmentation if source_segmentation is not None else target_segmentation) == pid)[0]
    #                 for pid in ordered_keys}

    random_seed = 0
    if rigid_params and "random_seed" in rigid_params:
        random_seed = rigid_params["random_seed"]
    if random_seed is None:
        random_seed = 0
    seed_base = int(random_seed)

    rng_source = np.random.default_rng(seed_base)
    rng_target = np.random.default_rng(seed_base + 1)

    # Build seam anchors from parts (mirroring part_rigid_alignment main)
    rigid_dir = os.path.join(log_dir, "rigid")
    try:
        source_anchors = compute_seam_anchors_from_parts(
            src_parts,
            graph_edges,
            k_per_edge=32,
            radius_scale=2.0,
            symmetric=True,
            rng=rng_source,
        )
    except Exception:
        source_anchors = None
    try:
        target_anchors = compute_seam_anchors_from_parts(
            tgt_parts,
            graph_edges,
            k_per_edge=32,
            radius_scale=2.0,
            symmetric=True,
            rng=rng_target,
        )
    except Exception:
        target_anchors = None

    # Stage 1: rigid using estimate_part_rigid_transforms (maps SOURCE -> TARGET)
    rigid_params_full = dict(
        outer_iters=15,
        keep_ratio_schedule=[0.7, 0.8, 0.85, 0.9],
        lm_damp=1e-6,
        verbose=True,
        export_meshes=True,
        log_dir=rigid_dir,
        source_faces=source_faces,
        part_indices=part_indices,
        init_mode="graph",
        # root_part_id=ordered_keys[0] if len(ordered_keys) else None,
        root_part_id="root",
        z_mode="seam_centroid_to_centroid_direction",
        random_seed=seed_base,
    )
    if rigid_params:
        rigid_params_full.update(rigid_params)
        rigid_params_full["random_seed"] = seed_base

    T = estimate_part_rigid_transforms(
        src_parts,
        tgt_parts,
        graph_edges,
        source_anchors=source_anchors,
        target_anchors=target_anchors,
        params=rigid_params_full,
        log_dir=rigid_dir,
    )

    # Apply per-part transforms to SOURCE vertices to produce an initial posed mesh for stage 2
    posed_parts = {}
    for k, d in src_parts.items():
        X = d["points"]
        c = d.get("centroid", X.mean(axis=0))
        posed_parts[k] = se3_apply(T[k]["R"], T[k]["t"], X, center=c)
    # Recompose posed mesh to original vertex order using part_indices
    N = int(np.max([idxs.max() for idxs in part_indices.values()]) + 1)
    posed_vertices = np.zeros((N, 3), dtype=source_vertices.dtype)
    for pid in part_indices.keys():
        idx = part_indices[pid]
        posed_vertices[idx] = posed_parts[pid]
    # Export posed rigid mesh
    try:
        import trimesh
        os.makedirs(rigid_dir, exist_ok=True)
        trimesh.Trimesh(vertices=posed_vertices, faces=source_faces, process=False).export(
            os.path.join(rigid_dir, "posed_rigid.obj")
        )
    except Exception:
        try:
            os.makedirs(rigid_dir, exist_ok=True)
            np.save(os.path.join(rigid_dir, "posed_rigid.npy"), posed_vertices)
        except Exception:
            pass

    # Stage 2-4: deformation graph starting from posed target
    deform_args = dict(
        source_vertices=posed_vertices,  # start from rigidly posed source
        source_faces=source_faces,
        target_vertices=target_vertices,
        target_segmentation=target_segmentation,
        log_dir=os.path.join(log_dir, "deform"),
    )
    if deform_kwargs:
        deform_args.update(deform_kwargs)
    # Skip Stage 1 (we already did rigid); optionally run 2-4 only
    deform_args.update({
        # "stages": [2,3,4],
        "stages": [1,2,3,4],
        "skip_per_part_rigid_init": True,
    })
    out_vertices = dg.run_pipeline_deform_graph(**deform_args)
    return out_vertices


__all__ = ["run_merged_pipeline", "build_parts_from_seg"]
