import numpy as np
import torch

src_kp_0 = np.load(
    "/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body256/part0_src_kp.npy"
)
src_kp_1 = np.load(
    "/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body256/part1_src_kp.npy"
)

src_kp = np.concatenate((src_kp_0, src_kp_1), axis=0)

src_kp_indices_0 = (
    torch.load(
        "/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body256/part0_src_kp_idx.pth"
    )
    .cpu()
    .numpy()
)
src_kp_indices_1 = (
    torch.load(
        "/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body256/part1_src_kp_idx.pth"
    )
    .cpu()
    .numpy()
)
src_full_pc = (
    np.load("/NAS/spa176/papr-retarget/point_clouds/butterfly/points_30000.npy") / 10.0
)
but_wing_indices = np.load("but_wing_indices.npy")
but_body_indices = np.setdiff1d(np.arange(len(src_full_pc)), but_wing_indices)

src_last_kp_0 = src_full_pc[but_wing_indices][src_kp_indices_0]
src_last_kp_1 = src_full_pc[but_body_indices][src_kp_indices_1]
src_last_kp = np.concatenate((src_last_kp_0[0], src_last_kp_1[0]), axis=0)

# Save the indexed points to another file
np.save("/NAS/spa176/papr-retarget/pcs/src_kp_0", src_kp)
np.save("/NAS/spa176/papr-retarget/pcs/src_kp_30000", src_last_kp)
