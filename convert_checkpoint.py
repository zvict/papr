import torch
import copy
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.ops import iterative_closest_point as icp
import os

# base_path = "/NAS/spa176/pim/experiments/stand-stage-2/"
base_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-start-1/"
save_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-1/"
# create folder if not exists
os.makedirs(save_path, exist_ok=True)

# state_dict = torch.load(base_path + "model_old_before_rotation.pth")
state_dict = torch.load(base_path + "model_no_rot_fix_name.pth")
# duplicate the model.pth to the same directory as model_old.pth using shutil
# shutil.copy(base_path + "model.pth", base_path + "model_no_rot_fix_name.pth")
step = list(state_dict.keys())[0]
state_dict = state_dict[step]

# print("Original state_dict keys:")
# print(state_dict.keys())

# change the key name inside state_dict from "points_conf_scores" to "points_influ_scores"
# state_dict["points_influ_scores"] = state_dict.pop("points_conf_scores")

# load deformed pc
deformed_pc = torch.load(
    "/NAS/spa176/papr-retarget/fit_pointcloud_logs/multi_mlp_icp_shift_pe8_pointnet_densify_concat/transform_wing_kp_64_body_kp_96_kpnn_5_bodykpnn_20_frame_skip_200_cdw1000.0_rigidw1.0_ldasw0.0_nn100_concat_0/deformed_src_pc_start.pth"
)
# # load rotation matrix
# RTs = torch.load("/NAS/spa176/papr-retarget/bird_to_but_RTs.pth")
# # apply the inverse transformation to the deformed_pc
# R = RTs.R
# T = RTs.T
# print("R shape", R.shape)
# print("T shape", T.shape)
# # inverse of R, R is of shape (1, 3, 3)
# R = R.squeeze(0)
# T = T.squeeze(0)
# R = R.t()
# T = -R @ T
# deformed_pc = deformed_pc @ R + T
deformed_pc = deformed_pc * 10.0

converged, rmse, Xt, RTs, t_history = icp(
    deformed_pc.unsqueeze(0), state_dict["points"].unsqueeze(0)
)
print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
# deformed_pc = Xt.squeeze(0)

R = RTs.R
T = RTs.T
# deformed_pc = (torch.bmm(deformed_pc.unsqueeze(0), R) + T[:, None, :]).squeeze(0)

# load the batch of deformed_pc
total_deformed_pcs = torch.load(
    "/NAS/spa176/papr-retarget/fit_pointcloud_logs/multi_mlp_icp_shift_pe8_pointnet_densify_concat/transform_wing_kp_64_body_kp_96_kpnn_5_bodykpnn_20_frame_skip_200_cdw1000.0_rigidw1.0_ldasw0.0_nn100_concat_0/total_deformed_pc.pth"
)  # shape (T, N, 3)
print("total_deformed_pcs shape", total_deformed_pcs.shape)
# apply the transformation
total_deformed_pcs = total_deformed_pcs * 10.0
total_deformed_pcs = torch.bmm(total_deformed_pcs, R.expand(total_deformed_pcs.shape[0], 3, 3)) + T[:, None, :].expand(total_deformed_pcs.shape[0], -1, 3)

# save the transformed deformed_pc
# torch.save(total_deformed_pcs, base_path + "total_deformed_pc_new.pth")
deformed_pc = total_deformed_pcs[0]

# plot the deformed_pc in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
plot_pc = deformed_pc.cpu().numpy()
ax.scatter(plot_pc[:, 0], plot_pc[:, 1], plot_pc[:, 2])
# limit the axis to the same range as the original pc
ax.set_xlim(-5, 5)
ax.set_ylim(-3, 3)
ax.set_zlim(-5, 8)
plt.savefig("deformed_pc_4.png")
plt.close()
# # plot the original pc in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# plot_pc = state_dict["points"].cpu().numpy()
# ax.scatter(plot_pc[:, 0], plot_pc[:, 1], plot_pc[:, 2])
# # limit the axis to the same range as the original pc
# ax.set_xlim(-5, 5)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-5, 8)
# plt.savefig("original_pc.png")
# plt.close()

# exit(0)


################ For saving a new checkpoint with the deformed_pc ################
# for each point in the deformed_pc, find the closest point in the original pc and copy the influence score and pc_feat from the original pc to the deformed pc

num_pts = deformed_pc.shape[0]
new_points_influ_scores = torch.empty(num_pts, 1, device=state_dict["points_influ_scores"].device)
new_pc_feat = torch.empty(
    num_pts,
    state_dict["pc_feats"].shape[1],
    device=state_dict["pc_feats"].device,
)

print("deformed_pc shape", deformed_pc.shape)
print("state_dict points shape", state_dict["points"].shape)
print("points_influ_scores shape", state_dict["points_influ_scores"].shape)
print("pc_feats shape", state_dict["pc_feats"].shape)

for i in range(deformed_pc.shape[0]):
    deformed_point = deformed_pc[i]
    diff = state_dict["points"] - deformed_point
    dist = torch.norm(diff, dim=1)
    min_dist, min_idx = torch.min(dist, dim=0)
    new_points_influ_scores[i] = state_dict["points_influ_scores"][min_idx]
    new_pc_feat[i] = state_dict["pc_feats"][min_idx]


state_dict["points_influ_scores"] = new_points_influ_scores
state_dict["pc_feats"] = new_pc_feat
state_dict["points"] = deformed_pc.to(state_dict["points"].device)

save_sd = {step: state_dict}
torch.save(save_sd, save_path + "model.pth")
#################################################################################
