import torch
import copy
import shutil
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.ops import iterative_closest_point as icp
import os


nn_body = 24
kpnn_body = 24
body_kp = 256
wing_kp = 96
# wing_kp = 192
# body_kp = 512
# reg_D = True
reg_D = False
inp_case = 0
# force_smooth = True
force_smooth = False
force_smooth_full = True
smooth_knn = 50

# reg_boundary = True
reg_boundary = False
boundary_threshold = 0.05

if body_kp == 256:
    target_exp_index = 9
elif body_kp == 512:
    target_exp_index = 11
elif body_kp == 192:
    target_exp_index = 10
elif body_kp == 96:
    target_exp_index = 8
else:
    raise ValueError("Invalid body_kp value")

if wing_kp == 96:
    target_exp_index = 12
# inp_case = 3
# base_path = "/NAS/spa176/pim/experiments/stand-stage-2/"
base_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-start-1/"
# save_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-6-kp-nnwing5-nnbody250-wkpnn50-bkpnn256-regD/"
# save_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-5-kp-nnwing5-nnbody190-wkpnn50-bkpnn192-regD-MSE/"
# save_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-8-kp-nnwing5-nnbody{nn_body}-wkpnn50-bkpnn8-regD/"
# save_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-8-kp-nnwing5-nnbody{nn_body}-wkpnn50-bkpnn{kpnn_body}/"
if reg_D:
    save_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-{target_exp_index}-kp-nnwing5-nnbody{nn_body}-wkpnn50-bkpnn{kpnn_body}-regD/"
else:
    if inp_case > 0:
        save_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-{target_exp_index}-kp-nnwing5-nnbody{nn_body}-wkpnn50-bkpnn{kpnn_body}-inp{inp_case}/"
    else:
        save_path = f"/NAS/spa176/papr-retarget/experiments/hummingbird-ft-{target_exp_index}-kp-nnwing5-nnbody{nn_body}-wkpnn50-bkpnn{kpnn_body}/"
if force_smooth:
    save_path = save_path[:-1] + "-smooth/"
elif force_smooth_full:
    save_path = save_path[:-1] + "-smooth-full/"
if reg_boundary:
    save_path = save_path[:-1] + "-RB/"
# save_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-9-kp-nnwing5-nnbody250-wkpnn50-bkpnn256-regD/"
# model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-start-1/model.pth"
model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-9-kp-nnwing5-nnbody16-wkpnn50-bkpnn16/model.pth"
# model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-8-kp-nnwing5-nnbody90-wkpnn50-bkpnn8-regD/model.pth"
# model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-8-kp-nnwing5-nnbody24-wkpnn50-bkpnn8-regD/model.pth"
# model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-4-kp-nnwing5-nnbody15-wkpnn50-bkpnn96-w1000/model.pth"
# model_path = "/NAS/spa176/papr-retarget/experiments/hummingbird-ft-5-kp-nnwing5-nnbody15-wkpnn50-bkpnn192-w1000/model.pth"
# exp_path = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/multi_mlp_icp_shift_pe8_pointnet_densify_concat_fps/L1_fix_wing_kp_192_body_kp_96_kpnn_50_bodykpnn_96_cdw1000.0_rigidw1.0_nnwing5_nnbody90_kp_only_regD/"
# exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body96/L1_fix_wing_kp_192_body_kp_96_kpnn_50_bodykpnn_8_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_regD_p3dNorm/"
# exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing192_body96/L1_fix_wing_kp_192_body_kp_96_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm/"
if reg_D:
    exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing{wing_kp}_body{body_kp}/L1_fix_wing_kp_192_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_regD_p3dNorm/"
else:
    if inp_case > 0:
        exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing{wing_kp}_body{body_kp}/L1_fix_wing_kp_192_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm_inp{inp_case}/"
    else:
        exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wing{wing_kp}_body{body_kp}/L1_fix_wing_kp_192_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm/"
# if force_smooth:
#     exp_path = exp_path[:-1] + "_smooth/"
if reg_boundary:
    exp_path = exp_path[:-1] + f"_RB/"
    # exp_path = exp_path[:-1] + f"_RB{boundary_threshold}/"
# exp_path = "/NAS/spa176/papr-retarget/fit_pointcloud_logs/multi_mlp_icp_shift_pe8_pointnet_densify_concat/L1_fix_wing_kp_192_body_kp_256_kpnn_50_bodykpnn_256_cdw1000.0_rigidw1.0_nnwing5_nnbody250_kp_only_regD/"
if wing_kp == 96:
    exp_path = f"/NAS/spa176/papr-retarget/fit_pointcloud_logs/ot_wingL{wing_kp}_wingR{wing_kp}_body{body_kp}/L1_fix_wing_kp_{wing_kp}_body_kp_{body_kp}_kpnn_50_bodykpnn_{kpnn_body}_cdw1.0_rigidw1.0_nnwing5_nnbody{nn_body}_kp_only_p3dNorm/"

# create folder if not exists
os.makedirs(save_path, exist_ok=True)
# copy the model fine to the save_path
shutil.copy(model_path, save_path + "model.pth")

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
    os.path.join(exp_path, "deformed_src_pc_start.pth")
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
    deformed_pc.unsqueeze(0), state_dict["points"].unsqueeze(0), max_iterations=300
)
print(f"ICP converged: {converged}, RMSE: {rmse}, Iterations: {len(t_history)}, Final Transformation: {Xt.shape}")
# deformed_pc = Xt.squeeze(0)

R = RTs.R
T = RTs.T
# deformed_pc = (torch.bmm(deformed_pc.unsqueeze(0), R) + T[:, None, :]).squeeze(0)

if force_smooth:
    deform_pc_name = "total_deformed_pc_smooth.pth"
elif force_smooth_full:
    if smooth_knn != 100:
        deform_pc_name = f"total_deformed_pc_smooth_full_{smooth_knn}.pth"
    else:
        deform_pc_name = "total_deformed_pc_smooth_full.pth"
else:
    deform_pc_name = "total_deformed_pc.pth"
# load the batch of deformed_pc
total_deformed_pcs = torch.load(
    os.path.join(exp_path, deform_pc_name)
)  # shape (T, N, 3)
print("total_deformed_pcs shape", total_deformed_pcs.shape)
# apply the transformation
total_deformed_pcs = total_deformed_pcs * 10.0
total_deformed_pcs = torch.bmm(total_deformed_pcs, R.expand(total_deformed_pcs.shape[0], 3, 3)) + T[:, None, :].expand(total_deformed_pcs.shape[0], -1, 3)

if force_smooth:
    save_pc_name = "total_deformed_pc_smooth.pth"
elif force_smooth_full:
    save_pc_name = f"total_deformed_pc_smooth_full_{smooth_knn}.pth"
else:
    save_pc_name = "total_deformed_pc_new.pth"
# save the transformed deformed_pc
torch.save(total_deformed_pcs, save_path + save_pc_name)
deformed_pc = total_deformed_pcs[0]

# # plot the deformed_pc in 3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# plot_pc = deformed_pc.cpu().numpy()
# ax.scatter(plot_pc[:, 0], plot_pc[:, 1], plot_pc[:, 2])
# # limit the axis to the same range as the original pc
# ax.set_xlim(-5, 5)
# ax.set_ylim(-3, 3)
# ax.set_zlim(-5, 8)
# plt.savefig("deformed_pc_4.png")
# plt.close()
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


# ################ For saving a new checkpoint with the deformed_pc ################
# # for each point in the deformed_pc, find the closest point in the original pc and copy the influence score and pc_feat from the original pc to the deformed pc

# num_pts = deformed_pc.shape[0]
# new_points_influ_scores = torch.empty(num_pts, 1, device=state_dict["points_influ_scores"].device)
# new_pc_feat = torch.empty(
#     num_pts,
#     state_dict["pc_feats"].shape[1],
#     device=state_dict["pc_feats"].device,
# )

# print("deformed_pc shape", deformed_pc.shape)
# print("state_dict points shape", state_dict["points"].shape)
# print("points_influ_scores shape", state_dict["points_influ_scores"].shape)
# print("pc_feats shape", state_dict["pc_feats"].shape)

# for i in range(deformed_pc.shape[0]):
#     deformed_point = deformed_pc[i]
#     diff = state_dict["points"] - deformed_point
#     dist = torch.norm(diff, dim=1)
#     min_dist, min_idx = torch.min(dist, dim=0)
#     new_points_influ_scores[i] = state_dict["points_influ_scores"][min_idx]
#     new_pc_feat[i] = state_dict["pc_feats"][min_idx]


# state_dict["points_influ_scores"] = new_points_influ_scores
# state_dict["pc_feats"] = new_pc_feat
# state_dict["points"] = deformed_pc.to(state_dict["points"].device)

# save_sd = {step: state_dict}
# torch.save(save_sd, save_path + "model.pth")
# #################################################################################
