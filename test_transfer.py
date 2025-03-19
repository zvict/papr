import yaml
import argparse
import torch
import os
import io
import shutil
from PIL import Image
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from logger import *
from dataset import get_dataset, get_loader
from models import get_model, get_loss
import lpips
import tqdm
from tqdm import tqdm
import open3d as o3d
from scipy.signal import windows

try:
    from skimage.measure import compare_ssim
except:
    from skimage.metrics import structural_similarity

    def compare_ssim(gt, img, win_size, channel_axis=2):
        return structural_similarity(
            gt, img, win_size=win_size, channel_axis=channel_axis
        )


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="PAPR")
    parser.add_argument("--opt", type=str, default="", help="Option file path")
    parser.add_argument("--resume", type=int, default=250000, help="Resume step")
    return parser.parse_args()


def test_step(
    frame,
    num_frames,
    model,
    device,
    dataset,
    batch,
    loss_fn,
    lpips_loss_fn_alex,
    lpips_loss_fn_vgg,
    args,
    test_losses,
    test_psnrs,
    test_ssims,
    test_lpips_alexs,
    test_lpips_vggs,
    resume_step,
):
    idx, _, img, rayd, rayo = batch
    c2w = dataset.get_c2w(idx.squeeze())

    N, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    img = img.to(device)
    c2w = c2w.to(device)

    topk = min([num_pts, model.select_k])
    pt_idxs = [topk * i // 5 for i in range(5)]

    # if args.models.transformer.embed.share_embed:
    #     encode = torch.zeros(N, H, W, len(pt_idxs), args.models.transformer.embed.d_ff_out)
    # else:
    #     encode = torch.zeros(N, H, W, len(pt_idxs), args.models.transformer.embed.value.d_ff_out)

    selected_points = torch.zeros(1, H, W, topk, 3)

    bkg_seq_len_attn = 0
    tx_opt = args.models.transformer
    feat_dim = (
        tx_opt.embed.d_ff_out
        if tx_opt.embed.share_embed
        else tx_opt.embed.value.d_ff_out
    )
    if model.bkg_feats is not None:
        bkg_seq_len_attn = model.bkg_feats.shape[0]
    feature_map = torch.zeros(N, H, W, 1, feat_dim).to(device)
    attn = torch.zeros(N, H, W, topk + bkg_seq_len_attn, 1).to(device)

    with torch.no_grad():
        for height_start in range(0, H, args.test.max_height):
            for width_start in range(0, W, args.test.max_width):
                height_end = min(height_start + args.test.max_height, H)
                width_end = min(width_start + args.test.max_width, W)

                (
                    feature_map[
                        :, height_start:height_end, width_start:width_end, :, :
                    ],
                    attn[:, height_start:height_end, width_start:width_end, :, :],
                ) = model.evaluate(
                    rayo,
                    rayd[:, height_start:height_end, width_start:width_end],
                    c2w,
                    step=resume_step,
                )

                selected_points[
                    :, height_start:height_end, width_start:width_end, :, :
                ] = model.selected_points

        if args.models.use_renderer:
            foreground_rgb = (
                model.renderer(feature_map.squeeze(-2).permute(0, 3, 1, 2))
                .permute(0, 2, 3, 1)
                .unsqueeze(-2)
            )  # (N, H, W, 1, 3)
            if model.bkg_feats is not None:
                bkg_attn = attn[..., topk:, :]
                if args.models.normalize_topk_attn:
                    rgb = (
                        foreground_rgb * (1 - bkg_attn)
                        + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                    )
                    bkg_mask = (
                        model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                    ).squeeze()
                else:
                    rgb = (
                        foreground_rgb
                        + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                    )
                    bkg_mask = (
                        model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                    ).squeeze()
                rgb = rgb.squeeze(-2)
            else:
                rgb = foreground_rgb.squeeze(-2)
            foreground_rgb = foreground_rgb.squeeze()
        else:
            rgb = feature_map.squeeze(-2)

        rgb = model.last_act(rgb)
        rgb = torch.clamp(rgb, 0, 1)

        test_loss = loss_fn(rgb, img)
        test_psnr = -10.0 * np.log(((rgb - img) ** 2).mean().item()) / np.log(10.0)
        test_ssim = compare_ssim(
            rgb.squeeze().detach().cpu().numpy(),
            img.squeeze().detach().cpu().numpy(),
            11,
            channel_axis=2,
        )
        test_lpips_alex = (
            lpips_loss_fn_alex(rgb.permute(0, 3, 1, 2), img.permute(0, 3, 1, 2))
            .squeeze()
            .item()
        )
        test_lpips_vgg = (
            lpips_loss_fn_vgg(rgb.permute(0, 3, 1, 2), img.permute(0, 3, 1, 2))
            .squeeze()
            .item()
        )

    test_losses.append(test_loss.item())
    test_psnrs.append(test_psnr)
    test_ssims.append(test_ssim)
    test_lpips_alexs.append(test_lpips_alex)
    test_lpips_vggs.append(test_lpips_vgg)

    print(
        f"Test frame: {frame}, test_loss: {test_losses[-1]:.4f}, test_psnr: {test_psnrs[-1]:.4f}, test_ssim: {test_ssims[-1]:.4f}, test_lpips_alex: {test_lpips_alexs[-1]:.4f}, test_lpips_vgg: {test_lpips_vggs[-1]:.4f}"
    )

    od = -rayo
    D = torch.sum(od * rayo)
    dists = torch.abs(
        torch.sum(selected_points.to(od.device) * od, -1) - D
    ) / torch.norm(od)
    if (
        model.bkg_feats is not None
        and model.bkg_type == 1
        and resume_step <= args.training.bkg_step
    ):
        dists = torch.cat(
            [dists, torch.ones(N, H, W, model.bkg_feats.shape[0]).to(dists.device) * 0],
            dim=-1,
        )
    cur_depth = (
        (torch.sum(attn.squeeze(-1).to(od.device) * dists, dim=-1))
        .detach()
        .cpu()
        .squeeze()
        .numpy()
        .astype(np.float32)
    )
    depth_np = cur_depth.copy()

    # To save the rendered images, depth maps, foreground rgb, and background mask
    # log_dir = os.path.join(args.save_dir, args.index, 'test', 'images')
    # os.makedirs(log_dir, exist_ok=True)
    # cur_depth /= args.dataset.coord_scale
    # cur_depth *= (65536 / 10)
    # cur_depth = cur_depth.astype(np.uint16)
    # imageio.imwrite(os.path.join(log_dir, "test-{:04d}-predrgb-PSNR{:.3f}-SSIM{:.4f}-LPIPSA{:.4f}-LPIPSV{:.4f}.png".format(frame, test_psnr, test_ssim, test_lpips_alex, test_lpips_vgg)), (rgb.squeeze().detach().cpu().numpy() * 255).astype(np.uint8))
    # imageio.imwrite(os.path.join(log_dir, "test-{:04d}-depth-PSNR{:.3f}-SSIM{:.4f}-LPIPSA{:.4f}-LPIPSV{:.4f}.png".format(frame, test_psnr, test_ssim, test_lpips_alex, test_lpips_vgg)), cur_depth)
    # imageio.imwrite(os.path.join(log_dir, "test-{:04d}-fgrgb-PSNR{:.3f}-SSIM{:.4f}-LPIPSA{:.4f}-LPIPSV{:.4f}.png".format(frame, test_psnr, test_ssim, test_lpips_alex, test_lpips_vgg)), (foreground_rgb.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8))
    # imageio.imwrite(os.path.join(log_dir, "test-{:04d}-bkgmask-PSNR{:.3f}-SSIM{:.4f}-LPIPSA{:.4f}-LPIPSV{:.4f}.png".format(frame, test_psnr, test_ssim, test_lpips_alex, test_lpips_vgg)), (bkg_mask.detach().cpu().numpy() * 255).astype(np.uint8))

    plots = {}

    coord_scale = args.dataset.coord_scale
    if "Barn" in args.dataset.path:
        coord_scale *= 1.5
    if "Family" in args.dataset.path:
        coord_scale *= 0.5
    pt_plot_scale = 1.0 * coord_scale

    plot_opt = args.test.plots
    th = -frame * (360.0 / num_frames)
    azims = np.linspace(180, -180, num_frames)
    azmin = azims[frame]

    points_np = model.points.detach().cpu().numpy()
    rgb_pred_np = rgb.squeeze().detach().cpu().numpy().astype(np.float32)
    rgb_gt_np = img.squeeze().detach().cpu().numpy().astype(np.float32)
    points_conf_scores_np = None
    if model.points_conf_scores is not None:
        points_conf_scores_np = (
            model.points_conf_scores.squeeze().detach().cpu().numpy()
        )

    if plot_opt.pcrgb:
        pcrgb_plot = get_test_pcrgb(
            frame,
            th,
            azmin,
            test_psnr,
            points_np,
            rgb_pred_np,
            rgb_gt_np,
            depth_np,
            pt_plot_scale,
            points_conf_scores_np,
        )
        plots["pcrgb"] = pcrgb_plot

    if (
        plot_opt.featattn
    ):  # Note that these plots are not necessarily meaningful since each ray has different top K points
        featmap_np = feature_map[0].squeeze().detach().cpu().numpy().astype(np.float32)
        attn_np = attn[0].squeeze().detach().cpu().numpy().astype(np.float32)
        get_test_featmap_attn(
            frame,
            th,
            points_np,
            rgb_pred_np,
            rgb_gt_np,
            pt_plot_scale,
            featmap_np,
            attn_np,
            points_conf_scores_np,
        )
        plots["featattn"] = img

    return plots


def get_depth(pc, rayo):
    od = -rayo
    dists = np.abs(np.sum((pc - rayo) * od, axis=-1)) / np.linalg.norm(od)
    return dists


def white2alpha(img):
    img = img.convert("RGBA")
    pixdata = img.load()

    width, height = img.size
    for y in range(height):
        for x in range(width):
            #             if pixdata[x, y] == (255, 255, 255, 255):
            if (
                pixdata[x, y][0] > 230
                and pixdata[x, y][1] > 230
                and pixdata[x, y][2] > 230
            ):
                pixdata[x, y] = (255, 255, 255, 0)
    return img


def get_batch(dataloader, index):
    for i, batch in enumerate(dataloader):
        if i == index:
            return batch
    raise IndexError("Index out of range")


def plot_pc(pc, colors, feat=False, focal=0.17):
    elev, azim, roll = -90, -90, 0
    pltscale = 0.03
    size = 12.8
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(projection="3d")
    ax.set_proj_type("persp", focal_length=focal)
    # ax.set_proj_type('ortho')
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.set_axis_off()

    if feat is True:
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], alpha=1.0, s=50, c=colors)
    else:
        cnames = ["viridis", "plasma", "inferno", "magma", "cividis"]
        cname = "plasma"
        # cname = 'Spectral'
        cmap = matplotlib.colormaps.get_cmap(cname)
        ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], alpha=1.0, s=80, c=cmap(-colors + 1))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    lim = pltscale * 1.0
    xlim = lim * 1.0
    ylim = lim * 1.0
    zlim = lim * 1.0
    # xlim = lim * 1.1
    # ylim = lim * 1.2
    # zlim = lim * 0.9
    # ax.set_xlim3d(-lim, lim)
    ax.set_xlim3d(-xlim, xlim)
    ax.set_ylim3d(-ylim, ylim)
    ax.set_zlim3d(-zlim, zlim)

    plt.tight_layout()
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)

    plt.close()

    return img


def get_all_weights_with_prefix(model_state_dict, prefix):
    weights = {}
    for key, value in model_state_dict.items():
        if prefix in key:
            weights[key] = value.detach().cpu()
    return weights


def smooth_point_cloud(point_clouds, window_size):
    # Create an empty list to store the smoothed point clouds
    smoothed_point_clouds = []

    # Pad the list of point clouds with copies of the first and last point clouds
    padded_point_clouds = ([point_clouds[0]] * (window_size // 2)) + point_clouds + ([point_clouds[-1]] * (window_size // 2))

    # Apply the moving average filter
    for i in range(window_size // 2, len(padded_point_clouds) - window_size // 2):
        # Get the window of point clouds for this iteration
        window = padded_point_clouds[i - window_size // 2 : i + window_size // 2 + 1]

        # Calculate the average point cloud for this window
        average_point_cloud = np.mean(window, axis=0)

        # Add the average point cloud to the list of smoothed point clouds
        smoothed_point_clouds.append(average_point_cloud)

    # Set the first and last point clouds to be the same as the original ones
    smoothed_point_clouds[0] = point_clouds[0]
    smoothed_point_clouds[-1] = point_clouds[-1]

    return smoothed_point_clouds


def plot_all(frame, testloader, args, model, device, smooth):
    deformed_pc_path = os.path.join(args.save_dir, args.index, args.eval.pc_file_path)
    batch = get_batch(testloader, frame)
    if smooth:
        rgb_save_dir = os.path.join(args.save_dir, args.index, f"rgb_smooth_{frame}")
        os.makedirs(rgb_save_dir, exist_ok=True)
        pc_save_dir = os.path.join(args.save_dir, args.index, f"pc_smooth_{frame}")
        os.makedirs(pc_save_dir, exist_ok=True)
    else:
        rgb_save_dir = os.path.join(args.save_dir, args.index, f"rgb_{frame}")
        os.makedirs(rgb_save_dir, exist_ok=True)
        pc_save_dir = os.path.join(args.save_dir, args.index, f"pc_{frame}")
        os.makedirs(pc_save_dir, exist_ok=True)
    idx, _, img, rayd, rayo = batch
    c2w = testloader.dataset.get_c2w(idx.squeeze())
    c2w_np = c2w.detach().cpu().numpy()
    # focal_x = testloader.dataset.focal_x

    total_deformed_pcs = torch.load(deformed_pc_path)

    # load "load_path_end" model
    # end_model_state_dict = torch.load(
    #     os.path.join(args.save_dir, args.load_path_end, "model.pth")
    # )
    # for step, state_dict in end_model_state_dict.items():
    #     end_model = state_dict
    # end_points = end_model["points"].data.cpu().numpy()
    # end_pc_feats = end_model["pc_feats"].data.cpu().numpy()
    # end_transformer_weights = get_all_weights_with_prefix(end_model, "transformer")
    # init_transformer_weights = get_all_weights_with_prefix(
    #     model.state_dict(), "transformer"
    # )

    result = {}

    N, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    img = img.to(device)
    c2w = c2w.to(device)

    topk = min([num_pts, model.select_k])
    pt_idxs = [topk * i // 5 for i in range(5)]
    # selected_points = torch.zeros(1, H, W, topk, 3)

    bkg_seq_len_attn = 0
    tx_opt = args.models.transformer
    feat_dim = (
        tx_opt.embed.d_ff_out
        if tx_opt.embed.share_embed
        else tx_opt.embed.value.d_ff_out
    )
    if model.bkg_feats is not None:
        bkg_seq_len_attn = model.bkg_feats.shape[0]

    # max_iter = (
    #     20000 if "max_iters" not in args.eval.keys() else int(args.eval.max_iters)
    # )
    skip_num = 1 if "skip_num" not in args.eval.keys() else int(args.eval.skip_num)
    focal = 0.17 if "focal" not in args.eval.keys() else float(args.eval.focal)
    dist_type = 1 if "dist_type" not in args.eval.keys() else int(args.eval.dist_type)
    # # find all point cloud files and sort them by name, the name convention is points_{step}.npy
    # pc_files = [
    #     os.path.join(pc_dir, f)
    #     for f in os.listdir(pc_dir)
    #     if os.path.isfile(os.path.join(pc_dir, f))
    #     and f.startswith("points_")
    #     and f.endswith(".npy")
    #     and int(os.path.splitext(os.path.basename(f))[0][7:]) <= max_iter
    # ]
    # pc_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0][7:]))

    # orignal_pc_length = len(pc_files)
    # if "load_path_finetune" in args.keys():
    #     print(f"!!!!!load finetune point clouds at {args.load_path_finetune}")
    #     pc_dir = os.path.join(args.save_dir, args.load_path_finetune, "point_clouds")
    #     max_finetune_iter = (
    #         20000
    #         if "max_finetune_iters" not in args.keys()
    #         else int(args.max_finetune_iters)
    #     )
    #     finetune_pc_files = [
    #         os.path.join(pc_dir, f)
    #         for f in os.listdir(pc_dir)
    #         if os.path.isfile(os.path.join(pc_dir, f))
    #         and f.startswith("points_")
    #         and f.endswith(".npy")
    #         and int(os.path.splitext(os.path.basename(f))[0][7:]) <= max_finetune_iter
    #     ]
    #     finetune_pc_files.sort(
    #         key=lambda f: int(os.path.splitext(os.path.basename(f))[0][7:])
    #     )
    #     finetune_skip = (
    #         2 if "finetune_skip" not in args.keys() else int(args.finetune_skip)
    #     )
    #     finetune_pc_files = finetune_pc_files[::finetune_skip]
    #     pc_files.extend(finetune_pc_files)

    # print(pc_files)
    rgb_results = []
    pc_results = []

    # # load all the pc files
    # pcs = [np.load(pc_file) for pc_file in pc_files]

    # select every 3 files in pc_files
    # pc_files = pc_files[::skip_num]

    # pc_files.append(end_points)

    # load all the pc files
    # pcs = [np.load(pc_file) if isinstance(pc_file, str) else pc_file for pc_file in pc_files]
    total_deformed_pcs = total_deformed_pcs.detach().cpu().numpy()
    total_deformed_pcs = [
        total_deformed_pcs[i] for i in range(total_deformed_pcs.shape[0])
    ]
    if skip_num > 1:
        total_deformed_pcs = total_deformed_pcs[::skip_num]

    if smooth:
        smoothing_window_size = 7 if "smoothing_window_size" not in args.eval.keys() else int(args.eval.smoothing_window_size)
        total_deformed_pcs = smooth_point_cloud(
            total_deformed_pcs, smoothing_window_size
        )

    # init_points = pcs[0]
    # init_pc_feats = model.pc_feats.data.cpu().numpy()
    # max_diff = np.sum(np.linalg.norm(init_points - end_points, axis=1))

    # design 2, load the end state dict
    # model.load_my_state_dict(
    #     end_model, exclude_keys=["points", "pc_feats", "points_conf_scores"]
    # )
    # model = model.to(device)
    # counter = 0
    # for pc_file in tqdm(pc_files):
    for cur_idx in tqdm(range(len(total_deformed_pcs))):
        # for pc_file in pc_files:
        # if isinstance(pc_file, str):
        #     pc = np.load(pc_file)
        # else:
        #     pc = pc_file
        # pc = pcs[counter]
        pc = total_deformed_pcs[cur_idx]
        model.points = torch.nn.Parameter(torch.from_numpy(pc).to(device))

        # cur_max_pt_diffs = np.sum(np.linalg.norm(init_points - pc, axis=1))
        # cur_ratio = min(cur_max_pt_diffs / max_diff, 1.0)

        # interpolate the point feature
        # cur_pc_feats = init_pc_feats * (1.0 - cur_ratio) + end_pc_feats * cur_ratio
        # print(f"cur_ratio: {cur_ratio}, init_pc_feats min: {init_pc_feats.min()}, max: {init_pc_feats.max()}, end_pc_feats min: {end_pc_feats.min()}, max: {end_pc_feats.max()}, cur_pc_feats min: {cur_pc_feats.min()}, max: {cur_pc_feats.max()}")
        # model.pc_feats = torch.nn.Parameter(torch.from_numpy(cur_pc_feats).to(device))
        # # # design 1, interpolate transformer weights
        # for key, value in init_transformer_weights.items():
        #     cur_value = (
        #         value * (1.0 - cur_ratio) + end_transformer_weights[key] * cur_ratio
        #     )
        #     model.state_dict()[key].copy_(cur_value.to(device))

        feature_map = torch.zeros(N, H, W, 1, feat_dim).to(device)
        attn = torch.zeros(N, H, W, topk + bkg_seq_len_attn, 1).to(device)

        with torch.no_grad():
            for height_start in range(0, H, args.eval.max_height):
                for width_start in range(0, W, args.eval.max_width):
                    height_end = min(height_start + args.eval.max_height, H)
                    width_end = min(width_start + args.eval.max_width, W)

                    (
                        feature_map[
                            :, height_start:height_end, width_start:width_end, :, :
                        ],
                        attn[:, height_start:height_end, width_start:width_end, :, :],
                    ) = model.evaluate(
                        rayo,
                        rayd[:, height_start:height_end, width_start:width_end],
                        c2w,
                        step=resume_step,
                    )

            if args.models.use_renderer:
                foreground_rgb = model.renderer(feature_map.squeeze(-2).permute(0, 3, 1, 2)).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
                if model.bkg_feats is not None:
                    bkg_attn = attn[..., topk:, :]
                    if args.models.normalize_topk_attn:
                        rgb = foreground_rgb * (1 - bkg_attn) + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                        # bkg_mask = (model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn).squeeze()
                    else:
                        rgb = foreground_rgb + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                        # bkg_mask = (model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn).squeeze()
                    rgb = rgb.squeeze(-2)
                else:
                    rgb = foreground_rgb.squeeze(-2)
                foreground_rgb = foreground_rgb.squeeze()
            else:
                rgb = feature_map.squeeze(-2)

            rgb = model.last_act(rgb)
            rgb = torch.clamp(rgb, 0, 1)

        rgb = rgb.squeeze().detach().cpu().numpy().astype(np.float32)
        rgb = (rgb * 255).astype(np.uint8)
        rgb_results.append(rgb)
        rgb = Image.fromarray(rgb)
        # save number in two digits
        save_img_name = f"{cur_idx:02d}.jpg"

        rgb.save(os.path.join(rgb_save_dir, save_img_name))

        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pc))
        blender2opencv = np.array(
            [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
        )
        pc.transform(np.linalg.inv(c2w_np @ blender2opencv))
        cur_points = np.asarray(pc.points)
        # cur_points = pc
        if dist_type == 0:
            dists = np.sqrt((cur_points**2).sum(axis=1))
        else:
            dists = get_depth(cur_points, c2w_np[:3, -1])
        colors = (dists - dists.min()) / (dists.max() - dists.min())
        img = plot_pc(cur_points, colors, focal=focal)

        # img = plot_pc_cam(cur_points, c2w_np, colors, W, focal_x)
        # convert img from RGBA to RGB with white background
        img.convert("RGB").save(os.path.join(pc_save_dir, save_img_name))
        pc_results.append(img)
        # counter += 1

    result["rgb"] = rgb_results
    result["pc"] = pc_results
    return result


def test(model, device, dataset, save_name, args, resume_step):
    # log_dir = os.path.join(args.save_dir, args.index, "test", "videos")
    # os.makedirs(log_dir, exist_ok=True)
    # pc_dir = os.path.join(args.save_dir, args.index, "point_clouds")

    smooth = args.eval.smooth
    testloader = get_loader(dataset, args.dataset, mode="test")
    print("testloader:", testloader)

    frame = args.eval.img_idx

    frames = plot_all(
        frame,
        testloader,
        args,
        model,
        device,
        smooth
    )

    if frames:
        for key, value in frames.items():
            # name = f"{args.index}-PSNR{test_psnr:.3f}-SSIM{test_ssim:.4f}-LPIPSA{test_lpips_alex:.4f}-LPIPSV{test_lpips_vgg:.4f}-{key}-{save_name}-step{resume_step}.mp4"
            # In case the name is too long
            if smooth:
                name = f"{args.index}-{frame}-{key}-smooth.mp4"
            else:
                name = f"{args.index}-{frame}-{key}.mp4"
            f = os.path.join(args.save_dir, args.index, name)
            imageio.mimwrite(f, value, fps=24, quality=10)
            if smooth:
                name = f"{args.index}-{frame}-{key}-loop-smooth.mp4"
            else:
                name = f"{args.index}-{frame}-{key}-loop.mp4"
            # name = name[-255:] if len(name) > 255 else name
            f = os.path.join(args.save_dir, args.index, name)
            # revese value and append to itself
            new_value = value[::-1]
            for _ in range(4):
                value.append(value[-1])
            value.extend(new_value)
            imageio.mimwrite(f, value, fps=24, quality=10)

    # print(
    #     f"Avg test loss: {test_loss:.4f}, test PSNR: {test_psnr:.4f}, test SSIM: {test_ssim:.4f}, test LPIPS Alex: {test_lpips_alex:.4f}, test LPIPS VGG: {test_lpips_vgg:.4f}"
    # )


def main(args, save_name, mode, resume_step=0):
    # log_dir = os.path.join(args.save_dir, args.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)
    dataset = get_dataset(args.dataset, mode=mode)

    if args.test.load_path:
        try:
            resume_step = model.load(os.path.join(args.save_dir, args.load_path))
        except:
            model_state_dict = torch.load(
                os.path.join(args.save_dir, args.load_path, "model.pth")
            )
            for step, state_dict in model_state_dict.items():
                resume_step = int(step)
                model.load_my_state_dict(state_dict)
        print(
            "!!!!! Loaded model from %s at step %s" % (args.test.load_path, resume_step)
        )
    else:
        try:
            model_state_dict = torch.load(
                os.path.join(args.save_dir, args.index, "model.pth")
            )
            for step, state_dict in model_state_dict.items():
                resume_step = int(step)
                model.load_my_state_dict(state_dict)
        except:
            model.load_my_state_dict(
                torch.load(
                    os.path.join(args.save_dir, args.index, f"model_{resume_step}.pth")
                )
            )
        print(
            "!!!!! Loaded model from %s at step %s"
            % (os.path.join(args.save_dir, args.index), resume_step)
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    test(model, device, dataset, save_name, args, resume_step)


if __name__ == "__main__":
    args = parse_args()
    with open(args.opt, "r") as f:
        config = yaml.safe_load(f)

    resume_step = args.resume

    log_dir = os.path.join(config["save_dir"], config["index"])
    os.makedirs(log_dir, exist_ok=True)

    # sys.stdout = Logger(os.path.join(log_dir, 'test.log'), sys.stdout)
    # sys.stderr = Logger(os.path.join(log_dir, 'test_error.log'), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    setup_seed(config["seed"])

    for i, dataset in enumerate(config["test"]["datasets"]):
        name = dataset["name"]
        mode = dataset["mode"]
        print(name, dataset)
        config["dataset"].update(dataset)
        args = DictAsMember(config)
        main(args, name, mode, resume_step)
