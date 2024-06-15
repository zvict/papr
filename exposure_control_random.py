import yaml
import argparse
import torch
import torch.nn as nn
import os
import io
import shutil
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import numpy as np
import random
import sys
from logger import *
from dataset import get_dataset, get_loader
from models import get_model, get_loss
import lpips
try:
    from skimage.measure import compare_ssim
except:
    from skimage.metrics import structural_similarity

    def compare_ssim(gt, img, win_size, channel_axis=2):
        return structural_similarity(gt, img, win_size=win_size, channel_axis=channel_axis, data_range=1.0)


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
    parser.add_argument('--opt', type=str, default="", help='Option file path')
    parser.add_argument('--resume', type=int, default=250000, help='Resume step')
    parser.add_argument('--resample', action='store_true', help='[Exposure control] Resample the latent code')
    parser.add_argument('--seed', type=int, default=1, help='[Exposure control] Random seed')
    parser.add_argument('--frame', type=int, default=0, help='[Exposure control] Test frame index')
    parser.add_argument('--scale', type=float, default=1.0, help='[Exposure control] Shading code scale')
    parser.add_argument('--num_samples', type=int, default=20, help='[Exposure control] Number of samples for random exposure control')
    return parser.parse_args()


def test_step(frame, num_frames, model, device, dataset, batch, loss_fn, lpips_loss_fn_alex, lpips_loss_fn_vgg, args, test_losses, test_psnrs, test_ssims, test_lpips_alexs, test_lpips_vggs, resume_step, resample, shading_codes, seed, i, scale):
    idx, _, img, rayd, rayo = batch
    c2w = dataset.get_c2w(idx.squeeze())

    if resample == False:
        cur_shading_code = shading_codes[0]
        # cur_shading_code = torch.randn(args.models.shading_code_dim, device=device) * args.models.shading_code_scale
    else:
        print("Before resample shading codes, eval shading codes mean: ", shading_codes[idx].mean(), shading_codes[idx].shape, shading_codes.shape)
        resample_shading_codes(shading_codes, args, model, dataset, idx, loss_fn, resume_step, scale)
        print("After resample shading codes, eval shading codes mean: ", shading_codes[idx].mean(), shading_codes[idx].shape, shading_codes.shape)
        cur_shading_code = shading_codes[idx].squeeze()

    N, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    img = img.to(device)
    c2w = c2w.to(device)

    topk = min([num_pts, model.select_k])
    selected_points = torch.zeros(1, H, W, topk, 3)

    bkg_seq_len_attn = 0
    tx_opt = args.models.transformer
    feat_dim = tx_opt.embed.d_ff_out if tx_opt.embed.share_embed else tx_opt.embed.value.d_ff_out
    if model.bkg_feats is not None:
        bkg_seq_len_attn = model.bkg_feats.shape[0]
    feature_map = torch.zeros(N, H, W, 1, feat_dim).to(device)
    attn = torch.zeros(N, H, W, topk + bkg_seq_len_attn, 1).to(device)

    with torch.no_grad():
        cur_affine = model.mapping_mlp(cur_shading_code)
        cur_affine_dim = cur_affine.shape[-1]
        cur_gamma, cur_beta = cur_affine[:cur_affine_dim // 2], cur_affine[cur_affine_dim // 2:]

        for height_start in range(0, H, args.test.max_height):
            for width_start in range(0, W, args.test.max_width):
                height_end = min(height_start + args.test.max_height, H)
                width_end = min(width_start + args.test.max_width, W)

                feature_map[:, height_start:height_end, width_start:width_end, :, :], \
                attn[:, height_start:height_end, width_start:width_end, :, :] = model.evaluate(rayo, rayd[:, height_start:height_end, width_start:width_end], c2w, step=resume_step)

                selected_points[:, height_start:height_end, width_start:width_end, :, :] = model.selected_points

        foreground_rgb = model.renderer(feature_map.squeeze(-2).permute(0, 3, 1, 2), gamma=cur_gamma, beta=cur_beta).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
            
        if model.bkg_feats is not None:
            bkg_attn = attn[..., topk:, :]
            bkg_mask = (model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn).squeeze()
            if args.models.normalize_topk_attn:
                rgb = foreground_rgb * (1 - bkg_attn) + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
            else:
                rgb = foreground_rgb + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
            rgb = rgb.squeeze(-2)
        else:
            rgb = foreground_rgb.squeeze(-2)
            bkg_mask = torch.zeros(N, H, W, 1).to(device)

        rgb = model.last_act(rgb)
        rgb = torch.clamp(rgb, 0, 1)

        test_loss = loss_fn(rgb, img)
        test_psnr = -10. * np.log(((rgb - img)**2).mean().item()) / np.log(10.)
        test_ssim = compare_ssim(rgb.squeeze().detach().cpu().numpy(), img.squeeze().detach().cpu().numpy(), 11, channel_axis=2)
        test_lpips_alex = lpips_loss_fn_alex(rgb.permute(0, 3, 1, 2), img.permute(0, 3, 1, 2)).squeeze().item()
        test_lpips_vgg = lpips_loss_fn_vgg(rgb.permute(0, 3, 1, 2), img.permute(0, 3, 1, 2)).squeeze().item()

    test_losses.append(test_loss.item())
    test_psnrs.append(test_psnr)
    test_ssims.append(test_ssim)
    test_lpips_alexs.append(test_lpips_alex)
    test_lpips_vggs.append(test_lpips_vgg)

    code_mean = cur_shading_code.mean().item()
    print(f"Test frame: {frame}, code mean: {code_mean}, test_loss: {test_losses[-1]:.4f}, test_psnr: {test_psnrs[-1]:.4f}, test_ssim: {test_ssims[-1]:.4f}, test_lpips_alex: {test_lpips_alexs[-1]:.4f}, test_lpips_vgg: {test_lpips_vggs[-1]:.4f}")

    od = -rayo
    D = torch.sum(od * rayo)
    dists = torch.abs(torch.sum(selected_points.to(od.device) * od, -1) - D) / torch.norm(od)
    if model.bkg_feats is not None:
        dists = torch.cat([dists, torch.ones(N, H, W, model.bkg_feats.shape[0]).to(dists.device) * 0], dim=-1)
    cur_depth = (torch.sum(attn.squeeze(-1).to(od.device) * dists, dim=-1)).detach().cpu().squeeze().numpy().astype(np.float32)
    depth_np = cur_depth.copy()

    if args.test.save_fig:
        # To save the rendered images, depth maps, foreground rgb, and background mask
        log_dir = os.path.join(args.save_dir, args.index, 'test', f'exposure_control_random_resample{resample}_seed{seed}_std{scale}')
        os.makedirs(log_dir, exist_ok=True)
        cur_depth /= args.dataset.coord_scale
        cur_depth *= (65536 / 10)
        cur_depth = cur_depth.astype(np.uint16)
        imageio.imwrite(os.path.join(log_dir, "test-{:04d}-{:02d}-predrgb-codeMean{:.4f}-PSNR{:.3f}-SSIM{:.4f}-LPIPSA{:.4f}-LPIPSV{:.4f}.png".format(frame, i, code_mean, test_psnr, test_ssim, test_lpips_alex, test_lpips_vgg)), (rgb.squeeze().detach().cpu().numpy() * 255).astype(np.uint8))
        # imageio.imwrite(os.path.join(log_dir, "test-{:04d}-{:02d}-depth-codeMean{:.4f}-PSNR{:.3f}-SSIM{:.4f}-LPIPSA{:.4f}-LPIPSV{:.4f}.png".format(frame, i, code_mean, test_psnr, test_ssim, test_lpips_alex, test_lpips_vgg)), cur_depth)
        # imageio.imwrite(os.path.join(log_dir, "test-{:04d}-{:02d}-fgrgb-codeMean{:.4f}-PSNR{:.3f}-SSIM{:.4f}-LPIPSA{:.4f}-LPIPSV{:.4f}.png".format(frame, i, code_mean, test_psnr, test_ssim, test_lpips_alex, test_lpips_vgg)), (foreground_rgb.squeeze().clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8))
        # imageio.imwrite(os.path.join(log_dir, "test-{:04d}-{:02d}-bkgmask-codeMean{:.4f}-PSNR{:.3f}-SSIM{:.4f}-LPIPSA{:.4f}-LPIPSV{:.4f}.png".format(frame, i, code_mean, test_psnr, test_ssim, test_lpips_alex, test_lpips_vgg)), (bkg_mask.detach().cpu().numpy() * 255).astype(np.uint8))

    plots = {}

    # if args.test.save_video:
    #     # To save the rendered videos
    #     coord_scale = args.dataset.coord_scale
    #     if "Barn" in args.dataset.path:
    #         coord_scale *= 1.5
    #     if "Family" in args.dataset.path:
    #         coord_scale *= 0.5
    #     pt_plot_scale = 1.0 * coord_scale

    #     plot_opt = args.test.plots
    #     th = -frame * (360. / num_frames)
    #     azims = np.linspace(180, -180, num_frames)
    #     azmin = azims[frame]

    #     points_np = model.points.detach().cpu().numpy()
    #     rgb_pred_np = rgb.squeeze().detach().cpu().numpy().astype(np.float32)
    #     rgb_gt_np = img.squeeze().detach().cpu().numpy().astype(np.float32)
    #     points_influ_scores_np = None
    #     if model.points_influ_scores is not None:
    #         points_influ_scores_np = model.points_influ_scores.squeeze().detach().cpu().numpy()

    #     if plot_opt.pcrgb:
    #         pcrgb_plot = get_test_pcrgb(frame, th, azmin, test_psnr, points_np,
    #                                     rgb_pred_np, rgb_gt_np, depth_np, pt_plot_scale, points_influ_scores_np)
    #         plots["pcrgb"] = pcrgb_plot

    #     if plot_opt.featattn:   # Note that these plots are not necessarily meaningful since each ray has different top K points
    #         featmap_np = feature_map[0].squeeze().detach().cpu().numpy().astype(np.float32)
    #         attn_np = attn[0].squeeze().detach().cpu().numpy().astype(np.float32)
    #         featattn_plot = get_test_featmap_attn(frame, th, points_np, rgb_pred_np, rgb_gt_np,
    #                             pt_plot_scale, featmap_np, attn_np, points_influ_scores_np)
    #         plots["featattn"] = featattn_plot

    return plots


def resample_shading_codes(shading_codes, args, model, dataset, img_id, loss_fn, step, full_img=False, scale=1.0):
    if full_img == True:
        img, rayd, rayo = dataset.get_full_img(img_id)
        c2w = dataset.get_c2w(img_id)
    else:
        _, _, img, rayd, rayo = dataset[img_id]
        c2w = dataset.get_c2w(img_id)
        img = torch.from_numpy(img).unsqueeze(0)
        rayd = torch.from_numpy(rayd).unsqueeze(0)
        rayo = torch.from_numpy(rayo).unsqueeze(0)

    sampled_shading_codes = torch.randn(args.models.shading_code_num_samples, args.models.shading_code_dim, device=model.device) * scale
    
    N, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(model.device)
    rayd = rayd.to(model.device)
    img = img.to(model.device)
    c2w = c2w.to(model.device)
    
    topk = min([num_pts, model.select_k])

    bkg_seq_len_attn = 0
    tx_opt = args.models.transformer
    feat_dim = tx_opt.embed.d_ff_out if tx_opt.embed.share_embed else tx_opt.embed.value.d_ff_out
    if model.bkg_feats is not None:
        bkg_seq_len_attn = model.bkg_feats.shape[0]
    feature_map = torch.zeros(N, H, W, 1, feat_dim).to(model.device)
    attn = torch.zeros(N, H, W, topk + bkg_seq_len_attn, 1).to(model.device)
    
    best_idx = 0
    best_loss = 1e10
    best_loss_idx = 0
    best_psnr = 0
    best_psnr_idx = 0

    with torch.no_grad():
        for height_start in range(0, H, args.eval.max_height):
            for width_start in range(0, W, args.eval.max_width):
                height_end = min(height_start + args.eval.max_height, H)
                width_end = min(width_start + args.eval.max_width, W)

                feature_map[:, height_start:height_end, width_start:width_end, :, :], \
                attn[:, height_start:height_end, width_start:width_end, :, :] = model.evaluate(rayo, rayd[:, height_start:height_end, width_start:width_end], c2w, step=step)
        
        for i in range(args.models.shading_code_num_samples):
            torch.cuda.empty_cache()
            cur_shading_code = sampled_shading_codes[i]
            cur_affine = model.mapping_mlp(cur_shading_code)
            cur_affine_dim = cur_affine.shape[-1]
            cur_gamma, cur_beta = cur_affine[:cur_affine_dim // 2], cur_affine[cur_affine_dim // 2:]
            
            foreground_rgb = model.renderer(feature_map.squeeze(-2).permute(0, 3, 1, 2), gamma=cur_gamma, beta=cur_beta).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)

            if model.bkg_feats is not None:
                bkg_attn = attn[..., topk:, :]
                if args.models.normalize_topk_attn:
                    rgb = foreground_rgb * (1 - bkg_attn) + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                else:
                    rgb = foreground_rgb + model.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
                rgb = rgb.squeeze(-2)
            else:
                rgb = foreground_rgb.squeeze(-2)

            rgb = model.last_act(rgb)
            # rgb = torch.clamp(rgb, 0, 1)

            eval_loss = loss_fn(rgb, img)
            eval_psnr = -10. * np.log(((rgb - img)**2).mean().item()) / np.log(10.)
            
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_loss_idx = i

            if eval_psnr > best_psnr:
                best_psnr = eval_psnr
                best_psnr_idx = i

            model.clear_grad()

    # print("Best loss:", best_loss, "Best loss idx:", best_loss_idx, "Best psnr:", best_psnr, "Best psnr idx:", best_psnr_idx)
    best_idx = best_loss_idx if args.models.shading_code_resample_select_by == "loss" else best_psnr_idx
    shading_codes[img_id] = sampled_shading_codes[best_idx]

    del rayo, rayd, img, c2w, attn
    del eval_loss


def test(model, device, dataset, save_name, args, resume_step, resample, shading_codes, test_frame, seed, scale, num_samples):
    testloader = get_loader(dataset, args.dataset, mode="test")
    print("testloader:", testloader)

    loss_fn = get_loss(args.training.losses)
    loss_fn = loss_fn.to(device)

    lpips_loss_fn_alex = lpips.LPIPS(net='alex', version='0.1')
    lpips_loss_fn_alex = lpips_loss_fn_alex.to(device)
    lpips_loss_fn_vgg = lpips.LPIPS(net='vgg', version='0.1')
    lpips_loss_fn_vgg = lpips_loss_fn_vgg.to(device)

    test_losses = []
    test_psnrs = []
    test_ssims = []
    test_lpips_alexs = []
    test_lpips_vggs = []

    frames = {}
    for frame, batch in enumerate(testloader):
        if frame != test_frame:
            continue

        for i in range(num_samples):
            print("test seed:", seed, "i:", i)
            shading_codes = torch.randn(1, args.models.shading_code_dim, device=device) * scale
            plots = test_step(frame, len(testloader), model, device, dataset, batch, loss_fn, lpips_loss_fn_alex,
                            lpips_loss_fn_vgg, args, test_losses, test_psnrs, test_ssims, test_lpips_alexs, 
                            test_lpips_vggs, resume_step, resample, shading_codes, seed, i, scale)

            # if plots:
            #     for key, value in plots.items():
            #         if key not in frames:
            #             frames[key] = []
            #         frames[key].append(value)

    test_loss = np.mean(test_losses)
    test_psnr = np.mean(test_psnrs)
    test_ssim = np.mean(test_ssims)
    test_lpips_alex = np.mean(test_lpips_alexs)
    test_lpips_vgg = np.mean(test_lpips_vggs)

    # if frames:
    #     for key, value in frames.items():
    #         name = f"{args.index}-PSNR{test_psnr:.3f}-SSIM{test_ssim:.4f}-LPIPSA{test_lpips_alex:.4f}-LPIPSV{test_lpips_vgg:.4f}-{key}-{save_name}-step{resume_step}.mp4"
    #         # In case the name is too long
    #         name = name[-255:] if len(name) > 255 else name
    #         log_dir = os.path.join(args.save_dir, args.index, 'test', 'videos')
    #         os.makedirs(log_dir, exist_ok=True)
    #         f = os.path.join(log_dir, name)
    #         imageio.mimwrite(f, value, fps=30, quality=10)

    print(f"Avg test loss: {test_loss:.4f}, test PSNR: {test_psnr:.4f}, test SSIM: {test_ssim:.4f}, test LPIPS Alex: {test_lpips_alex:.4f}, test LPIPS VGG: {test_lpips_vgg:.4f}")


def main(args, save_name, mode, resume_step=0, resample=False, frame=0, seed=0, scale=1.0, num_samples=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(args, device)
    dataset = get_dataset(args.dataset, mode=mode)

    if args.test.load_path:
        try:
            model_state_dict = torch.load(args.test.load_path)
            for step, state_dict in model_state_dict.items():
                resume_step = int(step)
                model.train_shading_codes = nn.Parameter(torch.zeros_like(state_dict['train_shading_codes']), requires_grad=False)
                model.eval_shading_codes = nn.Parameter(torch.zeros_like(state_dict['eval_shading_codes']), requires_grad=False)
                model.load_my_state_dict(state_dict)
        except:
            model_state_dict = torch.load(os.path.join(args.save_dir, args.test.load_path, "model.pth"))
            for step, state_dict in model_state_dict.items():
                resume_step = step
                model.train_shading_codes = nn.Parameter(torch.zeros_like(state_dict['train_shading_codes']), requires_grad=False)
                model.eval_shading_codes = nn.Parameter(torch.zeros_like(state_dict['eval_shading_codes']), requires_grad=False)
                model.load_my_state_dict(state_dict)
        print("!!!!! Loaded model from %s at step %s" % (args.test.load_path, resume_step))
    else:
        try:
            model_state_dict = torch.load(os.path.join(args.save_dir, args.index, "model.pth"))
            for step, state_dict in model_state_dict.items():
                resume_step = int(step)
                model.train_shading_codes = nn.Parameter(torch.zeros_like(state_dict['train_shading_codes']), requires_grad=False)
                model.eval_shading_codes = nn.Parameter(torch.zeros_like(state_dict['eval_shading_codes']), requires_grad=False)
                model.load_my_state_dict(state_dict)
        except:
            state_dict = torch.load(os.path.join(args.save_dir, args.index, f"model_{resume_step}.pth"))
            model.train_shading_codes = nn.Parameter(torch.zeros_like(state_dict['train_shading_codes']), requires_grad=False)
            model.eval_shading_codes = nn.Parameter(torch.zeros_like(state_dict['eval_shading_codes']), requires_grad=False)
            model.load_my_state_dict(state_dict)
        print("!!!!! Loaded model from %s at step %s" % (os.path.join(args.save_dir, args.index), resume_step))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if mode == 'train':
        shading_codes = model.train_shading_codes
        print("Using train shading_codes:", shading_codes.shape, shading_codes.min(), shading_codes.max())
    elif mode == 'test':
        shading_codes = model.eval_shading_codes
        print("Using eval shading_codes:", shading_codes.shape, shading_codes.min(), shading_codes.max())
    else:
        raise NotImplementedError

    test(model, device, dataset, save_name, args, resume_step, resample, shading_codes, frame, seed, scale, num_samples)


if __name__ == '__main__':

    args = parse_args()
    seed = args.seed
    frame = args.frame
    resample = args.resample
    scale = args.scale
    num_samples = args.num_samples
    with open(args.opt, 'r') as f:
        config = yaml.safe_load(f)

    resume_step = args.resume

    log_dir = os.path.join(config["save_dir"], config['index'])
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(log_dir, 'test.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, 'test_error.log'), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    # setup_seed(config['seed'])
    setup_seed(seed)

    for i, dataset in enumerate(config['test']['datasets']):
        name = dataset['name']
        mode = dataset['mode']
        print(name, dataset)
        config['dataset'].update(dataset)
        args = DictAsMember(config)

        cur_scale = args.models.shading_code_scale if scale == 1.0 else scale
        print("Using shading code scale:", cur_scale)

        assert args.models.use_renderer, "Currently only support using renderer for exposure control"
        assert args.models.mapping_mlp.use, "Mapping MLP must be used for exposure control"

        main(args, name, mode, resume_step, resample, frame, seed, cur_scale, num_samples)
