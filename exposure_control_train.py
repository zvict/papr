import yaml
import argparse
import torch
import torch.nn as nn
import os
import shutil
import zipfile
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import bisect
import time
import sys
import io
import imageio
from tqdm import tqdm
from PIL import Image
from utils import *
from dataset import get_dataset, get_loader
from models import get_model, get_loss


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
    parser.add_argument('--resume', type=int, default=0, help='Resume training')
    return parser.parse_args()


def eval_step(steps, model, device, dataset, eval_dataset, batch, loss_fn, train_out, args, train_losses, eval_losses, eval_psnrs, pt_lrs, attn_lrs):
    step = steps[-1]
    train_img_idx, _, train_patch, _, _  = batch
    train_img, train_rayd, train_rayo = dataset.get_full_img(train_img_idx[0])
    img, rayd, rayo = eval_dataset.get_full_img(args.eval.img_idx)
    c2w = dataset.get_c2w(args.eval.img_idx)

    print("Before resample shading codes, eval shading codes mean: ", model.eval_shading_codes[args.eval.img_idx].mean())
    resample_shading_codes(model.eval_shading_codes, args, model, eval_dataset, args.eval.img_idx, loss_fn, step, full_img=True)
    print("After resample shading codes, eval shading codes mean: ", model.eval_shading_codes[args.eval.img_idx].mean())
    
    N, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    img = img.to(device)
    c2w = c2w.to(device)

    topk = min([num_pts, model.select_k])

    selected_points = torch.zeros(1, H, W, topk, 3)

    bkg_seq_len_attn = 0
    feat_dim = args.models.attn.embed.value.d_ff_out
    if model.bkg_feats is not None:
        bkg_seq_len_attn = model.bkg_feats.shape[0]
    feature_map = torch.zeros(N, H, W, 1, feat_dim).to(device)
    attn = torch.zeros(N, H, W, topk + bkg_seq_len_attn, 1).to(device)

    with torch.no_grad():
        cur_shading_code = model.eval_shading_codes[args.eval.img_idx]
        cur_affine = model.mapping_mlp(cur_shading_code)
        cur_affine_dim = cur_affine.shape[-1]
        cur_gamma, cur_beta = cur_affine[:cur_affine_dim // 2], cur_affine[cur_affine_dim // 2:]

        for height_start in range(0, H, args.eval.max_height):
            for width_start in range(0, W, args.eval.max_width):
                height_end = min(height_start + args.eval.max_height, H)
                width_end = min(width_start + args.eval.max_width, W)

                feature_map[:, height_start:height_end, width_start:width_end, :, :], \
                attn[:, height_start:height_end, width_start:width_end, :, :] = model.evaluate(rayo, rayd[:, height_start:height_end, width_start:width_end], c2w, step=step)

                selected_points[:, height_start:height_end, width_start:width_end, :, :] = model.selected_points
        
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
        rgb = torch.clamp(rgb, 0, 1)

        eval_loss = loss_fn(rgb, img)
        eval_psnr = -10. * np.log(((rgb - img)**2).mean().item()) / np.log(10.)

        model.clear_grad()

    eval_losses.append(eval_loss.item())
    eval_psnrs.append(eval_psnr.item())

    print("Eval step:", step, "train_loss:", train_losses[-1], "eval_loss:", eval_losses[-1], "eval_psnr:", eval_psnrs[-1])

    log_dir = os.path.join(args.save_dir, args.index)
    os.makedirs(log_dir, exist_ok=True)
    if args.eval.save_fig:
        os.makedirs(os.path.join(log_dir, "train_main_plots"), exist_ok=True)
        os.makedirs(os.path.join(log_dir, "train_pcd_plots"), exist_ok=True)

        coord_scale = args.dataset.coord_scale
        pt_plot_scale = 1.0 * coord_scale
        if "Barn" in args.dataset.path:
            pt_plot_scale *= 1.8
        if "Family" in args.dataset.path:
            pt_plot_scale *= 0.5    

        # calculate depth, weighted sum the distances from top K points to image plane
        od = -rayo
        D = torch.sum(od * rayo)
        dists = torch.abs(torch.sum(selected_points.to(od.device) * od, -1) - D) / torch.norm(od)
        if model.bkg_feats is not None:
            dists = torch.cat([dists, torch.ones(N, H, W, model.bkg_feats.shape[0]).to(dists.device) * 0], dim=-1)
        cur_depth = (torch.sum(attn.squeeze(-1).to(od.device) * dists, dim=-1)).detach().cpu()

        train_tgt_rgb = train_img.squeeze().cpu().numpy().astype(np.float32)
        train_tgt_patch = train_patch[0].cpu().numpy().astype(np.float32)
        train_pred_patch = train_out[0]
        test_tgt_rgb = img.squeeze().cpu().numpy().astype(np.float32)
        test_pred_rgb = rgb.squeeze().detach().cpu().numpy().astype(np.float32)
        points_np = model.points.detach().cpu().numpy()
        depth = cur_depth.squeeze().numpy().astype(np.float32)
        points_influ_scores_np = None
        if model.points_influ_scores is not None:
            points_influ_scores_np = model.points_influ_scores.squeeze().detach().cpu().numpy()

        # main plot
        main_plot = get_training_main_plot(args.index, steps, train_tgt_rgb, train_tgt_patch, train_pred_patch, test_tgt_rgb, test_pred_rgb, train_losses, 
                                           eval_losses, points_np, pt_plot_scale, depth, pt_lrs, attn_lrs, eval_psnrs, points_influ_scores_np)
        save_name = os.path.join(log_dir, "train_main_plots", "%s_iter_%d.png" % (args.index, step))
        main_plot.save(save_name)

        # point cloud plot
        ro = train_rayo.squeeze().detach().cpu().numpy()
        rd = train_rayd.squeeze().detach().cpu().numpy()
        
        pcd_plot = get_training_pcd_plot(args.index, steps[-1], ro, rd, points_np, args.dataset.coord_scale, pt_plot_scale, points_influ_scores_np)
        save_name = os.path.join(log_dir, "train_pcd_plots", "%s_iter_%d.png" % (args.index, step))
        pcd_plot.save(save_name)

    model.save(step, log_dir)
    if step % 50000 == 0:
        torch.save(model.state_dict(), os.path.join(log_dir, "model_%d.pth" % step))

    torch.save(torch.tensor(train_losses), os.path.join(log_dir, "train_losses.pth"))
    torch.save(torch.tensor(eval_losses), os.path.join(log_dir, "eval_losses.pth"))
    torch.save(torch.tensor(eval_psnrs), os.path.join(log_dir, "eval_psnrs.pth"))

    return 0


def train_step(step, model, device, dataset, batch, loss_fn, args):
    img_idx, _, tgt, rayd, rayo = batch
    c2w = dataset.get_c2w(img_idx[0])

    rayo = rayo.to(device)
    rayd = rayd.to(device)
    tgt = tgt.to(device)
    c2w = c2w.to(device)

    shading_code = model.train_shading_codes[img_idx[0]]

    model.clear_grad()
    out = model(rayo, rayd, c2w, step, shading_code=shading_code)
    out = model.last_act(out)
    loss = loss_fn(out, tgt)
    model.scaler.scale(loss).backward()
    model.step(step)
    if args.scaler_min_scale > 0 and model.scaler.get_scale() < args.scaler_min_scale:
        model.scaler.update(args.scaler_min_scale)
    else:
        model.scaler.update()

    return loss.item(), out.detach().cpu().numpy()


def train_and_eval(start_step, model, device, dataset, eval_dataset, sample_dataset, losses, args):
    trainloader = get_loader(dataset, args.dataset, mode="train")

    loss_fn = get_loss(args.training.losses)
    loss_fn = loss_fn.to(device)

    log_dir = os.path.join(args.save_dir, args.index)
    os.makedirs(os.path.join(log_dir, "test"), exist_ok=True)
    log_dir = os.path.join(log_dir, "test")

    steps = []
    train_losses, eval_losses, eval_psnrs = losses
    pt_lrs = []
    attn_lrs = []

    avg_train_loss = 0.
    step = start_step
    eval_step_cnt = start_step
    pc_frames = []

    model.train_shading_codes = nn.Parameter(torch.randn(len(dataset), args.exposure_control.shading_code_dim, device=device) * args.exposure_control.shading_code_scale, requires_grad=False)
    model.eval_shading_codes = nn.Parameter(torch.randn(len(eval_dataset), args.exposure_control.shading_code_dim, device=device) * args.exposure_control.shading_code_scale, requires_grad=False)
    print("!!!!! train_shading_codes:", model.train_shading_codes.shape, model.train_shading_codes.min(), model.train_shading_codes.max())
    print("!!!!! eval_shading_codes:", model.eval_shading_codes.shape, model.eval_shading_codes.min(), model.eval_shading_codes.max())

    print("Start step:", start_step, "Total steps:", args.training.steps)
    start_time = time.time()
    while step < args.training.steps:
        for _, batch in enumerate(trainloader):
            if step % args.exposure_control.shading_code_resample_iter == 0:  # Resample shading codes
                print("Resampling shading codes")
                print("Before resampling:", model.train_shading_codes.shape, model.train_shading_codes.min(), model.train_shading_codes.max())
                for img_idx in tqdm(range(len(sample_dataset))):
                    resample_shading_codes(model.train_shading_codes, args, model, sample_dataset, img_idx, loss_fn, step)
                print("After resampling:", model.train_shading_codes.shape, model.train_shading_codes.min(), model.train_shading_codes.max())

            loss, out = train_step(step, model, device, dataset, batch, loss_fn, args)
            avg_train_loss += loss
            step += 1
            eval_step_cnt += 1
            
            if step % 200 == 0:
                time_used = time.time() - start_time
                print("Train step:", step, "loss:", loss, "attn_lr:", model.attn_lr, "pts_lr:", model.pts_lr, "scale:", model.scaler.get_scale(), f"time: {time_used:.2f}s")
                print(model.mapping_mlp.model.model[7].weight[0, :5])
                start_time = time.time()

            if (step % args.eval.step == 0) or (step % 500 == 0 and step < 10000):
                train_losses.append(avg_train_loss / eval_step_cnt)
                pt_lrs.append(model.pts_lr)
                attn_lrs.append(model.attn_lr)
                steps.append(step)
                eval_step(steps, model, device, dataset, eval_dataset, batch, loss_fn, out, args, train_losses, eval_losses, eval_psnrs, pt_lrs, attn_lrs)
                avg_train_loss = 0.
                eval_step_cnt = 0

            if ((step - 1) % 200 == 0) and args.eval.save_fig:
                coord_scale = args.dataset.coord_scale
                pt_plot_scale = 0.8 * coord_scale
                if "Barn" in args.dataset.path:
                    pt_plot_scale *= 1.5
                if "Family" in args.dataset.path:
                    pt_plot_scale *= 0.5    

                pc_dir = os.path.join(log_dir, "point_clouds")
                os.makedirs(pc_dir, exist_ok=True)

                points_np = model.points.detach().cpu().numpy()
                points_influ_scores_np = None
                if model.points_influ_scores is not None:
                    points_influ_scores_np = model.points_influ_scores.squeeze().detach().cpu().numpy()
                pcd_plot = get_training_pcd_single_plot(step, points_np, pt_plot_scale, points_influ_scores_np)
                pc_frames.append(pcd_plot)
                
                if step == 1:
                    pcd_plot.save(os.path.join(pc_dir, "init_pcd.png"))

            if step >= args.training.steps:
                break

    if args.eval.save_fig and pc_frames != []:
        f = os.path.join(log_dir, f"{args.index}-pc.mp4")
        imageio.mimwrite(f, pc_frames, fps=30, quality=10)

    print("Training finished!")

            
def main(args, eval_args, sample_args, resume):
    log_dir = os.path.join(args.save_dir, args.index)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(args, device)
    dataset = get_dataset(args.dataset, mode="train")
    sample_dataset = get_dataset(sample_args.dataset, mode="train")
    eval_dataset = get_dataset(eval_args.dataset, mode="test")
    model = model.to(device)

    # if torch.__version__ >= "2.0":
    #     model = torch.compile(model)

    start_step = 0
    losses = [[], [], []]
    if resume > 0:
        start_step = model.load(log_dir)

        train_losses = torch.load(os.path.join(log_dir, "train_losses.pth")).tolist()
        eval_losses = torch.load(os.path.join(log_dir, "eval_losses.pth")).tolist()
        eval_psnrs = torch.load(os.path.join(log_dir, "eval_psnrs.pth")).tolist()
        losses = [train_losses, eval_losses, eval_psnrs]

        print("!!!!! Resume from step %s" % start_step)
    elif args.load_path:
        try:
            model_state_dict = torch.load(args.load_path)
            for step, state_dict in model_state_dict.items():
                resume_step = int(step)
                model.load_my_state_dict(state_dict)
        except:
            model_state_dict = torch.load(os.path.join(args.save_dir, args.load_path, "model.pth"))
            for step, state_dict in model_state_dict.items():
                resume_step = step
                model.load_my_state_dict(state_dict)
        print("!!!!! Loaded model from %s at step %s" % (args.load_path, resume_step))

    train_and_eval(start_step, model, device, dataset, eval_dataset, sample_dataset, losses, args)
    print(torch.cuda.memory_summary())


if __name__ == '__main__':

    with open("configs/default.yml", 'r') as f:
        default_config = yaml.safe_load(f)

    args = parse_args()
    with open(args.opt, 'r') as f:
        config = yaml.safe_load(f)
    
    train_config = copy.deepcopy(default_config)
    update_dict(train_config, config)

    sample_config = copy.deepcopy(train_config)
    sample_config['dataset']['patches']['height'] = train_config['exposure_control']['shading_code_resample_size']
    sample_config['dataset']['patches']['width'] = train_config['exposure_control']['shading_code_resample_size']
    sample_config = DictAsMember(sample_config)

    eval_config = copy.deepcopy(train_config)
    eval_config['dataset'].update(eval_config['eval']['dataset'])
    eval_config = DictAsMember(eval_config)
    train_config = DictAsMember(train_config)

    assert train_config.models.use_renderer, "Currently only support using renderer for exposure control"

    log_dir = os.path.join(train_config.save_dir, train_config.index)
    os.makedirs(log_dir, exist_ok=True)

    sys.stdout = Logger(os.path.join(log_dir, 'train.log'), sys.stdout)
    sys.stderr = Logger(os.path.join(log_dir, 'train_error.log'), sys.stderr)

    shutil.copyfile(__file__, os.path.join(log_dir, os.path.basename(__file__)))
    shutil.copyfile(args.opt, os.path.join(log_dir, os.path.basename(args.opt)))

    find_all_python_files_and_zip(".", os.path.join(log_dir, "code.zip"))

    setup_seed(train_config.seed)

    main(train_config, eval_config, sample_config, args.resume)