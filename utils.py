import sys
from datetime import datetime
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import os
import zipfile
import torch
import random
import copy


class DictAsMember(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = DictAsMember(value)
        return value


def update_dict(original, param):
    for key in param.keys():
        if type(param[key]) == dict:
            update_dict(original[key], param[key])
        elif type(param[key]) == list and key == "datasets":
            for i in range(len(param[key])):
                name = param[key][i]['name']
                for j in range(len(original[key])):
                    if original[key][j]['name'] == name:
                        for k in param[key][i].keys():
                            original[key][j][k] = param[key][i][k]
                        break
                else:
                    new_param = copy.deepcopy(original[key][0])
                    update_dict(new_param, param[key][i])
                    original[key].append(new_param)
        else:
            original[key] = param[key]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def find_all_python_files_and_zip(src_dir, dst_path):
    # find all python files in src_dir
    python_files = []
    for root, dirs, files in os.walk(src_dir):
        if 'experiment' in root:
            continue
        for cur_file in files:
            if cur_file.endswith('.py'):
                python_files.append(os.path.join(root, cur_file))

    # zip all python files
    with zipfile.ZipFile(dst_path, 'w') as zip_file:
        for cur_file in python_files:
            zip_file.write(cur_file, os.path.relpath(cur_file, src_dir))


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')
        ct = datetime.now()
        self.log.write('*'*50 + '\n' + str(ct) + '\n' + '*'*50 + '\n')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_colors(weights):
    N = weights.shape[0]
    weights = (weights - weights.min()) / (weights.max() - weights.min())
    colors = np.full((N, 3), [1., 0., 0.])
    colors[:, 0] *= weights[:N]
    colors[:, 2] = (1 - weights[:N])
    return colors


def get_training_main_plot(index, steps, train_tgt_rgb, train_tgt_patch, train_pred_patch, test_tgt_tgb, test_pred_rgb, train_losses,
                           eval_losses, points_np, pt_plot_scale, depth_np, pt_lrs, attn_lrs, eval_psnrs, points_conf_scores_np=None):
    step = steps[-1]
    fig = plt.figure(figsize=(20, 10))

    ax = fig.add_subplot(2, 5, 1)
    ax.imshow(train_tgt_rgb)
    ax.set_title(f'Iteration: {step} train norm')

    ax = fig.add_subplot(2, 5, 2)
    ax.imshow(train_tgt_patch)
    ax.set_title(f'Iteration: {step} train norm patch')

    ax = fig.add_subplot(2, 5, 3)
    ax.imshow(train_pred_patch)
    ax.set_title(f'Iteration: {step} train output')

    ax = fig.add_subplot(2, 5, 4)
    ax.plot(steps, train_losses, label='train')
    ax.plot(steps, eval_losses, label='eval')
    ax.legend()
    ax.set_title('losses')

    ax = fig.add_subplot(2, 5, 5, projection='3d')
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    cur_color = "grey"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color)
    ax.set_title('Point Cloud')

    ax = fig.add_subplot(2, 5, 6)
    cd = ax.imshow(depth_np)
    fig.colorbar(cd, ax=ax)
    ax.set_title(f'depth map')

    ax = fig.add_subplot(2, 5, 7)
    ax.imshow(test_tgt_tgb)
    ax.set_title(f'Iteration: {step} eval norm')

    ax = fig.add_subplot(2, 5, 8)
    ax.imshow(test_pred_rgb)
    ax.set_title(f'Iteration: {step} eval predict')

    ax = fig.add_subplot(2, 5, 9)
    ax.plot(steps, np.log10(pt_lrs), label="pt lr")
    ax.plot(steps, np.log10(attn_lrs), label="attn lr")
    ax.legend()
    ax.set_title('learning rates log10')

    ax = fig.add_subplot(2, 5, 10)
    ax.plot(steps, eval_psnrs)
    ax.set_title('eval psnr')

    fig.suptitle("Main Plot\n%s\niter %d\nnum pts: %d" % (index, step, points_np.shape[0]))

    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()

    return img


def get_training_pcd_plot(index, step, ro, rd, points_np, coord_scale, pt_plot_scale, points_conf_scores_np=None):
    num_plots = 6 if points_conf_scores_np is not None else 4
    fig = plt.figure(figsize=(5 * num_plots, 6))

    H, W, _ = rd.shape

    ax = fig.add_subplot(1, num_plots, 1, projection='3d')
    ax.view_init(elev=0., azim=90)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=0.8 * coord_scale)
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(ro[0], ro[1], ro[2], rd[H//2, W//2, 0], rd[H//2, W//2, 1], rd[H//2, W//2, 2], length=2, alpha=1, color="blue")
    ax.set_title('Point Cloud View 1')

    ax = fig.add_subplot(1, num_plots, 2, projection='3d')
    ax.view_init(elev=0., azim=180)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=0.8 * coord_scale)
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(ro[0], ro[1], ro[2], rd[H//2, W//2, 0], rd[H//2, W//2, 1], rd[H//2, W//2, 2], length=2, alpha=1, color="blue")
    ax.set_title('Point Cloud View 2')

    ax = fig.add_subplot(1, num_plots, 3, projection='3d')
    ax.view_init(elev=0., azim=270)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=0.8 * coord_scale)
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(ro[0], ro[1], ro[2], rd[H//2, W//2, 0], rd[H//2, W//2, 1], rd[H//2, W//2, 2], length=2, alpha=1, color="blue")
    ax.set_title('Point Cloud View 3')

    ax = fig.add_subplot(1, num_plots, 4, projection='3d')
    ax.view_init(elev=89.9, azim=90)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=0.8 * coord_scale)
    ax.scatter(ro[0], ro[1], ro[2], c="red", s=10)
    ax.quiver(ro[0], ro[1], ro[2], rd[H//2, W//2, 0], rd[H//2, W//2, 1], rd[H//2, W//2, 2], length=2, alpha=1, color="blue")
    ax.set_title('Point Cloud View 1 Up')

    if points_conf_scores_np is not None:
        ax = fig.add_subplot(1, num_plots, 5)
        ax.scatter(range(len(points_conf_scores_np)), points_conf_scores_np)
        ax.set_title('Confidence Scores scatter plot')

        ax = fig.add_subplot(1, num_plots, 6)
        bins = np.linspace(-1, 1, 100).tolist()
        ax.hist(points_conf_scores_np, bins=bins)
        ax.set_title('Confidence Scores histogram')

    fig.suptitle("Point Clouds\n%s\niter %d" % (index, step))
    
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()

    return img


def get_training_pcd_single_plot(step, points_np, pt_plot_scale, points_conf_scores_np=None):
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20., azim=90 + (step / 500) * (720. / 500))
    ax.set_xlim3d(-pt_plot_scale * 1.5, pt_plot_scale * 1.5)
    ax.set_ylim3d(-pt_plot_scale * 1.5, pt_plot_scale * 1.5)
    ax.set_zlim3d(-pt_plot_scale * 1.5, pt_plot_scale * 1.5)
    ax.set_axis_off()
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color)

    fig.suptitle("iter %d\n#points: %d" % (step, points_np.shape[0]))
    fig.tight_layout()

    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()
    
    return img


def get_test_pcrgb(frame, th, azmin, test_psnr, points_np, rgb_pred_np, rgb_gt_np, depth_np, pt_plot_scale, points_conf_scores_np=None):
    fig = plt.figure(figsize=(30, 10))

    ax = fig.add_subplot(1, 5, 1, projection='3d')
    ax.axis('off')
    ax.view_init(elev=20., azim=90 - th)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    cur_color = "grey"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color, s=0.5)

    ax = fig.add_subplot(1, 5, 2, projection='3d')
    # ax.axis('off')
    ax.view_init(elev=0., azim=azmin)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    switch_yz = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=np.float32)
    cur_points_np = points_np @ switch_yz
    cur_points_np[:, 2] = -cur_points_np[:, 2]
    cur_color = "orange"
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter3D(cur_points_np[:, 0], cur_points_np[:, 1], cur_points_np[:, 2], c=cur_color, s=0.5)

    ax = fig.add_subplot(1, 5, 3)
    ax.axis('off')
    ax.imshow(rgb_pred_np)

    ax = fig.add_subplot(1, 5, 4)
    ax.axis('off')
    ax.imshow(rgb_gt_np)

    ax = fig.add_subplot(1, 5, 5)
    ax.axis('off')
    ax.imshow(depth_np)

    fig.suptitle("Point Cloud and RGB\nframe %d, PSNR %.3f, num points %d" % (frame, test_psnr, points_np.shape[0]))

    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()

    return img


def get_test_featmap_attn(frame, th, points_np, rgb_pred_np, rgb_gt_np, pt_plot_scale, featmap_np, attn_np, points_conf_scores_np=None):
    fig = plt.figure(figsize=(20, 15))

    ax = fig.add_subplot(3, 5, 1, projection='3d')
    ax.axis('off')
    ax.view_init(elev=20., azim=90 - th)
    ax.set_xlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_ylim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_zlim3d(-pt_plot_scale, pt_plot_scale)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if points_conf_scores_np is not None:
        cur_color = get_colors(points_conf_scores_np)
    ax.scatter(points_np[:, 0], points_np[:, 1], points_np[:, 2], c=cur_color)

    ax = fig.add_subplot(3, 5, 2)
    ax.axis('off')
    ax.imshow(rgb_gt_np)

    ax = fig.add_subplot(3, 5, 3)
    ax.axis('off')
    ax.imshow(rgb_pred_np)

    C = featmap_np.shape[-1]
    for i in range(min(5, C)):
        ax = fig.add_subplot(3, 5, 6 + i)
        ax.axis('off')
        cur_dim = C * i // 5
        cur_min = featmap_np[..., cur_dim].min()
        cur_max = featmap_np[..., cur_dim].max()
        ax.imshow(featmap_np[..., cur_dim])
        ax.set_title(f'featmap dim {cur_dim}\nmin: %.3f, max: %.3f' % (cur_min, cur_max))

    K = attn_np.shape[-1]
    for i in range(min(K-1, 4)):
        ax = fig.add_subplot(3, 5, 11 + i)
        ax.axis('off')
        cur_dim = K * i // 4
        cur_min = attn_np[..., cur_dim].min()
        cur_max = attn_np[..., cur_dim].max()
        ax.imshow(attn_np[..., cur_dim])
        ax.set_title(f'attn dim {cur_dim}\nmin: %.3f, max: %.3f' % (cur_min, cur_max))

    ax = fig.add_subplot(3, 5, 15)
    ax.axis('off')
    cur_min = attn_np[..., -1].min()
    cur_max = attn_np[..., -1].max()
    ax.imshow(attn_np[..., -1])
    ax.set_title(f'attn dim -1\nmin: %.3f, max: %.3f' % (cur_min, cur_max))

    fig.suptitle("feature map and attention\nframe %d\n" % (frame))
    
    canvas = fig.canvas
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    data = buffer.getvalue()
    buffer.write(data)
    img = Image.open(buffer)
    plt.close()

    return img


def resample_shading_codes(shading_codes, args, model, dataset, img_id, loss_fn, step, full_img=False):
    if full_img == True:
        img, rayd, rayo = dataset.get_full_img(img_id)
        c2w = dataset.get_c2w(img_id)
    else:
        _, _, img, rayd, rayo = dataset[img_id]
        c2w = dataset.get_c2w(img_id)
        img = torch.from_numpy(img).unsqueeze(0)
        rayd = torch.from_numpy(rayd).unsqueeze(0)
        rayo = torch.from_numpy(rayo).unsqueeze(0)

    sampled_shading_codes = torch.randn(args.exposure_control.shading_code_num_samples, 
                                        args.exposure_control.shading_code_dim, device=model.device) \
                                            * args.exposure_control.shading_code_scale
    
    N, H, W, _ = rayd.shape
    num_pts, _ = model.points.shape

    rayo = rayo.to(model.device)
    rayd = rayd.to(model.device)
    img = img.to(model.device)
    c2w = c2w.to(model.device)
    
    topk = min([num_pts, model.select_k])

    bkg_seq_len_attn = 0
    feat_dim = args.models.attn.embed.value.d_ff_out
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
        
        for i in range(args.exposure_control.shading_code_num_samples):
            torch.cuda.empty_cache()
            cur_shading_code = sampled_shading_codes[i]
            cur_affine = model.mapping_mlp(cur_shading_code)
            cur_affine_dim = cur_affine.shape[-1]
            cur_gamma, cur_beta = cur_affine[:cur_affine_dim // 2], cur_affine[cur_affine_dim // 2:]
            # print(cur_shading_code.min().item(), cur_shading_code.max().item(), cur_gamma.min().item(), cur_gamma.max().item(), cur_beta.min().item(), cur_beta.max().item())
            
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
    best_idx = best_loss_idx if args.exposure_control.shading_code_resample_select_by == "loss" else best_psnr_idx
    shading_codes[img_id] = sampled_shading_codes[best_idx]

    del rayo, rayd, img, c2w, attn
    del eval_loss
