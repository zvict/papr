import numpy as np
import torch
import math
from .load_t2 import load_t2_data
from .load_nerfsyn import load_blender_data
from .load_dtu_colmap_coord import load_dtu_colmap_coord_data
from .load_dtu_colmap_coord_with_train import load_dtu_colmap_coord_data_with_train


def cam_to_world(coords, c2w, vector=True):
    """
        coords: [N, H, W, 3] or [H, W, 3] or [K, 3]
        c2w: [N, 4, 4] or [4, 4]
    """
    if vector:  # Convert to homogeneous coordinates
        coords = torch.cat([coords, torch.zeros_like(coords[..., :1])], -1)
    else:
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], -1)

    if coords.ndim == 5:
        assert c2w.ndim == 2
        B, H, W, N, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 1, 1, 1, 4, 4), -1)    # [B, H, W, N, 3]
    elif coords.ndim == 4:
        assert c2w.ndim == 3
        _, H, W, _ = coords.shape
        N = c2w.shape[0]
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(N, 1, 1, 4, 4), -1)  # [N, H, W, 4]
    elif coords.ndim == 3:
        assert c2w.ndim == 2
        H, W, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 1, 4, 4), -1)    # [H, W, 4]
    elif coords.ndim == 2:
        assert c2w.ndim == 2
        K, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 4, 4), -1)   # [K, 4]
    else:
        raise ValueError('Wrong dimension of coords')
    return transformed_coords[..., :3]


def world_to_cam(coords, c2w, vector=True):
    """
        coords: [N, H, W, 3] or [H, W, 3] or [K, 3]
        c2w: [N, 4, 4] or [4, 4]
    """
    if vector:  # Convert to homogeneous coordinates
        coords = torch.cat([coords, torch.zeros_like(coords[..., :1])], -1)
    else:
        coords = torch.cat([coords, torch.ones_like(coords[..., :1])], -1)

    c2w = torch.inverse(c2w)
    if coords.ndim == 5:
        assert c2w.ndim == 2
        B, H, W, N, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 1, 1, 1, 4, 4), -1)    # [B, H, W, N, 3]
    elif coords.ndim == 4:
        assert c2w.ndim == 3
        _, H, W, _ = coords.shape
        N = c2w.shape[0]
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(N, 1, 1, 4, 4), -1)  # [N, H, W, 4]
    elif coords.ndim == 3:
        assert c2w.ndim == 2
        H, W, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 1, 4, 4), -1)    # [H, W, 4]
    elif coords.ndim == 2:
        assert c2w.ndim == 2
        K, _ = coords.shape
        transformed_coords = torch.sum(
            coords.unsqueeze(-2) * c2w.reshape(1, 4, 4), -1)   # [K, 4]
    else:
        raise ValueError('Wrong dimension of coords')
    return transformed_coords[..., :3]


def get_rays(H, W, focal_x, focal_y, c2w, fineness=1):
    N = c2w.shape[0]
    width = torch.linspace(
        0, W / focal_x, steps=int(W / fineness) + 1, dtype=torch.float32)
    height = torch.linspace(
        0, H / focal_y, steps=int(H / fineness) + 1, dtype=torch.float32)
    y, x = torch.meshgrid(height, width, indexing='ij')
    pixel_size_x = width[1] - width[0]
    pixel_size_y = height[1] - height[0]
    x = (x - W / focal_x / 2 + pixel_size_x / 2)[:-1, :-1]
    y = -(y - H / focal_y / 2 + pixel_size_y / 2)[:-1, :-1]
    # [H, W, 3], vectors, since the camera is at the origin
    dirs_d = torch.stack([x, y, -torch.ones_like(x)], -1)
    rays_d = cam_to_world(dirs_d.unsqueeze(0), c2w)  # [N, H, W, 3]
    rays_o = c2w[:, :3, -1]       # [N, 3]
    return rays_o, rays_d / torch.norm(rays_d, dim=-1, keepdim=True)


def extract_patches(imgs, rays_o, rays_d, args):
    patch_opt = args.patches
    N, H, W, C = imgs.shape

    num_patches = patch_opt.max_patches
    rayd_patches = np.zeros((N, num_patches, patch_opt.height, patch_opt.width, 3), dtype=np.float32)
    rayo_patches = np.zeros((N, num_patches, 3), dtype=np.float32)
    img_patches = np.zeros((N, num_patches, patch_opt.height, patch_opt.width, C), dtype=np.float32)

    for i in range(N):
        for n_patch in range(num_patches):
            start_height = np.random.randint(0, H - patch_opt.height)
            start_width = np.random.randint(0, W - patch_opt.width)
            end_height = start_height + patch_opt.height
            end_width = start_width + patch_opt.width
            rayd_patches[i, n_patch, :, :] = rays_d[i, start_height:end_height, start_width:end_width]
            rayo_patches[i, n_patch, :] = rays_o[i, :]
            img_patches[i, n_patch, :, :] = imgs[i, start_height:end_height, start_width:end_width]

    return img_patches, rayd_patches, rayo_patches, num_patches


def load_meta_data(args, mode="train"):
    """
    0 -----------> W
    |
    |
    |
    ⬇
    H
    [H, W, 4]
    """
    image_paths = None

    if args.type == "synthetic":
        images, poses, hwf, image_paths = load_blender_data(
            args.path, split=mode, factor=args.factor, read_offline=args.read_offline)
        print('Loaded blender', images.shape, hwf, args.path)

        H, W, focal = hwf
        hwf = [H, W, focal, focal]

        if args.white_bg:
            images = images[..., :3] * \
                images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    elif args.type == "t2":
        images, poses, hwf, image_paths = load_t2_data(
            args.path, factor=args.factor, split=mode, read_offline=args.read_offline)
        print('Loaded t2', images.shape, hwf, args.path,
              images.min(), images.max(), images[0, 10, 10, :])

        if args.white_bg and images.shape[-1] == 4:
            images = images[..., :3] * \
                images[..., -1:] + (1. - images[..., -1:])
        elif not args.white_bg:
            images = images[..., :3]
            mask = images.sum(-1) == 3.0
            images[mask] = 0.

    elif args.type == "dtu_colmap_coord":
        images, poses, hwf, image_paths = load_dtu_colmap_coord_data(
            args.path, split=mode, factor=args.factor, read_offline=args.read_offline)
        print('Loaded DTU', images.shape, hwf, args.path)
        
        H, W, focal_x, focal_y = hwf
        hwf = [H, W, focal_x, focal_y]
        
        masks = images[..., -1:]
         
        if args.white_bg:
            images = images[..., :3] * \
                images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]
    
    elif args.type == "dtu_colmap_coord_with_train":
        images, poses, hwf, image_paths = load_dtu_colmap_coord_data_with_train(
            args.path, split=mode, factor=args.factor, read_offline=args.read_offline)
        print('Loaded DTU', images.shape, hwf, args.path)
        
        H, W, focal_x, focal_y = hwf
        hwf = [H, W, focal_x, focal_y]
        
        masks = images[..., -1:]
         
        if args.white_bg:
            images = images[..., :3] * \
                images[..., -1:] + (1. - images[..., -1:])
        else:
            images = images[..., :3]

    else:
        raise ValueError("Unknown dataset type: {}".format(args.type))

    H, W, focal_x, focal_y = hwf

    images = torch.from_numpy(images).float()
    poses = torch.from_numpy(poses).float()

    return images, poses, H, W, focal_x, focal_y, image_paths


def rgb2norm(img):
    norm_vec = np.stack([img[..., 0] * 2.0 / 255.0 - 1.0,
                         img[..., 1] * 2.0 / 255.0 - 1.0,
                         img[..., 2] * 2.0 / 255.0 - 1.0,
                         img[..., 3] / 255.0], axis=-1)
    return norm_vec
