import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np
from .utils import normalize_vector, create_learning_rate_fn, add_points_knn, activation_func
from .mlp import get_mapping_mlp
from .tx import get_transformer
from .renderer import get_generator


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class PAPR(nn.Module):
    def __init__(self, args, device='cuda'):
        super(PAPR, self).__init__()
        self.args = args
        self.eps = args.eps
        self.device = device

        self.use_amp = args.use_amp
        self.amp_dtype = torch.float16 if args.amp_dtype == 'float16' else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        point_opt = args.geoms.points
        pc_feat_opt = args.geoms.point_feats
        bkg_feat_opt = args.geoms.background

        self.register_buffer('select_k', torch.tensor(
            point_opt.select_k, device=device, dtype=torch.int32))

        self.coord_scale = args.dataset.coord_scale

        if point_opt.load_path:
            if point_opt.load_path.endswith('.pth') or point_opt.load_path.endswith('.pt'):
                points = torch.load(point_opt.load_path, map_location='cpu')
                points = np.asarray(points).astype(np.float32)
                np.random.shuffle(points)
                points = points[:args.max_num_pts, :]
                points = torch.from_numpy(points).float()
            print("Loaded points from {}, shape: {}, dtype {}".format(point_opt.load_path, points.shape, points.dtype))
            print("Loaded points scale: ", points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max(), points[:, 2].min(), points[:, 2].max())
        else:
            # Initialize point positions
            pt_init_center = [i * self.coord_scale for i in point_opt.init_center]
            pt_init_scale = [i * self.coord_scale for i in point_opt.init_scale]
            if point_opt.init_type == 'sphere': # initial points on a sphere
                points = self._sphere_pc(pt_init_center, point_opt.num, pt_init_scale)
            elif point_opt.init_type == 'cube': # initial points in a cube
                points = self._cube_normal_pc(pt_init_center, point_opt.num, pt_init_scale)
            elif point_opt.init_type == "norm-cube":
                points = self._cube_normal_pc(
                    pt_init_center, point_opt.num, pt_init_scale
                )
            else:
                raise NotImplementedError("Point init type [{:s}] is not found".format(point_opt.init_type))
            print("Initialized points scale: ", points[:, 0].min(), points[:, 0].max(), points[:, 1].min(), points[:, 1].max(), points[:, 2].min(), points[:, 2].max())
        self.points = torch.nn.Parameter(points, requires_grad=True)

        # Initialize point influence scores
        self.points_influ_scores = torch.nn.Parameter(torch.ones(
            points.shape[0], 1, device=device) * point_opt.influ_init_val, requires_grad=True)

        # Initialize mapping MLP, only if fine-tuning with IMLE for the exposure control
        self.mapping_mlp = None
        if args.models.mapping_mlp.use:
            self.mapping_mlp = get_mapping_mlp(
                args.models, use_amp=self.use_amp, amp_dtype=self.amp_dtype)

        # Initialize UNet
        if args.models.use_renderer:
            tx_opt = args.models.transformer
            feat_dim = tx_opt.embed.d_ff_out if tx_opt.embed.share_embed else tx_opt.embed.value.d_ff_out
            self.renderer = get_generator(args.models.renderer.generator, in_c=feat_dim,
                                          out_c=3, use_amp=self.use_amp, amp_dtype=self.amp_dtype)
            print("Number of parameters of renderer: ", count_parameters(self.renderer))
        else:
            assert (args.models.transformer.embed.share_embed and args.models.transformer.embed.d_ff_out == 3) or \
                (not args.models.transformer.embed.share_embed and args.models.transformer.embed.value.d_ff_out == 3), \
                "Value embedding MLP should have output dim 3 if not using renderer"

        # Initialize background score and features
        if bkg_feat_opt.init_type == 'random':
            bkg_feat_init_func = torch.rand
        elif bkg_feat_opt.init_type == 'zeros':
            bkg_feat_init_func = torch.zeros
        elif bkg_feat_opt.init_type == 'ones':
            bkg_feat_init_func = torch.ones
        else:
            raise NotImplementedError(
                "Background init type [{:s}] is not found".format(bkg_feat_opt.init_type))
        feat_dim = 3
        self.bkg_feats = nn.Parameter(bkg_feat_init_func(bkg_feat_opt.seq_len, feat_dim, device=device) * bkg_feat_opt.init_scale, requires_grad=bkg_feat_opt.learnable)
        self.bkg_score = torch.tensor(bkg_feat_opt.constant, device=device, dtype=torch.float32).reshape(1)

        # Initialize point features
        self.use_pc_feats = pc_feat_opt.use_ink or pc_feat_opt.use_inq or pc_feat_opt.use_inv
        if self.use_pc_feats:
            self.pc_feats = nn.Parameter(torch.randn(points.shape[0], pc_feat_opt.dim), requires_grad=True)
            print("Point features: ", self.pc_feats.shape, self.pc_feats.min(), self.pc_feats.max(), self.pc_feats.mean(), self.pc_feats.std())

        v_extra_dim = 0
        k_extra_dim = 0
        q_extra_dim = 0
        if pc_feat_opt.use_inv:
            v_extra_dim = self.pc_feats.shape[-1]
            print("Using v_extra_dim: ", v_extra_dim)
        if pc_feat_opt.use_ink:
            k_extra_dim = self.pc_feats.shape[-1]
            print("Using k_extra_dim: ", k_extra_dim)
        if pc_feat_opt.use_inq:
            q_extra_dim = self.pc_feats.shape[-1]
            print("Using q_extra_dim: ", q_extra_dim)

        self.last_act = activation_func(args.models.last_act)

        # Initialize proximity attention layer(s)
        transformer = get_transformer(args.models.transformer,
                                      seq_len=point_opt.num,
                                      v_extra_dim=v_extra_dim,
                                      k_extra_dim=k_extra_dim,
                                      q_extra_dim=q_extra_dim,
                                      eps=self.eps,
                                      use_amp=self.use_amp,
                                      amp_dtype=self.amp_dtype)
        self.transformer = transformer

        self.init_optimizers(total_steps=0)

    def init_optimizers(self, total_steps):
        lr_opt = self.args.training.lr
        print("LR factor: ", lr_opt.lr_factor)
        optimizer_points = torch.optim.Adam([self.points], lr=lr_opt.points.base_lr * lr_opt.lr_factor)
        optimizer_tx = torch.optim.Adam(self.transformer.parameters(), lr=lr_opt.transformer.base_lr * lr_opt.lr_factor, weight_decay=lr_opt.transformer.weight_decay)
        optimizer_points_influ_scores = torch.optim.Adam([self.points_influ_scores], lr=lr_opt.points_influ_scores.base_lr * lr_opt.lr_factor, weight_decay=lr_opt.points_influ_scores.weight_decay)

        debug = False
        lr_scheduler_points = create_learning_rate_fn(optimizer_points, self.args.training.steps, lr_opt.points, debug=debug)
        lr_scheduler_tx = create_learning_rate_fn(optimizer_tx, self.args.training.steps, lr_opt.transformer, debug=debug)
        lr_scheduler_points_influ_scores = create_learning_rate_fn(optimizer_points_influ_scores, self.args.training.steps, lr_opt.points_influ_scores, debug=debug)

        self.optimizers = {
            "points": optimizer_points,
            "transformer": optimizer_tx,
            "points_influ_scores": optimizer_points_influ_scores,
        }

        self.schedulers = {
            "points": lr_scheduler_points,
            "transformer": lr_scheduler_tx,
            "points_influ_scores": lr_scheduler_points_influ_scores,
        }

        if self.use_pc_feats:
            optimizer_pc_feats = torch.optim.Adam([self.pc_feats], lr=lr_opt.feats.base_lr * lr_opt.lr_factor, weight_decay=lr_opt.feats.weight_decay)
            lr_scheduler_pc_feats = create_learning_rate_fn(optimizer_pc_feats, self.args.training.steps, lr_opt.feats, debug=debug)

            self.optimizers["pc_feats"] = optimizer_pc_feats
            self.schedulers["pc_feats"] = lr_scheduler_pc_feats

        if self.mapping_mlp is not None:
            optimizer_mapping_mlp = torch.optim.Adam(self.mapping_mlp.parameters(), lr=lr_opt.mapping_mlp.base_lr * lr_opt.lr_factor, weight_decay=lr_opt.mapping_mlp.weight_decay)
            lr_scheduler_mapping_mlp = create_learning_rate_fn(optimizer_mapping_mlp, self.args.training.steps, lr_opt.mapping_mlp, debug=debug)

            self.optimizers["mapping_mlp"] = optimizer_mapping_mlp
            self.schedulers["mapping_mlp"] = lr_scheduler_mapping_mlp

        if self.args.models.use_renderer:
            optimizer_renderer = torch.optim.Adam(self.renderer.parameters(), lr=lr_opt.generator.base_lr * lr_opt.lr_factor, weight_decay=lr_opt.generator.weight_decay)
            lr_scheduler_renderer = create_learning_rate_fn(optimizer_renderer, self.args.training.steps, lr_opt.generator, debug=debug)

            self.optimizers["renderer"] = optimizer_renderer
            self.schedulers["renderer"] = lr_scheduler_renderer

        if self.bkg_feats is not None and self.args.geoms.background.learnable:
            optimizer_bkg_feats = torch.optim.Adam([self.bkg_feats], lr=lr_opt.bkg_feats.base_lr * lr_opt.lr_factor, weight_decay=lr_opt.bkg_feats.weight_decay)
            lr_scheduler_bkg_feats = create_learning_rate_fn(optimizer_bkg_feats, self.args.training.steps, lr_opt.bkg_feats, debug=debug)

            self.optimizers["bkg_feats"] = optimizer_bkg_feats
            self.schedulers["bkg_feats"] = lr_scheduler_bkg_feats

        for name in self.args.training.fix_keys:
            if name in self.optimizers:
                print("Fixing {}".format(name))
                self.optimizers.pop(name)
                self.schedulers.pop(name)

        if total_steps > 0:
            for _, scheduler in self.schedulers.items():
                if scheduler is not None:
                    for _ in range(total_steps):
                        scheduler.step()

    def clear_optimizer(self):
        self.optimizers.clear()
        del self.optimizers

    def clear_scheduler(self):
        self.schedulers.clear()
        del self.schedulers

    def clear_grad(self):
        for _, optimizer in self.optimizers.items():
            if optimizer is not None:
                optimizer.zero_grad()

    def _sphere_pc(self, center, num_pts, scale):
        xs, ys, zs = [], [], []
        phi = math.pi * (3. - math.sqrt(5.))
        for i in range(num_pts):
            y = 1 - (i / float(num_pts - 1)) * 2
            radius = math.sqrt(1 - y * y)
            theta = phi * i
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            xs.append(x * scale[0] + center[0])
            ys.append(y * scale[1] + center[1])
            zs.append(z * scale[2] + center[2])
        points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
        return torch.from_numpy(points).float()

    def _semi_sphere_pc(self, center, num_pts, scale, flatten="-z", flatten_coord=0.0):
        xs, ys, zs = [], [], []
        phi = math.pi * (3. - math.sqrt(5.))
        for i in range(num_pts):
            y = 1 - (i / float(num_pts - 1)) * 2
            radius = math.sqrt(1 - y * y)
            theta = phi * i
            x = math.cos(theta) * radius
            z = math.sin(theta) * radius
            xs.append(x * scale[0] + center[0])
            ys.append(y * scale[1] + center[1])
            zs.append(z * scale[2] + center[2])
        points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
        points = torch.from_numpy(points).float()
        if flatten == "-z":
            points[:, 2] = torch.clamp(points[:, 2], min=flatten_coord)
        elif flatten == "+z":
            points[:, 2] = torch.clamp(points[:, 2], max=flatten_coord)
        elif flatten == "-y":
            points[:, 1] = torch.clamp(points[:, 1], min=flatten_coord)
        elif flatten == "+y":
            points[:, 1] = torch.clamp(points[:, 1], max=flatten_coord)
        elif flatten == "-x":
            points[:, 0] = torch.clamp(points[:, 0], min=flatten_coord)
        elif flatten == "+x":
            points[:, 0] = torch.clamp(points[:, 0], max=flatten_coord)
        else:
            raise ValueError("Invalid flatten type")
        return points

    def _cube_pc(self, center, num_pts, scale):
        xs = np.random.uniform(-scale[0], scale[0], num_pts) + center[0]
        ys = np.random.uniform(-scale[1], scale[1], num_pts) + center[1]
        zs = np.random.uniform(-scale[2], scale[2], num_pts) + center[2]
        points = np.stack([np.array(xs), np.array(ys), np.array(zs)], axis=-1)
        return torch.from_numpy(points).float()

    def _cube_normal_pc(self, center, num_pts, scale):
        axis_num_pts = int(num_pts ** (1.0 / 3.0))
        xs = np.linspace(-scale[0], scale[0], axis_num_pts) + center[0]
        ys = np.linspace(-scale[1], scale[1], axis_num_pts) + center[1]
        zs = np.linspace(-scale[2], scale[2], axis_num_pts) + center[2]
        points = np.array([[i, j, k] for i in xs for j in ys for k in zs])
        rest_num_pts = num_pts - points.shape[0]
        if rest_num_pts > 0:
            rest_points = self._cube_pc(center, rest_num_pts, scale)
            points = np.concatenate([points, rest_points], axis=0)
        return torch.from_numpy(points).float()

    def _calculate_global_distances(self, rays_o, rays_d, points):
        """
        Select the top-k points with the smallest distance to the rays from all points

        Args:
            rays_o: (N, 3)
            rays_d: (N, H, W, 3)
            points: (num_pts, 3)
        Returns:
            select_k_ind: (N, H, W, select_k)
        """
        N, H, W, _ = rays_d.shape
        num_pts, _ = points.shape

        rays_d = rays_d.unsqueeze(-2)  # (N, H, W, 1, 3)
        rays_o = rays_o.reshape(N, 1, 1, 1, 3)
        points = points.reshape(1, 1, 1, num_pts, 3)

        v = points - rays_o    # (N, 1, 1, num_pts, 3)
        proj = rays_d * (torch.sum(v * rays_d, dim=-1) / (torch.sum(rays_d * rays_d, dim=-1) + self.eps)).unsqueeze(-1)
        D = v - proj    # (N, H, W, num_pts, 3)
        feature = torch.norm(D, dim=-1)

        _, select_k_ind = feature.topk(self.select_k, dim=-1, largest=False, sorted=self.args.geoms.points.select_k_sorted)  # (N, H, W, select_k)

        return select_k_ind

    def _calculate_distances(self, rays_o, rays_d, points, c2w):
        """
        Calculate the distances from top-k points to rays   TODO: redundant with _calculate_global_distances

        Args:
            rays_o: (N, 3)
            rays_d: (N, H, W, 3)
            points: (N, H, W, select_k, 3)
            c2w: (N, 4, 4)
        Returns:
            proj_dists: (N, H, W, select_k, 1)
            dists_to_rays: (N, H, W, select_k, 1)
            proj: (N, H, W, select_k, 3)    # the vector s in Figure 2
            D: (N, H, W, select_k, 3)    # the vector t in Figure 2
        """
        N, H, W, _ = rays_d.shape

        rays = normalize_vector(rays_d, eps=self.eps).unsqueeze(-2)  # (N, H, W, 1, 3)
        v = points - rays_o.reshape(N, 1, 1, 1, 3)    # (N, 1, 1, num_pts, 3)
        proj = rays * (torch.sum(v * rays, dim=-1) / (torch.sum(rays * rays, dim=-1) + self.eps)).unsqueeze(-1)
        D = v - proj    # (N, H, W, num_pts, 3)

        dists_to_rays = torch.norm(D, dim=-1, keepdim=True)
        proj_dists = torch.norm(proj, dim=-1, keepdim=True)

        return proj_dists, dists_to_rays, proj, D

    def _get_points(self, rays_o, rays_d, c2w, step=-1):
        """
        Select the top-k points with the smallest distance to the rays

        Args:
            rays_o: (N, 3)
            rays_d: (N, H, W, 3)
            c2w: (N, 4, 4)
        Returns:
            selected_points: (N, H, W, select_k, 3)
            select_k_ind: (N, H, W, select_k)
        """
        points = self.points
        N, H, W, _ = rays_d.shape
        if self.select_k >= points.shape[0] or self.select_k < 0:
            select_k_ind = torch.arange(points.shape[0], device=points.device).expand(N, H, W, -1)
        else:
            select_k_ind = self._calculate_global_distances(rays_o, rays_d, points)   # (N, H, W, num_pts)
        selected_points = points[select_k_ind, :]  # (N, H, W, select_k, 3)
        self.selected_points = selected_points

        return selected_points, select_k_ind

    def prune_points(self, thresh):
        if self.points_influ_scores is not None:
            if self.args.training.prune_type == '<':
                mask = (self.points_influ_scores[:, 0] > thresh)
            elif self.args.training.prune_type == '>':
                mask = (self.points_influ_scores[:, 0] < thresh)
            print(
                "@@@@@@@@@  pruned {}/{}".format(torch.sum(mask == 0), mask.shape[0]))

            cur_requires_grad = self.points.requires_grad
            self.points = nn.Parameter(self.points[mask, :], requires_grad=cur_requires_grad)
            print("@@@@@@@@@ New points: ", self.points.shape)

            cur_requires_grad = self.points_influ_scores.requires_grad
            self.points_influ_scores = nn.Parameter(self.points_influ_scores[mask, :], requires_grad=cur_requires_grad)
            print("@@@@@@@@@ New points_influ_scores: ", self.points_influ_scores.shape)

            if self.use_pc_feats:
                cur_requires_grad = self.pc_feats.requires_grad
                self.pc_feats = nn.Parameter(self.pc_feats[mask, :], requires_grad=cur_requires_grad)
                print("@@@@@@@@@ New pc_feats: ", self.pc_feats.shape)

            return torch.sum(mask == 0)
        return 0

    def add_points(self, add_num):
        points = self.points.detach().cpu()
        point_features = None
        cur_num_points = points.shape[0]

        if 'max_points' in self.args and self.args.max_points > 0 and (cur_num_points + add_num) >= self.args.max_points:
            add_num = self.args.max_points - cur_num_points
            if add_num <= 0:
                return 0

        if self.use_pc_feats:
            point_features = self.pc_feats.detach().cpu()

        new_points, num_new_points, new_influ_scores, new_point_features = add_points_knn(points, self.points_influ_scores.detach().cpu(), add_num=add_num,
                                                                                            k=self.args.geoms.points.add_k, comb_type=self.args.geoms.points.add_type,
                                                                                            sample_k=self.args.geoms.points.add_sample_k, sample_type=self.args.geoms.points.add_sample_type,
                                                                                            point_features=point_features)
        print("@@@@@@@@@  added {} points".format(num_new_points))

        if num_new_points > 0:
            cur_requires_grad = self.points.requires_grad
            self.points = nn.Parameter(torch.cat([points, new_points], dim=0).to(self.points.device), requires_grad=cur_requires_grad)
            print("@@@@@@@@@ New points: ", self.points.shape)

            if self.points_influ_scores is not None:
                cur_requires_grad = self.points_influ_scores.requires_grad
                self.points_influ_scores = nn.Parameter(torch.cat([self.points_influ_scores, new_influ_scores.to(self.points_influ_scores.device)], dim=0), requires_grad=cur_requires_grad)
                print("@@@@@@@@@ New points_influ_scores: ", self.points_influ_scores.shape)

            if self.use_pc_feats:
                cur_requires_grad = self.pc_feats.requires_grad
                self.pc_feats = nn.Parameter(torch.cat([self.pc_feats, new_point_features.to(self.pc_feats.device)], dim=0), requires_grad=cur_requires_grad)
                print("@@@@@@@@@ New pc_feats: ", self.pc_feats.shape)

        return num_new_points

    def _get_kqv(self, rays_o, rays_d, points, c2w, select_k_ind, step=-1):
        """
        Get the key, query, value for the proximity attention layer(s)
        """
        pd, d2r, vec_p2o, vec_p2r = self._calculate_distances(rays_o, rays_d, points, c2w)

        k_type = self.args.models.transformer.k_type
        k_L = self.args.models.transformer.embed.k_L
        if k_type == 1:
            key = [points.detach(), vec_p2o, vec_p2r]
        elif k_type == 2:
            key = [d2r]
        else:
            raise ValueError('Invalid key type')
        assert len(key) == (len(k_L))

        q_type = self.args.models.transformer.q_type
        q_L = self.args.models.transformer.embed.q_L
        if q_type == 1:
            query = [rays_d.unsqueeze(-2)]
        elif q_type == 2:
            query = [torch.ones_like(rays_d).unsqueeze(-2)]
        else:
            raise ValueError('Invalid query type')
        assert len(query) == (len(q_L))

        v_type = self.args.models.transformer.v_type
        v_L = self.args.models.transformer.embed.v_L
        if v_type == 1:
            value = [vec_p2o, vec_p2r]
        elif v_type == 2:
            value = [pd, d2r]
        else:
            raise ValueError('Invalid value type')
        assert len(value) == (len(v_L))

        # Add extra features that won't be passed through positional encoding
        k_extra = None
        q_extra = None
        v_extra = None
        if self.args.geoms.point_feats.use_ink:
            k_extra = [self.pc_feats[select_k_ind, :]]
        if self.args.geoms.point_feats.use_inq:
            q_extra = [self.pc_feats[select_k_ind, :]]
        if self.args.geoms.point_feats.use_inv:
            v_extra = [self.pc_feats[select_k_ind, :]]

        return key, query, value, k_extra, q_extra, v_extra

    def step(self, step=-1):
        for _, optimizer in self.optimizers.items():
            if optimizer is not None:
                self.scaler.step(optimizer)

        for _, scheduler in self.schedulers.items():
            if scheduler is not None:
                scheduler.step()

        self.tx_lr = 0
        if 'transformer' in self.optimizers:
            if self.schedulers['transformer'] is not None:
                self.tx_lr = self.schedulers['transformer'].get_last_lr()[0]
            else:
                self.tx_lr = self.optimizers['transformer'].param_groups[0]['lr']

        self.pts_lr = 0
        if 'points' in self.optimizers:
            if self.schedulers['points'] is not None:
                self.pts_lr = self.schedulers['points'].get_last_lr()[0]
            else:
                self.pts_lr = self.optimizers['points'].param_groups[0]['lr']

    def evaluate(self, rays_o, rays_d, c2w, step=-1, shading_code=None):
        points, select_k_ind = self._get_points(rays_o, rays_d, c2w, step)
        self.select_k_ind = select_k_ind
        key, query, value, k_extra, q_extra, v_extra = self._get_kqv(rays_o, rays_d, points, c2w, select_k_ind, step)
        N, H, W, _ = rays_d.shape
        num_pts = points.shape[-2]

        cur_points_influ_score = self.points_influ_scores[select_k_ind] if self.points_influ_scores is not None else None

        _, _, embedv, _, scores = self.transformer(key, query, value, k_extra, q_extra, v_extra, step=step)

        embedv = embedv.reshape(N, H, W, -1, embedv.shape[-1])
        scores = scores.reshape(N, H, W, -1, 1)

        if cur_points_influ_score is not None:
            scores = scores * cur_points_influ_score
        if self.bkg_feats is not None:
            bkg_seq_len = self.bkg_feats.shape[0]
            scores = torch.cat([scores, self.bkg_score.expand(N, H, W, bkg_seq_len, -1)], dim=-2)
            attn = F.softmax(scores, dim=3) # (N, H, W, num_pts+bkg_seq_len, 1)
            topk_attn = attn[..., :num_pts, :]
            if self.args.models.normalize_topk_attn:
                topk_attn = topk_attn / torch.sum(topk_attn, dim=3, keepdim=True)
            fused_features = torch.sum(embedv * topk_attn, dim=3, keepdim=True)   # (N, H, W, 1, C)
        else:
            attn = F.softmax(scores, dim=3)
            if self.args.models.normalize_topk_attn:
                attn = attn / torch.sum(attn, dim=3, keepdim=True)
            fused_features = torch.sum(embedv * attn, dim=3, keepdim=True)   # (N, H, W, 1, C)

        return fused_features, attn

    def forward(self, rays_o, rays_d, c2w, step=-1, shading_code=None):
        gamma, beta = None, None
        if shading_code is not None and self.mapping_mlp is not None:
            affine = self.mapping_mlp(shading_code)
            affine_dim = affine.shape[-1]
            gamma, beta = affine[:affine_dim//2], affine[affine_dim//2:]

        points, select_k_ind = self._get_points(rays_o, rays_d, c2w, step)
        key, query, value, k_extra, q_extra, v_extra = self._get_kqv(rays_o, rays_d, points, c2w, select_k_ind, step)
        N, H, W, _ = rays_d.shape
        num_pts = points.shape[-2]

        cur_points_influ_scores = self.points_influ_scores[select_k_ind] if self.points_influ_scores is not None else None

        _, _, embedv, encode, scores = self.transformer(key, query, value, k_extra, q_extra, v_extra, step=step)

        if step >= 0 and step % 200 == 0:
            print(' encode:', step, encode.shape, encode.min().item(),
                  encode.max().item(), encode.mean().item(), encode.std().item())
            print(' embedv:', step, embedv.shape, embedv.min().item(),
                  embedv.max().item(), embedv.mean().item(), embedv.std().item())
            print(' scores:', step, scores.shape, scores.min().item(),
                  scores.max().item(), scores.mean().item(), scores.std().item())

        embedv = embedv.reshape(N, H, W, -1, embedv.shape[-1])
        scores = scores.reshape(N, H, W, -1, 1)

        if cur_points_influ_scores is not None:
            # Multiply the influence scores to the attention scores
            scores = scores * cur_points_influ_scores

        if self.bkg_feats is not None:
            bkg_seq_len = self.bkg_feats.shape[0]
            scores = torch.cat([scores, self.bkg_score.expand(N, H, W, bkg_seq_len, -1)], dim=-2)
            attn = F.softmax(scores, dim=3) # (N, H, W, num_pts+bkg_seq_len, 1)
            topk_attn = attn[..., :num_pts, :]
            bkg_attn = attn[..., num_pts:, :]
            if self.args.models.normalize_topk_attn:
                topk_attn = topk_attn / torch.sum(topk_attn, dim=3, keepdim=True)
            fused_features = torch.sum(embedv * topk_attn, dim=3)   # (N, H, W, C)

            if self.args.models.use_renderer:
                foreground = self.renderer(fused_features.permute(0, 3, 1, 2), gamma=gamma, beta=beta).permute(0, 2, 3, 1).unsqueeze(-2)   # (N, H, W, 1, 3)
            else:
                foreground = fused_features.unsqueeze(-2)

            if self.args.models.normalize_topk_attn:
                rgb = foreground * (1 - bkg_attn) + self.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
            else:
                rgb = foreground + self.bkg_feats.expand(N, H, W, -1, -1) * bkg_attn
            rgb = rgb.squeeze(-2)
        else:
            attn = F.softmax(scores, dim=3)
            fused_features = torch.sum(embedv * attn, dim=3)   # (N, H, W, C)
            if self.args.models.use_renderer:
                rgb = self.renderer(fused_features.permute(0, 3, 1, 2), gamma=gamma, beta=beta).permute(0, 2, 3, 1)   # (N, H, W, 3)
            else:
                rgb = fused_features

        if step >= 0 and step % 1000 == 0:
            print(' feat map:', step, fused_features.shape, fused_features.min().item(),
                  fused_features.max().item(), fused_features.mean().item(), fused_features.std().item())
            print(' predict rgb:', step, rgb.shape, rgb.min().item(),
                  rgb.max().item(), rgb.mean().item(), rgb.std().item())

        return rgb

    def save(self, step, save_dir):
        torch.save({str(step): self.state_dict()},
                   os.path.join(save_dir, 'model.pth'))

        optimizers_state_dict = {}
        for name, optimizer in self.optimizers.items():
            if optimizer is not None:
                optimizers_state_dict[name] = optimizer.state_dict()
            else:
                optimizers_state_dict[name] = None
        torch.save(optimizers_state_dict, os.path.join(
            save_dir, 'optimizers.pth'))

        schedulers_state_dict = {}
        for name, scheduler in self.schedulers.items():
            if scheduler is not None:
                schedulers_state_dict[name] = scheduler.state_dict()
            else:
                schedulers_state_dict[name] = None
        torch.save(schedulers_state_dict, os.path.join(
            save_dir, 'schedulers.pth'))

        scaler_state_dict = self.scaler.state_dict()
        torch.save(scaler_state_dict, os.path.join(
            save_dir, 'scaler.pth'))

    def load(self, load_dir, load_optimizer=False):
        if load_optimizer == True:
            optimizers_state_dict = torch.load(
                os.path.join(load_dir, 'optimizers.pth'))
            for name, optimizer in self.optimizers.items():
                if optimizer is not None:
                    optimizer.load_state_dict(optimizers_state_dict[name])
                else:
                    assert optimizers_state_dict[name] is None

            schedulers_state_dict = torch.load(
                os.path.join(load_dir, 'schedulers.pth'))
            for name, scheduler in self.schedulers.items():
                if scheduler is not None:
                    scheduler.load_state_dict(schedulers_state_dict[name])
                else:
                    assert schedulers_state_dict[name] is None

        if os.path.exists(os.path.join(load_dir, 'scaler.pth')):
            scaler_state_dict = torch.load(
                os.path.join(load_dir, 'scaler.pth'))
            self.scaler.load_state_dict(scaler_state_dict)

        model_state_dict = torch.load(os.path.join(load_dir, 'model.pth'))
        for step, state_dict in model_state_dict.items():
            # self.load_state_dict(state_dict)
            self.load_my_state_dict(state_dict)
            return int(step)

    def load_my_state_dict(self, state_dict, exclude_keys=[]):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            print(name, param.shape)
            for exclude_key in exclude_keys:
                if exclude_key in name:
                    print("exclude", name)
                    break
            else:
                if name not in ["points", "points_influ_scores", "pc_feats"]:
                    if isinstance(param, nn.Parameter):
                        # backwards compatibility for serialized parameters
                        param = param.data
                    try:
                        own_state[name].copy_(param)
                    except:
                        print("Can't load", name)

        self.points = nn.Parameter(
            state_dict['points'].data, requires_grad=self.points.requires_grad)
        if self.points_influ_scores is not None:
            self.points_influ_scores = nn.Parameter(
                state_dict['points_influ_scores'].data, requires_grad=self.points_influ_scores.requires_grad)
        self.pc_feats = nn.Parameter(state_dict['pc_feats'].data, requires_grad=self.pc_feats.requires_grad)
        print("load pc_feats", self.pc_feats.shape, self.pc_feats.min(), self.pc_feats.max())
