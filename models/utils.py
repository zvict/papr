import torch
from torch import nn
import torch.optim.lr_scheduler as lr_scheduler
import scipy
from scipy.spatial import KDTree
import numpy as np


def add_points_knn(coords, influ_scores, add_num, k, comb_type="mean", sample_type="random", sample_k=10, point_features=None, 
                   last_coord_grad=None, acc_coord_grad=None, acc_coord_grad_norm=None, grad_cnt=None, hybrid_weight=0.5):
    """
    Add points to the point cloud by kNN
    """
    pc = KDTree(coords)
    N = coords.shape[0]

    # Step 1: Determine where to add points
    if N <= add_num and "random" in comb_type:
        inds = np.random.choice(N, add_num, replace=True)
        query_coords = coords[inds, :]
    elif N <= add_num:
        query_coords = coords
        inds = list(range(N))
    else:
        if sample_type == "random":
            inds = np.random.choice(N, add_num, replace=False)
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-std":
            assert k >= 2
            nns_dists, nns_inds = pc.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.std(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-mean":
            assert k >= 2
            nns_dists, nns_inds = pc.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.mean(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-max":
            assert k >= 2
            nns_dists, nns_inds = pc.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.max(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "top-knn-min":
            assert k >= 2
            nns_dists, nns_inds = pc.query(coords, k=sample_k)
            inds = np.argsort(nns_dists.min(axis=-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "influ-scores-max":
            inds = np.argsort(influ_scores.squeeze())[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "influ-scores-min":
            inds = np.argsort(influ_scores.squeeze())[:add_num]
            query_coords = coords[inds, :]
        elif sample_type == "influ-scores-max":
            inds = np.argsort(influ_scores.squeeze())[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "influ-scores-min":
            inds = np.argsort(influ_scores.squeeze())[:add_num]
            query_coords = coords[inds, :]
        elif sample_type == "last-coord-grad-max":
            inds = np.argsort(last_coord_grad.abs().sum(-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-max":
            inds = np.argsort(acc_coord_grad.abs().sum(-1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-cnt-max":
            inds = np.argsort(acc_coord_grad.abs().sum(-1) / (grad_cnt + 1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-norm-max":  # the togo now
            inds = np.argsort(acc_coord_grad_norm)[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-norm-cnt-max":
            inds = np.argsort(acc_coord_grad_norm / (grad_cnt + 1))[-add_num:]
            query_coords = coords[inds, :]
        elif sample_type == "acc-coord-grad-norm-max-hybrid-top-knn-std":
            inds_a = np.argsort(acc_coord_grad_norm)
            ranks_a = np.zeros_like(inds_a)
            ranks_a[inds_a] = np.arange(len(inds_a))
            
            assert k >= 2
            pc = KDTree(coords)
            nns_dists, nns_inds = pc.query(coords, k=sample_k+1)
            nns_dists = nns_dists[:, 1:]
            inds_b = np.argsort(nns_dists.std(axis=-1))
            ranks_b = np.zeros_like(inds_b)
            ranks_b[inds_b] = np.arange(len(inds_b))

            ranks = hybrid_weight * ranks_a + (1 - hybrid_weight) * ranks_b
            inds = np.argsort(ranks)[-add_num:]
            query_coords = coords[inds, :]
        else:
            raise NotImplementedError

    # Step 2: Add points by kNN
    new_features = None
    move_scale = 2.0
    if comb_type == "duplicate":
        noise = np.random.randn(3).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        noise *= k
        new_coords = (query_coords + noise)
        new_influ_scores = influ_scores[inds, :]
        if point_features is not None:
            new_features = point_features[inds, :]
    elif comb_type == "clone":
        new_coords = query_coords
        new_influ_scores = influ_scores[inds, :]
        if point_features is not None:
            new_features = point_features[inds, :]
    else:
        pc = KDTree(coords)
        nns_dists, nns_inds = pc.query(query_coords, k=k+1)
        nns_dists = nns_dists.astype(np.float32)
        nns_dists = nns_dists[:, 1:]
        nns_inds = nns_inds[:, 1:]
        if comb_type == "mean":
            new_coords = coords[nns_inds, :].mean(
                axis=-2)  # (Nq, k, 3) -> (Nq, 3)
            new_influ_scores = influ_scores[nns_inds, :].mean(axis=-2)
            if point_features is not None:
                new_features = point_features[nns_inds, :].mean(axis=-2)
        elif comb_type == "random":
            rnd_w = np.random.uniform(0, 1, (query_coords.shape[0], k)).astype(np.float32)
            rnd_w /= rnd_w.sum(axis=-1, keepdims=True)
            new_coords = (coords[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            new_influ_scores = (influ_scores[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            if point_features is not None:
                new_features = (point_features[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
        elif comb_type == "random-softmax":
            rnd_w = np.random.randn(query_coords.shape[0], k).astype(np.float32)
            rnd_w = scipy.special.softmax(rnd_w, axis=-1)
            new_coords = (coords[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            new_influ_scores = (influ_scores[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
            if point_features is not None:
                new_features = (point_features[nns_inds, :] * rnd_w.reshape(-1, k, 1)).sum(axis=-2)
        elif comb_type == "weighted":
            new_coords = (coords[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            new_influ_scores = (influ_scores[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_features is not None:
                new_features = (point_features[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
        elif comb_type == "along-last-coord-grad":
            new_coords = query_coords + last_coord_grad[inds, :] * move_scale
            nns_dists, nns_inds = pc.query(new_coords, k=k)
            nns_dists = nns_dists.astype(np.float32)
            new_influ_scores = (influ_scores[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_features is not None:
                new_features = (point_features[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
        elif comb_type == "along-acc-coord-grad":
            new_coords = query_coords + acc_coord_grad[inds, :] * move_scale
            nns_dists, nns_inds = pc.query(new_coords, k=k)
            nns_dists = nns_dists.astype(np.float32)
            new_influ_scores = (influ_scores[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
            if point_features is not None:
                new_features = (point_features[nns_inds, :] * (1 / (nns_dists + 1e-6)).reshape(-1, k, 1)).sum(axis=-2) / (1 / (nns_dists + 1e-6)).sum(axis=-1, keepdims=True)
        else:
            raise NotImplementedError
    return new_coords, len(new_coords), new_influ_scores, new_features


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
        N, H, W, _ = coords.shape
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
        N, H, W, _ = coords.shape
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


def activation_func(act_type='leakyrelu', neg_slope=0.2, inplace=True, num_channels=128, a=1., b=1., trainable=False):
    act_type = act_type.lower()
    if act_type == 'none':
        layer = nn.Identity()
    elif act_type == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_channels)
    elif act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == '+1':
        layer = PlusOneActivation()
    elif act_type == 'relu+1':
        layer = nn.Sequential(nn.ReLU(inplace), PlusOneActivation())
    elif act_type == 'tanh':
        layer = nn.Tanh()
    elif act_type == 'shifted_tanh':
        layer = ShiftedTanh()
    elif act_type == 'sigmoid':
        layer = nn.Sigmoid()
    elif act_type == 'gelu':
        layer = nn.GELU()
    elif act_type == 'gaussian':
        layer = GaussianActivation(a, trainable)
    elif act_type == 'quadratic':
        layer = QuadraticActivation(a, trainable)
    elif act_type == 'multi-quadratic':
        layer = MultiQuadraticActivation(a, trainable)
    elif act_type == 'laplacian':
        layer = LaplacianActivation(a, trainable)
    elif act_type == 'super-gaussian':
        layer = SuperGaussianActivation(a, b, trainable)
    elif act_type == 'expsin':
        layer = ExpSinActivation(a, trainable)
    elif act_type == 'clamp':
        layer = Clamp(0, 1)
    elif 'sine' in act_type:
        layer = Sine(factor=a)
    elif 'softplus' in act_type:
        a, b, c = [float(i) for i in act_type.split('_')[1:]]
        print(
            'Softplus activation: a={:.2f}, b={:.2f}, c={:.2f}'.format(a, b, c))
        layer = SoftplusActivation(a, b, c)
    else:
        raise NotImplementedError(
            'activation layer [{:s}] is not found'.format(act_type))
    return layer


def posenc(x, L_embed, factor=2.0, without_self=False, mult_factor=1.0):
    if without_self:
        rets = []
    else:
        rets = [x]
    for i in range(L_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn(factor**i * x * mult_factor))
    # return torch.cat(rets, 1)
    # To make sure the dimensions of the same meaning are together
    return torch.flatten(torch.stack(rets, -1), start_dim=-2, end_dim=-1)


class PoseEnc(nn.Module):
    def __init__(self, factor=2.0, mult_factor=1.0):
        super(PoseEnc, self).__init__()
        self.factor = factor
        self.mult_factor = mult_factor

    def forward(self, x, L_embed, without_self=False):
        return posenc(x, L_embed, self.factor, without_self, self.mult_factor)


def normalize_vector(x, eps=0.):
    # assert(x.shape[-1] == 3)
    return x / (torch.norm(x, dim=-1, keepdim=True) + eps)


def create_learning_rate_fn(optimizer, max_steps, args, debug=False):
    """Create learning rate schedule."""
    if args.type == "none":
        return None

    if args.warmup > 0:
        warmup_start_factor = 1e-16
    else:
        warmup_start_factor = 1.0

    warmup_fn = lr_scheduler.LinearLR(optimizer,
                                      start_factor=warmup_start_factor,
                                      end_factor=1.0,
                                      total_iters=args.warmup,
                                      verbose=debug)

    if args.type == "linear":
        decay_fn = lr_scheduler.LinearLR(optimizer,
                                         start_factor=1.0,
                                         end_factor=0.,
                                         total_iters=max_steps - args.warmup,
                                         verbose=debug)
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    elif args.type == "cosine":
        cosine_steps = max(max_steps - args.warmup, 1)
        decay_fn = lr_scheduler.CosineAnnealingLR(optimizer,
                                                  T_max=cosine_steps,
                                                  verbose=debug)
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    elif args.type == "cosine-hlfperiod":
        cosine_steps = max(max_steps - args.warmup, 1) * 2
        decay_fn = lr_scheduler.CosineAnnealingLR(optimizer,
                                                  T_max=cosine_steps,
                                                  verbose=debug)
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    elif args.type == "exp":
        decay_fn = lr_scheduler.ExponentialLR(optimizer,
                                              gamma=args.gamma,
                                              verbose=debug)
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    elif args.type == "stop":
        decay_fn = lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=0.0, verbose=debug)
        schedulers = [warmup_fn, decay_fn]
        milestones = [args.warmup]

    else:
        raise NotImplementedError

    schedule_fn = lr_scheduler.SequentialLR(optimizer,
                                            schedulers=schedulers,
                                            milestones=milestones,
                                            verbose=debug)

    return schedule_fn


class Sine(nn.Module):
    def __init__(self, factor=30):
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return torch.sin(x * self.factor)


class Clamp(nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clamp(x, self.min_val, self.max_val)


class ShiftedTanh(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (torch.tanh(x) + 1) / 2


class SoftplusActivation(nn.Module):
    def __init__(self, c1=1, c2=1, c3=0):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

    def forward(self, x):
        return self.c1 * nn.functional.softplus(self.c2 * x + self.c3)


class GaussianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))


class QuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)


class MultiQuadraticActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return 1/(1+(self.a*x)**2)**0.5


class LaplacianActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x)/self.a)


class SuperGaussianActivation(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x**2/(2*self.a**2))**self.b


class ExpSinActivation(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a*torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a*x))


class PlusOneActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x + 1
