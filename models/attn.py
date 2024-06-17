import torch
import torch.nn as nn
from torch import autocast
import math
from .mlp import MLP
from .utils import PoseEnc, activation_func


def get_proximity_attention_layer(args, v_extra_dim=0, k_extra_dim=0, q_extra_dim=0, eps=1e-6, use_amp=False, amp_dtype=torch.float16):
    k_dim_map = {
        1: [3, 3, 3],
    }
    k_dim = k_dim_map[args.k_type]

    q_dim_map = {
        1: [3],
    }
    q_dim = q_dim_map[args.q_type]

    v_dim_map = {
        1: [3, 3],
    }
    v_dim = v_dim_map[args.v_type]

    return ProximityAttention(d_k=k_dim, d_q=q_dim, d_v=v_dim, d_model=args.d_model, embed_args=args.embed, 
                              score_act=args.score_act, d_ko=k_extra_dim, d_qo=q_extra_dim,
                              d_vo=v_extra_dim, eps=eps, use_amp=use_amp, amp_dtype=amp_dtype)


class LayerNorm(nn.Module):
    "Construct a layernorm module"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(query, key, kernel_type):
    """
        Compute Attention Scores
        query: [batch_size, 1, query_len, d_kq] or [batch_size, query_len, d_kq]
        key:   [batch_size, 1, seq_len, d_kq] or [batch_size, seq_len, d_kq]
    """
    d_kq = query.size(-1)

    if kernel_type == "scaled-dot":
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_kq)
    elif kernel_type == "-scaled-dot":
        scores = -torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_kq)
    elif kernel_type == "dot":
        scores = torch.matmul(query, key.transpose(-2, -1))
    elif kernel_type == "-dot":
        scores = -torch.matmul(query, key.transpose(-2, -1))
    elif kernel_type == "l1-dist":
        scores = torch.norm(query.unsqueeze(-2) -
                            key.unsqueeze(-3), p=1, dim=-1)
    elif kernel_type == "-l1-dist":
        scores = -torch.norm(query.unsqueeze(-2) -
                             key.unsqueeze(-3), p=1, dim=-1)
    elif kernel_type == "l2-dist":
        scores = torch.norm(query.unsqueeze(-2) -
                            key.unsqueeze(-3), p=2, dim=-1)
    elif kernel_type == "-l2-dist":
        scores = -torch.norm(query.unsqueeze(-2) -
                             key.unsqueeze(-3), p=2, dim=-1)
    elif kernel_type == "scaled-l2-dist":
        scores = torch.norm(query.unsqueeze(-2) -
                            key.unsqueeze(-3), p=2, dim=-1) / math.sqrt(d_kq)
    elif kernel_type == "-scaled-l2-dist":
        scores = -torch.norm(query.unsqueeze(-2) -
                             key.unsqueeze(-3), p=2, dim=-1) / math.sqrt(d_kq)
    elif kernel_type == "cosine":
        scores = torch.matmul(query, key.transpose(-2, -1)) / (
            torch.norm(query, dim=-1, keepdim=True)
            * torch.norm(key, dim=-1, keepdim=True).transpose(-2, -1)
        )
    else:
        raise ValueError("Unknown kernel type: {}".format(kernel_type))

    return scores


class FeedForward(nn.Module):
    "Implements FFN module."

    def __init__(self, d_input, d_output, d_ff, n_layer=2, act="relu", last_act="none", dropout=0.1, norm="layernorm",
                 residual=False, act_a=1.0, act_b=1.0, act_trainable=False, use_wn=False, eps=1e-6, skip_layers=[],
                 half_layers=[]):
        super(FeedForward, self).__init__()
        self.eps = eps
        self.d_input = d_input
        self.d_output = d_output
        if norm == "layernorm":
            self.innorm = LayerNorm(d_input, eps)
            self.outnorm = LayerNorm(d_output, eps)
        elif norm == "none":
            self.innorm = nn.Identity()
            self.outnorm = nn.Identity()
        else:
            raise ValueError("Invalid attention norm type")
        self.dropout = nn.Dropout(dropout)
        self.mlp = MLP(d_input, n_layer, d_ff, d_output, act_type=act, last_act_type=last_act, use_wn=use_wn,
                       a=act_a, b=act_b, trainable=act_trainable, skip_layers=skip_layers, half_layers=half_layers)
        self.residual = residual

    def forward(self, x):
        if self.residual and x.shape[-1] == self.d_output:
            return self.outnorm(x + self.dropout(self.mlp(self.innorm(x))))
        else:
            return self.outnorm(self.dropout(self.mlp(self.innorm(x))))


class Embeddings(nn.Module):
    def __init__(self, d_k, d_q, d_v, d_model, args, d_ko=0, d_qo=0, d_vo=0, eps=1e-6):
        super(Embeddings, self).__init__()
        self.d_k = d_k
        self.d_q = d_q
        self.d_v = d_v
        self.args = args
        self.embed_type = args.embed_type
        self.d_model = d_model
        self.d_ko = d_ko
        self.d_qo = d_qo
        self.d_vo = d_vo
        self.eps = eps

        self.posenc = PoseEnc(args.pe_factor, args.pe_mult_factor)

        if self.embed_type == 1:
            # Positional Encoding with itself
            d_k = sum([d + d * 2 * args.k_L[i] for i, d in enumerate(d_k)]) + d_ko
            d_q = sum([d + d * 2 * args.q_L[i] for i, d in enumerate(d_q)]) + d_qo
            d_v = sum([d + d * 2 * args.v_L[i] for i, d in enumerate(d_v)]) + d_vo

        elif self.embed_type == 2:
            # Positional Encoding without itself
            d_k = sum([d * 2 * args.k_L[i] for i, d in enumerate(d_k)]) + d_ko
            d_q = sum([d * 2 * args.q_L[i] for i, d in enumerate(d_q)]) + d_qo
            d_v = sum([d * 2 * args.v_L[i] for i, d in enumerate(d_v)]) + d_vo

        else:
            raise ValueError(
                'Unknown embedding type: {}'.format(self.embed_type))

        self.embed_k = FeedForward(d_k, args.key.d_ff_out, args.key.d_ff, args.key.n_ff_layer, args.key.ff_act,
                                    args.key.ff_last_act, args.key.dropout_ff, args.key.norm, args.key.residual_ff,
                                    args.key.ff_act_a, args.key.ff_act_b, args.key.ff_act_trainable, args.key.use_wn, eps,
                                    args.key.skip_layers, args.key.half_layers)
        self.embed_q = FeedForward(d_q, args.query.d_ff_out, args.query.d_ff, args.query.n_ff_layer, args.query.ff_act,
                                    args.query.ff_last_act, args.query.dropout_ff, args.query.norm, args.query.residual_ff,
                                    args.query.ff_act_a, args.query.ff_act_b, args.query.ff_act_trainable, args.query.use_wn, eps,
                                    args.query.skip_layers, args.query.half_layers)
        self.embed_v = FeedForward(d_v, args.value.d_ff_out, args.value.d_ff, args.value.n_ff_layer, args.value.ff_act,
                                    args.value.ff_last_act, args.value.dropout_ff, args.value.norm, args.value.residual_ff,
                                    args.value.ff_act_a, args.value.ff_act_b, args.value.ff_act_trainable, args.value.use_wn, eps,
                                    args.value.skip_layers, args.value.half_layers)

    def forward(self, k_features, q_features, v_features, k_other=None, q_other=None, v_other=None):
        """
            k_features: [(B, H, W, N, Dk_i)]
            q_features: [(B, H, W, 1, Dq_i)]
            v_features: [(B, H, W, N, Dv_i)]
        """

        if self.embed_type == 1:
            pe_k_features = [self.posenc(f, self.args.k_L[i]) for i, f in enumerate(k_features)]
            pe_q_features = [self.posenc(f, self.args.q_L[i]) for i, f in enumerate(q_features)]
            pe_v_features = [self.posenc(f, self.args.v_L[i]) for i, f in enumerate(v_features)]

        elif self.embed_type == 2:
            pe_k_features = [self.posenc(f, self.args.k_L[i], without_self=True) for i, f in enumerate(k_features)]
            pe_q_features = [self.posenc(f, self.args.q_L[i], without_self=True) for i, f in enumerate(q_features)]
            pe_v_features = [self.posenc(f, self.args.v_L[i], without_self=True) for i, f in enumerate(v_features)]

        else:
            raise ValueError('Unknown embedding type: {}'.format(self.embed_type))

        if self.d_ko > 0: pe_k_features = pe_k_features + k_other
        if self.d_qo > 0: pe_q_features = pe_q_features + q_other
        if self.d_vo > 0: pe_v_features = pe_v_features + v_other

        k = torch.cat(pe_k_features, dim=-1).flatten(0, 2)
        q = torch.cat(pe_q_features, dim=-1).flatten(0, 2)
        v = torch.cat(pe_v_features, dim=-1).flatten(0, 2)

        k = self.embed_k(k)
        q = self.embed_q(q)
        v = self.embed_v(v)

        return k, q, v


class AttentionLayer(nn.Module):
    def __init__(self, embed_args, d_model, score_act_type):
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.w_k = nn.Linear(embed_args.key.d_ff_out, d_model)
        self.w_q = nn.Linear(embed_args.query.d_ff_out, d_model)

        nn.init.xavier_uniform_(self.w_k.weight)
        nn.init.xavier_uniform_(self.w_q.weight)

        self.score_act = activation_func(score_act_type)

    def forward(self, key, query, value):
        nbatches, nseqv, _ = value.shape
        _, nseqk, _ = key.shape
        assert nseqv == nseqk

        key = self.w_k(key)
        query = self.w_q(query)

        key = key.view(nbatches, -1, 1, self.d_model).transpose(1, 2)
        query = query.view(nbatches, -1, 1, self.d_model).transpose(1, 2)

        # [nbatches, nhead, nseq, nseq]
        scores = attention(query, key, "scaled-dot")
        scores = self.score_act(scores)
        return scores


class ProximityAttention(nn.Module):
    def __init__(self, d_k, d_q, d_v, d_model, embed_args, score_act="none",
                 d_ko=0, d_qo=0, d_vo=0, eps=1e-6, use_amp=False, amp_dtype=torch.float16):
        super(ProximityAttention, self).__init__()
        self.eps = eps
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype

        self.embed = Embeddings(d_k, d_q, d_v, d_model, embed_args, d_ko, d_qo, d_vo, eps)

        self.attention_layer = AttentionLayer(embed_args, d_model, score_act)

    def forward(self, k_features, q_features, v_features, k_other=None, q_other=None, v_other=None, step=-1):
        """
            k_features: [(H, W, N, Dk_i)]
            q_features: [(H, W, 1, Dq_i)] or [(H, W, N, Dq_i)]
            v_features: [(H, W, N, Dv_i)]
        """

        with autocast(device_type='cuda', dtype=self.amp_dtype, enabled=self.use_amp):
            k, q, v = self.embed(k_features, q_features, v_features, k_other, q_other, v_other)
            scores = self.attention_layer(k, q, v)
        
        return k, q, v, scores
