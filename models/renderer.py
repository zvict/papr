import torch
from .unet import SmallUNet
from .mlp import MLP


class MLPGenerator(torch.nn.Module):
    def __init__(self, inp_dim=2, num_layers=3, num_channels=128, out_dim=2, act_type="leakyrelu", last_act_type="none",
                 use_wn=True, a=1., b=1., trainable=False, skip_layers=[], bias=True, half_layers=[], residual_layers=[],
                 residual_dims=[]):
        super(MLPGenerator, self).__init__()
        self.mlp = MLP(inp_dim=inp_dim, num_layers=num_layers, num_channels=num_channels, out_dim=out_dim,
                          act_type=act_type, last_act_type=last_act_type, use_wn=use_wn, a=a, b=b, trainable=trainable,
                          skip_layers=skip_layers, bias=bias, half_layers=half_layers, residual_layers=residual_layers,
                          residual_dims=residual_dims)
        
    def forward(self, x, residuals=[], gamma=None, beta=None): # (N, C, H, W)
        return self.mlp(x.permute(0, 2, 3, 1), residuals).permute(0, 3, 1, 2)



def get_generator(args, in_c, out_c, use_amp=False, amp_dtype=torch.float16):
    if args.type == "small-unet":
        opt = args.small_unet
        return SmallUNet(in_c, out_c, bilinear=opt.bilinear, single=opt.single, norm=opt.norm, last_act=opt.last_act,
                         use_amp=use_amp, amp_dtype=amp_dtype, affine_layer=opt.affine_layer)
    elif args.type == "mlp":
        opt = args.mlp
        return MLPGenerator(inp_dim=in_c, num_layers=opt.num_layers, num_channels=opt.num_channels, out_dim=out_c,
                            act_type=opt.act_type, last_act_type=opt.last_act_type, use_wn=opt.use_wn, a=opt.act_a, b=opt.act_b,
                            trainable=opt.act_trainable, skip_layers=opt.skip_layers, bias=opt.bias, half_layers=opt.half_layers,
                            residual_layers=opt.residual_layers, residual_dims=opt.residual_dims)
    else:
        raise NotImplementedError(
            'generator type [{:d}] is not supported'.format(args.type))
