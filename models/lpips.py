import torch
import torch.nn as nn
from torchvision import models as tv
from collections import namedtuple
import os


class vgg16(nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = tv.vgg16(weights=tv.VGG16_Weights.IMAGENET1K_V1 if pretrained else None).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        self.slice5 = nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2,
                          h_relu3_3, h_relu4_3, h_relu5_3)

        return out


class ScalingLayer(nn.Module):
    # For rescaling the input to vgg16
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor(
            [-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor(
            [.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(
        torch.sum(in_feat ** 2, dim=1, keepdim=True) + eps)
    return in_feat / (norm_factor + eps)


def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


class NetLinLayer(nn.Module):
    ''' A single linear layer used as placeholder for LPIPS learnt weights '''

    def __init__(self):
        super(NetLinLayer, self).__init__()
        self.weight = None

    def forward(self, inp):
        out = torch.sum(self.weight * inp, 1, keepdim=True)
        return out


class LPNet(nn.Module):
    def __init__(self):
        super(LPNet, self).__init__()

        self.scaling_layer = ScalingLayer()
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.L = 5
        self.lins = [NetLinLayer() for _ in range(self.L)]
        self.lins = nn.ModuleList(self.lins)
        model_path = os.path.abspath(
            os.path.join('.', 'vgg.pth'))
        print('Loading model from: %s' % model_path)
        weights = torch.load(model_path, map_location='cpu')
        for ll in range(self.L):
            self.lins[ll].weight = nn.Parameter(
                weights["lin%d.model.1.weight" % ll])

    def forward(self, in0, in1):
        in0 = in0.permute(0, 3, 1, 2)
        in1 = in1.permute(0, 3, 1, 2)
        in0 = 2 * in0 - 1
        in0_input = self.scaling_layer(in0)
        in1 = 2 * in1 - 1
        in1_input = self.scaling_layer(in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(
                outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
               for kk in range(self.L)]

        val = res[0]
        for ll in range(1, self.L):
            val += res[ll]

        return val.squeeze()
