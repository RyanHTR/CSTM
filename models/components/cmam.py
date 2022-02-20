import torch
import torch.nn as nn
import torch.nn.functional as F
from .i3d import Unit3Dpy

__all__ = ['CMAM']


class ConvBN(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=1,
                 padding='SAME', activation='relu'):
        super(ConvBN, self).__init__()
        if padding == 'SAME':
            pad = (kernel_size - 1) // 2
        else:
            pad = 0
        self.activation = activation
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, padding=pad, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        if activation == 'relu':
            self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.relu(x)
        return x


class Interp(nn.Module):
    def __init__(self, scale=2):
        super(Interp, self).__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale)


class CMAM(nn.Module):
    def __init__(self, visual_channel_in, visual_channel_out, txt_channel,
                 upsample=False, num_dim=2, zero_init=False):
        super(CMAM, self).__init__()
        self.upsample = upsample
        self.num_dim = num_dim
        self.zero_init = zero_init
        self.txt_channel = txt_channel
        if self.num_dim == 2:
            conv_block = ConvBN
            kernel = 3
        else:
            conv_block = Unit3Dpy
            kernel = (3, 3, 3)
        if self.upsample:
            self.vt = nn.Sequential(
                Interp(2),
                conv_block(visual_channel_in + 8, visual_channel_out, kernel_size=kernel))
        else:
            self.vt = conv_block(visual_channel_in + 8, visual_channel_out, kernel_size=kernel)

        self.v_key = conv_block(visual_channel_out, txt_channel, activation=None)

        self.l_query = nn.Sequential(
            nn.Conv1d(txt_channel, txt_channel, 1, bias=False),
            nn.BatchNorm1d(txt_channel)
        )

        self.gamma = nn.Sequential(
            nn.Linear(txt_channel, visual_channel_out),
            nn.Sigmoid()
        )

        if self.zero_init:
            self.gamma[0].weight.data.zero_()
            self.gamma[0].bias.data.zero_()

    def forward(self, visual_feat, txt, spatial):
        # txt: [B, C, N]
        up_video = self.vt(torch.cat([visual_feat, spatial], 1))
        key = self.v_key(up_video)   # [B, C, H, W], [B, C, T, H, W]
        key = key.reshape(key.shape[0], key.shape[1], -1)  # [B, C, HW], [B, C, THW]
        query = self.l_query(txt)   # [B, C, N]
        attn_map = torch.einsum('bcn,bct->bnt', key, query)   # [B, HW, N], [B, THW, N]
        attn_map = attn_map.sum(1, keepdims=True)     # [B, 1, N]
        attn_map = F.normalize(attn_map, p=2, dim=-1)

        attn_map = torch.softmax(attn_map, -1)   # [B, 1, T]
        weighted_txt = (attn_map * txt).sum(-1)   # [B, C]

        if self.num_dim == 2:
            txt_gamma = self.gamma(weighted_txt)[:, :, None, None]
        else:
            txt_gamma = self.gamma(weighted_txt)[:, :, None, None, None]
        if self.zero_init:
            up_video = up_video * (txt_gamma - 0.5)
        else:
            up_video = up_video * txt_gamma
        return up_video
