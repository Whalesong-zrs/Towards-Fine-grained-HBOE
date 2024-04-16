# Copyright (c) OpenMMLab. All rights reserved.

import math

import torch
import torch.nn as nn
from mmcv.cnn import build_activation_layer, build_conv_layer, build_norm_layer
from mmcv.cnn.bricks.transformer import build_dropout
from mmengine.model import BaseModule, trunc_normal_init
from torch.nn.functional import pad

from .hrnet import Bottleneck, HRModule, HRNet
import os


maps = []
def nlc_to_nchw(x, hw_shape):

    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()


def build_drop_path(drop_path_rate):
    """Build drop path layer."""
    return build_dropout(dict(type='DropPath', drop_prob=drop_path_rate))

class WindowMSA(nn.Module):

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 with_rpe=True,
                 init_cfg=None):

        super(WindowMSA, self).__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5

        self.with_rpe = with_rpe
        if self.with_rpe:
            # define a parameter table of relative position bias
            # 比如说3*3的窗口，最左上角的点相对最右下角的点的坐标是(-2, -2),最右下角对于最左上角的点的坐标是(2, ,2)
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(
                    (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                    num_heads))  # 2*Wh-1 * 2*Ww-1, nH
                
            Wh, Ww = self.window_size
            rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
            rel_position_index = rel_index_coords + rel_index_coords.T
            rel_position_index = rel_position_index.flip(1).contiguous()
            self.register_buffer('relative_position_index', rel_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        trunc_normal_init(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.with_rpe:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    self.window_size[0] * self.window_size[1],
                    self.window_size[0] * self.window_size[1],
                    -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)
       
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

       
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class LocalWindowSelfAttention(nn.Module):
    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 with_rpe=True,
                 with_pad_mask=False,
                 init_cfg=None):
        super(LocalWindowSelfAttention, self).__init__()
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.with_pad_mask = with_pad_mask
        self.attn = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=window_size,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            with_rpe=with_rpe,
            init_cfg=init_cfg)

    def forward(self, x, H, W, **kwargs):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        Wh, Ww = self.window_size

        # center-pad the feature on H and W axes
        pad_h = math.ceil(H / Wh) * Wh - H
        pad_w = math.ceil(W / Ww) * Ww - W
        x = pad(x, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2))

        # permute
        x = x.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, Wh * Ww, C)  # (B*num_window, Wh*Ww, C)

        # attention
        if self.with_pad_mask and pad_h > 0 and pad_w > 0:
            pad_mask = x.new_zeros(1, H, W, 1)
            pad_mask = pad(
                pad_mask, [
                    0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ],
                value=-float('inf'))
            pad_mask = pad_mask.view(1, math.ceil(H / Wh), Wh,
                                     math.ceil(W / Ww), Ww, 1)
            pad_mask = pad_mask.permute(1, 3, 0, 2, 4, 5)
            pad_mask = pad_mask.reshape(-1, Wh * Ww)
            pad_mask = pad_mask[:, None, :].expand([-1, Wh * Ww, -1])
            out = self.attn(x, pad_mask, **kwargs)
        else:
            out = self.attn(x, **kwargs)

        # reverse permutation
        out = out.reshape(B, math.ceil(H / Wh), math.ceil(W / Ww), Wh, Ww, C)
        out = out.permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, H + pad_h, W + pad_w, C)

        # de-pad
        out = out[:, pad_h // 2:H + pad_h // 2, pad_w // 2:W + pad_w // 2]

        return out.reshape(B, N, C)


class CrossFFN(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 dw_act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN'),
                 init_cfg=None):
        super(CrossFFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = build_activation_layer(act_cfg)
        self.norm1 = build_norm_layer(norm_cfg, hidden_features)[1]
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1)
        self.act2 = build_activation_layer(dw_act_cfg)
        self.norm2 = build_norm_layer(norm_cfg, hidden_features)[1]
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = build_activation_layer(act_cfg)
        self.norm3 = build_norm_layer(norm_cfg, out_features)[1]

        # put the modules togather
        self.layers = [
            self.fc1, self.norm1, self.act1, self.dw3x3, self.norm2, self.act2,
            self.fc2, self.norm3, self.act3
        ]

    def forward(self, x, H, W):
        """Forward function."""
        x = nlc_to_nchw(x, (H, W))
        for layer in self.layers:
            x = layer(x)
        x = nchw_to_nlc(x)
        return x


class OEFormerBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_features,
                 out_features,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.0,
                 drop_path=0.0,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN'),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None,
                 **kwargs):

        super(OEFormerBlock, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = build_norm_layer(transformer_norm_cfg, in_features)[1]
        self.attn = LocalWindowSelfAttention(
            in_features,
            num_heads=num_heads,
            window_size=window_size,
            init_cfg=None,
            **kwargs)

        self.norm2 = build_norm_layer(transformer_norm_cfg, out_features)[1]
        self.ffn = CrossFFN(
            in_features=in_features,
            hidden_features=int(in_features * mlp_ratio),
            out_features=out_features,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dw_act_cfg=act_cfg,
            init_cfg=None)

        self.drop_path = build_drop_path(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        """Forward function."""
        B, C, H, W = x.size()
        # Attention
        x = x.view(B, C, -1).permute(0, 2, 1)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        # FFN
        x = x + self.drop_path(self.ffn(self.norm2(x), H, W))
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

    def extra_repr(self):
        """(Optional) Set the extra information about this module."""
        return 'num_heads={}, window_size={}, mlp_ratio={}'.format(
            self.num_heads, self.window_size, self.mlp_ratio)


class OEFomerModule(HRModule):

    def __init__(self,
                 num_branches,
                 block,
                 num_blocks,
                 num_inchannels,
                 num_channels,
                 num_heads,
                 num_window_sizes,
                 num_mlp_ratios,
                 multiscale_output=True,
                 drop_paths=0.0,
                 with_rpe=True,
                 with_pad_mask=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 with_cp=False,
                 upsample_cfg=dict(mode='bilinear', align_corners=False),
                 **kwargs):

        self.transformer_norm_cfg = transformer_norm_cfg
        self.drop_paths = drop_paths
        self.num_heads = num_heads
        self.num_window_sizes = num_window_sizes
        self.num_mlp_ratios = num_mlp_ratios
        self.with_rpe = with_rpe
        self.with_pad_mask = with_pad_mask

        super().__init__(num_branches, block, num_blocks, num_inchannels,
                         num_channels, multiscale_output, with_cp, conv_cfg,
                         norm_cfg, upsample_cfg, **kwargs)

    def _make_one_branch(self,
                         branch_index,
                         block,
                         num_blocks,
                         num_channels,
                         stride=1):
        """Build one branch."""
        assert stride == 1 and self.in_channels[branch_index] == num_channels[
            branch_index]
        layers = []
        layers.append(
            block(
                self.in_channels[branch_index],
                num_channels[branch_index],
                num_heads=self.num_heads[branch_index],
                window_size=self.num_window_sizes[branch_index],
                mlp_ratio=self.num_mlp_ratios[branch_index],
                drop_path=self.drop_paths[0],
                norm_cfg=self.norm_cfg,
                transformer_norm_cfg=self.transformer_norm_cfg,
                init_cfg=None,
                with_rpe=self.with_rpe,
                with_pad_mask=self.with_pad_mask))

        self.in_channels[
            branch_index] = self.in_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.in_channels[branch_index],
                    num_channels[branch_index],
                    num_heads=self.num_heads[branch_index],
                    window_size=self.num_window_sizes[branch_index],
                    mlp_ratio=self.num_mlp_ratios[branch_index],
                    drop_path=self.drop_paths[i],
                    norm_cfg=self.norm_cfg,
                    transformer_norm_cfg=self.transformer_norm_cfg,
                    init_cfg=None,
                    with_rpe=self.with_rpe,
                    with_pad_mask=self.with_pad_mask))
        return nn.Sequential(*layers)

    def _make_fuse_layers(self):
        """Build fuse layers."""
        if self.num_branches == 1:
            return None
        num_branches = self.num_branches
        num_inchannels = self.in_channels
        fuse_layers = []
        for i in range(num_branches if self.multiscale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_inchannels[j],
                                num_inchannels[i],
                                kernel_size=1,
                                stride=1,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_inchannels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i),
                                mode=self.upsample_cfg['mode'],
                                align_corners=self.
                                upsample_cfg['align_corners'])))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            with_out_act = False
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            with_out_act = True
                        sub_modules = [
                            build_conv_layer(
                                self.conv_cfg,
                                num_inchannels[j],
                                num_inchannels[j],
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=num_inchannels[j],
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg,
                                             num_inchannels[j])[1],
                            build_conv_layer(
                                self.conv_cfg,
                                num_inchannels[j],
                                num_outchannels_conv3x3,
                                kernel_size=1,
                                stride=1,
                                bias=False,
                            ),
                            build_norm_layer(self.norm_cfg,
                                             num_outchannels_conv3x3)[1]
                        ]
                        if with_out_act:
                            sub_modules.append(nn.ReLU(False))
                        conv3x3s.append(nn.Sequential(*sub_modules))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        """Return the number of input channels."""
        return self.in_channels


class OEFormer(HRNet):
    blocks_dict = {'BOTTLENECK': Bottleneck, 'OEFORMERBLOCK': OEFormerBlock}

    def __init__(
        self,
        extra,
        in_channels=3,
        conv_cfg=None,
        norm_cfg=dict(type='BN', requires_grad=True),
        transformer_norm_cfg=dict(type='LN', eps=1e-6),
        norm_eval=False,
        with_cp=False,
        zero_init_residual=False,
        frozen_stages=-1,
        init_cfg=[
            dict(type='Normal', std=0.001, layer=['Conv2d']),
            dict(type='Constant', val=1, layer=['_BatchNorm', 'GroupNorm'])
        ],
    ):
        print(extra)
        # stochastic depth
        depths = [
            extra[stage]['num_blocks'][0] * extra[stage]['num_modules']
            for stage in ['stage2', 'stage3', 'stage4']
        ]
        depth_s2, depth_s3, _ = depths
        drop_path_rate = extra['drop_path_rate']
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]
        extra['stage2']['drop_path_rates'] = dpr[0:depth_s2]
        extra['stage3']['drop_path_rates'] = dpr[depth_s2:depth_s2 + depth_s3]
        extra['stage4']['drop_path_rates'] = dpr[depth_s2 + depth_s3:]

       
        upsample_cfg = extra.get('upsample', {
            'mode': 'bilinear',
            'align_corners': False
        })
        extra['upsample'] = upsample_cfg
        self.transformer_norm_cfg = transformer_norm_cfg
        self.with_rpe = extra.get('with_rpe', True)
        self.with_pad_mask = extra.get('with_pad_mask', False)

        super().__init__(extra, in_channels, conv_cfg, norm_cfg, norm_eval,
                         with_cp, zero_init_residual, frozen_stages, init_cfg)

    def _make_stage(self,
                    layer_config,
                    num_inchannels,
                    multiscale_output=True):
        """Make each stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']
        block = self.blocks_dict[layer_config['block']]
        num_heads = layer_config['num_heads']
        num_window_sizes = layer_config['window_sizes']
        num_mlp_ratios = layer_config['mlp_ratios']
        drop_path_rates = layer_config['drop_path_rates']

        modules = []
        for i in range(num_modules):
            # multiscale_output is only used at the last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                OEFomerModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    num_heads,
                    num_window_sizes,
                    num_mlp_ratios,
                    reset_multiscale_output,
                    drop_paths=drop_path_rates[num_blocks[0] *
                                               i:num_blocks[0] * (i + 1)],
                    with_rpe=self.with_rpe,
                    with_pad_mask=self.with_pad_mask,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    transformer_norm_cfg=self.transformer_norm_cfg,
                    with_cp=self.with_cp,
                    upsample_cfg=self.upsample_cfg))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def init_weights(self, pretrained=''):
        # logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained, map_location='cpu')
            self.load_state_dict(pretrained_state_dict, strict=False)
        elif pretrained:
            raise ValueError('{} is not exist!'.format(pretrained))

