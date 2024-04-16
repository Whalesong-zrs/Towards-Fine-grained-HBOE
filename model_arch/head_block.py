# Copyright (c) OpenMMLab. All rights reserved.

import math

import torch
import torch.nn as nn
from torch.nn.functional import pad
import torch.nn.functional as F
import os
def nlc_to_nchw(x, hw_shape):
    H, W = hw_shape
    assert len(x.shape) == 3
    B, L, C = x.shape
    assert L == H * W, 'The seq_len doesn\'t match H, W'
    return x.transpose(1, 2).reshape(B, C, H, W)


def nchw_to_nlc(x):
    assert len(x.shape) == 4
    return x.flatten(2).transpose(1, 2).contiguous()



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

        # super().__init__(init_cfg=init_cfg)
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
        # trunc_normal_init(self.relative_position_bias_table, std=0.02)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
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
        # print('attn', attn.shape)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
    
    
    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


# class LocalWindowSelfAttention(BaseModule):
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
        # super().__init__(init_cfg=init_cfg)
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
        # pos emb
        self.pos_emb = nn.Sequential(
            nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False, groups=embed_dims),
            nn.GELU(),
            nn.Conv2d(embed_dims, embed_dims, 3, 1, 1, bias=False, groups=embed_dims),
        )

    def forward(self, x, H, W, **kwargs):
        """Forward function."""
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

        # pos emb
        x_windows = x.view(-1, Wh, Ww, C).permute(0, 3, 1, 2)  # 调整为 (num_windows*B, C, Wh, Ww)
        pos_emb = self.pos_emb(x_windows)  # 计算位置嵌入
        x_windows = x_windows + pos_emb  # 将位置嵌入加到特征上
        x = x_windows.permute(0, 2, 3, 1).reshape(-1, Wh * Ww, C)  # 调整回 (B*num_window, Wh*Ww, C)

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
    

# class CrossFFN(BaseModule):
class CrossFFN(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_cfg=dict(type='GELU'),
                 dw_act_cfg=dict(type='GELU'),
                norm_cfg=dict(type='BN'),
                 init_cfg=None):
        super(CrossFFN, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=1)
        self.act1 = nn.GELU()
        self.norm1 = nn.SyncBatchNorm(hidden_features)
        self.dw3x3 = nn.Conv2d(
            hidden_features,
            hidden_features,
            kernel_size=3,
            stride=1,
            groups=hidden_features,
            padding=1)
        self.act2 = nn.GELU()
        self.norm2 = nn.SyncBatchNorm(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.act3 = nn.GELU()
        self.norm3 = nn.SyncBatchNorm(out_features)


    def forward(self, x, H, W):
        x = nlc_to_nchw(x, (H, W))
        

        x = self.fc1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.dw3x3(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.fc2(x)
        x = self.norm3(x)
        x = self.act3(x)

        x = nchw_to_nlc(x)
        return x


# class HRFormerBlock(BaseModule):
class OEHeadBlock(nn.Module):

    expansion = 1

    def __init__(self,
                 in_features,
                 out_features,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.0,
                 drop_path=0.0,
                 act_cfg=dict(type='GELU'),
                #  norm_cfg=dict(type='SyncBN'),
                norm_cfg=dict(type='BN'),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 init_cfg=None,
                 **kwargs):
        super(OEHeadBlock, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.LayerNorm(in_features, eps=1e-6)
        self.attn = LocalWindowSelfAttention(
            in_features,
            num_heads=num_heads,
            window_size=window_size,
            init_cfg=None,
            **kwargs)

        self.norm2 = nn.LayerNorm(out_features, eps=1e-6)
        self.ffn = CrossFFN(
            in_features=in_features,
            hidden_features=int(in_features * mlp_ratio),
            out_features=out_features,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dw_act_cfg=act_cfg,
            init_cfg=None)

        
    def forward(self, x):
        """Forward function."""
        B, C, H, W = x.size()
        # Attention
        x = x.view(B, C, -1).permute(0, 2, 1)
    
        x = x + self.attn(self.norm1(x), H, W)
        self.feature_map = self.attn(self.norm1(x), H, W).detach()
        self.feature_map = self.feature_map.permute(0, 2, 1).view(B, C, H, W)
        # FFN
        x = x + self.ffn(self.norm2(x), H, W)
        x = x.permute(0, 2, 1).view(B, C, H, W)
        return x

    def get_feature_map(self):
        # 返回保存的特征图
        return self.feature_map

if __name__ == "__main__":
    model = OEHeadBlock(in_features=128, out_features=128, num_heads=8)
    
    model.eval()
    model.cuda()
    print(22222222222222222)
    input = torch.rand(1, 128, 256, 192)
    input = input.cuda()
    output = model(input)
    print(output.shape)
    
