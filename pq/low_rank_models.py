# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
from timm.layers.helpers import to_2tuple
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from models import DiTBlock
from models import DiT

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class attn_uv(Attention):
    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: nn.Module = nn.LayerNorm,
            qkv_use_uv: bool = True, 
            qkv_len: int = 0, 
            proj_use_uv: bool = True, 
            proj_len: int = 0
    ) -> None:
        super(attn_uv, self).__init__(dim, num_heads, qkv_bias, qk_norm, attn_drop, proj_drop, norm_layer)

        self.qkv_use_uv = qkv_use_uv
        if not self.qkv_use_uv:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv_u = nn.Linear(dim, qkv_len, bias=qkv_bias)
            self.qkv_v = nn.Linear(qkv_len, dim * 3)
            del self.qkv

        self.proj_use_uv = proj_use_uv
        if not self.proj_use_uv:
            self.proj = nn.Linear(dim, dim)
        else:
            self.proj_u = nn.Linear(dim, proj_len)
            self.proj_v = nn.Linear(proj_len, dim)
            del self.proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        if not self.qkv_use_uv:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            qkv = self.qkv_v(self.qkv_u(x)).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)

        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        if not self.proj_use_uv:
            x = self.proj(x)
        else:
            x = self.proj_v(self.proj_u(x))
        x = self.proj_drop(x)
        return x

class mlp_uv(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
            fc1_use_uv=True, 
            fc2_use_uv=True, 
            fc1_len = 0, 
            fc2_len = 0
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.fc1_use_uv = fc1_use_uv
        self.fc2_use_uv = fc2_use_uv
        if not self.fc1_use_uv:
            self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        else:
            self.fc1_u = linear_layer(in_features, fc1_len, bias=bias[0])
            self.fc1_v = linear_layer(fc1_len, hidden_features, bias=bias[0])
        
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()

        if not self.fc2_use_uv:
            self.fc2 = linear_layer(hidden_features, out_features, bias=bias[0])
        else:
            self.fc2_u = linear_layer(hidden_features, fc2_len, bias=bias[0])
            self.fc2_v = linear_layer(fc2_len, out_features, bias=bias[0])

        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        if not self.fc1_use_uv:
            x = self.fc1(x)
        else:
            x = self.fc1_v(self.fc1_u(x))
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        if not self.fc2_use_uv:
            x = self.fc2(x)
        else:
            x = self.fc2_v(self.fc2_u(x))
        x = self.drop2(x)
        return x

class DiTBlock_uv(DiTBlock):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0,
                 qkv_use_uv=True, proj_use_uv=True, fc1_use_uv=True, fc2_use_uv=True, adaln_use_uv=True, 
                 qkv_len=0, proj_len=0,  fc1_len=0, fc2_len=0, adaln_len=0,
                 **block_kwargs):
        super(DiTBlock_uv, self).__init__(hidden_size, num_heads, mlp_ratio, **block_kwargs)
        self.attn = attn_uv(
            hidden_size, num_heads=num_heads, qkv_bias=True,
            qkv_use_uv=qkv_use_uv, proj_use_uv=proj_use_uv,
            qkv_len=qkv_len, proj_len=proj_len,
            **block_kwargs
        )
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = mlp_uv(
            in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0,
            fc1_use_uv=fc1_use_uv, fc2_use_uv=fc2_use_uv,
            fc1_len=fc1_len, fc2_len=fc2_len
        )
        self.adaln_use_uv = adaln_use_uv
        if not self.adaln_use_uv:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
        else:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, adaln_len, bias=True),
                nn.Linear(adaln_len, 6 * hidden_size, bias=True)
            )

class DiT_uv(DiT):
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        qkv_use_uv=[], proj_use_uv=[], fc1_use_uv=[], fc2_use_uv=[], adaln_use_uv=[],
        qkv_len=[], proj_len=[], fc1_len=[], fc2_len=[], adaln_len=[]
    ):
        super(DiT_uv, self).__init__(input_size=32, patch_size=2, in_channels=4, hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0, class_dropout_prob=0.1, num_classes=1000, learn_sigma=True)

        self.blocks = nn.ModuleList([
            DiTBlock_uv(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                qkv_use_uv=qkv_use_uv[i], proj_use_uv=proj_use_uv[i], fc1_use_uv=fc1_use_uv[i], fc2_use_uv=fc2_use_uv[i], adaln_use_uv=adaln_use_uv[i],
                qkv_len=qkv_len[i], proj_len=proj_len[i], fc1_len=fc1_len[i], fc2_len=fc2_len[i], adaln_len=adaln_len[i]
            ) for i in range(depth)
        ])
        self.initialize_weights()


def DiT_uv_XL_2(**kwargs):
    return DiT_uv(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_uv_XL_4(**kwargs):
    return DiT_uv(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_uv_XL_8(**kwargs):
    return DiT_uv(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_uv_L_2(**kwargs):
    return DiT_uv(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_uv_L_4(**kwargs):
    return DiT_uv(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_uv_L_8(**kwargs):
    return DiT_uv(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_uv_B_2(**kwargs):
    return DiT_uv(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_uv_B_4(**kwargs):
    return DiT_uv(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_uv_B_8(**kwargs):
    return DiT_uv(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_uv_S_2(**kwargs):
    return DiT_uv(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_uv_S_4(**kwargs):
    return DiT_uv(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_uv_S_8(**kwargs):
    return DiT_uv(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_uv_models = {
    'DiT-XL/2': DiT_uv_XL_2,  'DiT-XL/4': DiT_uv_XL_4,  'DiT-XL/8': DiT_uv_XL_8,
    'DiT-L/2':  DiT_uv_L_2,   'DiT-L/4':  DiT_uv_L_4,   'DiT-L/8':  DiT_uv_L_8,
    'DiT-B/2':  DiT_uv_B_2,   'DiT-B/4':  DiT_uv_B_4,   'DiT-B/8':  DiT_uv_B_8,
    'DiT-S/2':  DiT_uv_S_2,   'DiT-S/4':  DiT_uv_S_4,   'DiT-S/8':  DiT_uv_S_8,
}
