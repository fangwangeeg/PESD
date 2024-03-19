"""
Transformer for EEG classification

The core idea is slicing, which means to split the signal along the time dimension. Slice is just like the patch in Vision Transformer.
"""

import torch.nn.functional as F
from torch import nn
import torch
import math

def trunc_normal_(tensor, mean=0., std=1.):
    # Truncated normal initialization
    # Reference: https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/9
    with torch.no_grad():
        tensor.normal_(mean, std)
        tensor.clamp_(-2 * std, 2 * std)

class DropPath (nn.Module) :
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None) :
        super (DropPath, self).__init__ ()
        self.drop_prob = drop_prob

    def forward(self, x) :
        return drop_path (x, self.drop_prob, self.training)

    def extra_repr(self) -> str :
        return 'p={}'.format (self.drop_prob)


class Mlp (nn.Module) :
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) :
        super ().__init__ ()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear (in_features, hidden_features)
        self.act = act_layer ()
        self.fc2 = nn.Linear (hidden_features, out_features)
        self.drop = nn.Dropout (drop)

    def forward(self, x) :
        x = self.fc1 (x)
        x = self.act (x)
        # x = self.drop(x)
        # commit this for the orignal BERT implement
        x = self.fc2 (x)
        x = self.drop (x)
        return x


class Attention (nn.Module) :
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., window_size=None, attn_head_dim=None) :
        super ().__init__ ()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None :
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear (dim, all_head_dim * 3, bias = False)
        if qkv_bias :
            self.q_bias = nn.Parameter (torch.zeros (all_head_dim))
            self.v_bias = nn.Parameter (torch.zeros (all_head_dim))
        else :
            self.q_bias = None
            self.v_bias = None

        if window_size :
            self.window_size = window_size
            self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
            self.relative_position_bias_table = nn.Parameter (
                torch.zeros (self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
            # cls to token & token 2 cls & cls to cls

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange (window_size[0])
            coords_w = torch.arange (window_size[1])
            coords = torch.stack (torch.meshgrid ([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten (coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute (1, 2, 0).contiguous ()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += window_size[1] - 1
            relative_coords[:, :, 0] *= 2 * window_size[1] - 1
            relative_position_index = \
                torch.zeros (size = (window_size[0] * window_size[1] + 1,) * 2, dtype = relative_coords.dtype)
            relative_position_index[1 :, 1 :] = relative_coords.sum (-1)  # Wh*Ww, Wh*Ww
            relative_position_index[0, 0 :] = self.num_relative_distance - 3
            relative_position_index[0 :, 0] = self.num_relative_distance - 2
            relative_position_index[0, 0] = self.num_relative_distance - 1

            self.register_buffer ("relative_position_index", relative_position_index)
        else :
            self.window_size = None
            self.relative_position_bias_table = None
            self.relative_position_index = None

        self.attn_drop = nn.Dropout (attn_drop)
        self.proj = nn.Linear (all_head_dim, dim)
        self.proj_drop = nn.Dropout (proj_drop)

    def forward(self, x, rel_pos_bias=None) :
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None :
            qkv_bias = torch.cat ((self.q_bias, torch.zeros_like (self.v_bias, requires_grad = False), self.v_bias))
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear (input = x, weight = self.qkv.weight, bias = qkv_bias)
        qkv = qkv.reshape (B, N, 3, self.num_heads, -1).permute (2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose (-2, -1))

        if self.relative_position_bias_table is not None :
            relative_position_bias = \
                self.relative_position_bias_table[self.relative_position_index.view (-1)].view (
                    self.window_size[0] * self.window_size[1] + 1,
                    self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute (2, 0, 1).contiguous ()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze (0)

        if rel_pos_bias is not None :
            attn = attn + rel_pos_bias

        attn = attn.softmax (dim = -1)
        attn = self.attn_drop (attn)

        x = (attn @ v).transpose (1, 2).reshape (B, N, -1)
        x = self.proj (x)
        x = self.proj_drop (x)
        return x


class Block (nn.Module) :

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., init_values=None, act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 window_size=None, attn_head_dim=None) :
        super ().__init__ ()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads = num_heads, qkv_bias = qkv_bias, qk_scale = qk_scale,
            attn_drop = attn_drop, proj_drop = drop, window_size = window_size, attn_head_dim = attn_head_dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath (drop_path) if drop_path > 0. else nn.Identity ()
        self.norm2 = norm_layer (dim)
        mlp_hidden_dim = int (dim * mlp_ratio)
        self.mlp = Mlp (in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop)

        if init_values is not None and init_values > 0 :
            self.gamma_1 = nn.Parameter (init_values * torch.ones ((dim)), requires_grad = True)
            self.gamma_2 = nn.Parameter (init_values * torch.ones ((dim)), requires_grad = True)
        else :
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, rel_pos_bias=None) :
        if self.gamma_1 is None :
            x = x + self.drop_path (self.attn (self.norm1 (x), rel_pos_bias = rel_pos_bias))
            x = x + self.drop_path (self.mlp (self.norm2 (x)))
        else :
            x = x + self.drop_path (self.gamma_1 * self.attn (self.norm1 (x), rel_pos_bias = rel_pos_bias))
            x = x + self.drop_path (self.gamma_2 * self.mlp (self.norm2 (x)))
        return x


class PatchEmbed (nn.Module) :
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super ().__init__ ()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.projection = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim//2, kernel_size=1),
            nn.BatchNorm2d(embed_dim//2),
            nn.LeakyReLU(),
            nn.Conv2d(embed_dim//2, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU()
        )

    def forward(self, x):
        x = x.float()  # 将输入张量转换为 float 类型
        x = self.projection(x)  # (batch_size, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2)  # (batch_size, embed_dim, num_patches_h * num_patches_w)
        x = x.transpose(1, 2)  # (batch_size, num_patches_h * num_patches_w, embed_dim)
        return x
class RelativePositionBias (nn.Module) :

    def __init__(self, window_size, num_heads) :
        super ().__init__ ()
        self.window_size = window_size
        self.num_relative_distance = (2 * window_size[0] - 1) * (2 * window_size[1] - 1) + 3
        self.relative_position_bias_table = nn.Parameter (
            torch.zeros (self.num_relative_distance, num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        # cls to token & token 2 cls & cls to cls

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange (window_size[0])
        coords_w = torch.arange (window_size[1])
        coords = torch.stack (torch.meshgrid ([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten (coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute (1, 2, 0).contiguous ()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        relative_position_index = \
            torch.zeros (size = (window_size[0] * window_size[1] + 1,) * 2, dtype = relative_coords.dtype)
        relative_position_index[1 :, 1 :] = relative_coords.sum (-1)  # Wh*Ww, Wh*Ww
        relative_position_index[0, 0 :] = self.num_relative_distance - 3
        relative_position_index[0 :, 0] = self.num_relative_distance - 2
        relative_position_index[0, 0] = self.num_relative_distance - 1

        self.register_buffer ("relative_position_index", relative_position_index)

        # trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self) :
        relative_position_bias = \
            self.relative_position_bias_table[self.relative_position_index.view (-1)].view (
                self.window_size[0] * self.window_size[1] + 1,
                self.window_size[0] * self.window_size[1] + 1, -1)  # Wh*Ww,Wh*Ww,nH
        return relative_position_bias.permute (2, 0, 1).contiguous ()  # nH, Wh*Ww, Wh*Ww

class VisionTransformer(nn.Module):
    def __init__(self, patch_size, image_size, channel_num, num_classes, embed_dim=256, depth=6,
                 num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate = 0.1,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                 use_mean_pooling=True, init_scale=0.001):
        super().__init__()
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size, in_chans=channel_num, embed_dim=embed_dim)
        num_patches = max_seq_length=(image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1])   # self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = None
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=self.patch_embed.patch_shape, num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.use_rel_pos_bias = use_rel_pos_bias
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        # trunc_normal_(self.mask_token, std=.02)
        if isinstance(self.head, nn.Linear):
            trunc_normal_(self.head.weight, std=.02)
        self.apply(self._init_weights)
        self.fix_init_weight()

        if isinstance(self.head, nn.Linear):
            self.head.weight.data.mul_(init_scale)
            self.head.bias.data.mul_(init_scale)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        x = self.norm(x)
        if self.fc_norm is not None:
            t = x[:, 1:, :]
            return self.fc_norm(t.mean(1))
        else:
            return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)
        return x

    def get_intermediate_layers(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        features = []
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias)
            features.append(x)

        return features
