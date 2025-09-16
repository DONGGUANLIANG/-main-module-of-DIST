
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
from timm.models.layers import DropPath, trunc_normal_
from torch.autograd import Variable
import numpy as np
import copy

# =====================
# helpers
# =====================

def exists(val):
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding=0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# =====================
# basic blocks
# =====================

class RearrangeImage(nn.Module):
    def forward(self, x):
        return rearrange(x, 'b (h w) c -> b c h w', h=int(math.sqrt(x.shape[1])))

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim=None, dropout=0.):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, out_dim=None, dropout=0., se=0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.out_dim = out_dim
        self.dim = dim

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if out_dim is None:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, dim),) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(nn.Linear(inner_dim, out_dim),) if project_out else nn.Identity()

        self.to_out_dropout = nn.Dropout(dropout)

        self.se = se
        if self.se > 0:
            self.se_layer = SE(dim)

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        if self.se:
            out = self.se_layer(out)
        out = self.to_out_dropout(out)
        if self.out_dim is not None and self.out_dim != self.dim:
            out = out + v.squeeze(1)
        return out

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, attn_out_dim=None, ff_out_dim=None, dropout=0., se=0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.attn_out_dim = attn_out_dim
        self.dim = dim
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, out_dim=attn_out_dim, dropout=dropout, se=se)),
                PreNorm(dim if not attn_out_dim else attn_out_dim, FeedForward(dim if not attn_out_dim else attn_out_dim, mlp_dim, out_dim=ff_out_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            if self.attn_out_dim is not None and self.dim != self.attn_out_dim:
                x = attn(x)
            else:
                x = attn(x) + x
            x = ff(x) + x
        return x

class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):  # x: [B, N, C]
        x = torch.transpose(x, 1, 2)  # [B, C, N]
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        x = x * y.expand_as(x)
        x = torch.transpose(x, 1, 2)  # [B, N, C]
        return x

# =====================
# Original backbones (kept for compatibility)
# =====================

class T2TViT(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth=None, heads=None, mlp_dim=None, pool='cls',
                 channels=3, dim_head=64, dropout=0., emb_dropout=0., transformer=None, t2t_layers=((7, 4), (3, 2), (3, 2))):
        super().__init__()
        assert pool in {'cls', 'mean'}

        layers = []
        layer_dim = [channels * 7 * 7, 64 * 3 * 3]
        output_image_size = image_size

        for i, (kernel_size, stride) in enumerate(t2t_layers):
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)
            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size=kernel_size, stride=stride, padding=stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim=layer_dim[i], heads=1, depth=1, dim_head=64, mlp_dim=64, attn_out_dim=64, ff_out_dim=64, dropout=dropout) if not is_last else nn.Identity(),
            ])

        layers.append(nn.Linear(layer_dim[1], dim))
        self.to_patch_embedding = nn.Sequential(*layers)

        self.pos_embedding = nn.Parameter(torch.randn(1, output_image_size ** 2 + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)])
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :n + 1]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        return x

class DiT(nn.Module):
    def __init__(self, *, image_size, num_classes, dim, depth=None, heads=None, mlp_dim=None, pool='cls',
                 channels=3, dim_head=64, dropout=0., emb_dropout=0., patch_emb='share', time_emb=False,
                 pos_emb='share', use_scale=False, transformer=None, t2t_layers=((7, 4), (3, 2), (3, 2))):
        super().__init__()
        assert pool in {'cls', 'mean'}

        layers = []
        layer_dim = [channels * 7 * 7, 64 * 3 * 3]
        output_image_size = image_size

        for i, (kernel_size, stride) in enumerate(t2t_layers):
            is_first = i == 0
            is_last = i == (len(t2t_layers) - 1)
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)
            layers.extend([
                RearrangeImage() if not is_first else nn.Identity(),
                nn.Unfold(kernel_size=kernel_size, stride=stride, padding=stride // 2),
                Rearrange('b c n -> b n c'),
                Transformer(dim=layer_dim[i], heads=1, depth=1, dim_head=64, mlp_dim=64, attn_out_dim=64, ff_out_dim=64, dropout=dropout) if not is_last else nn.Identity(),
            ])

        layers.append(nn.Linear(layer_dim[1], dim))

        self.patch_emb = patch_emb
        if self.patch_emb == 'share':
            self.to_patch_embedding = nn.Sequential(*layers)
        elif self.patch_emb == 'isolated':
            layers_before = copy.deepcopy(layers)
            layers_after = copy.deepcopy(layers)
            self.to_patch_embedding_before = nn.Sequential(*layers_before)
            self.to_patch_embedding_after = nn.Sequential(*layers_after)

        self.pos_emb = pos_emb
        if self.pos_emb == 'share':
            self.pos_embedding_before_and_after = nn.Parameter(torch.randn(1, output_image_size ** 2, dim))
        elif self.pos_emb == 'sin':
            self.pos_embedding_before_and_after = self.get_sinusoid_encoding(output_image_size ** 2, dim)
        elif self.pos_emb == 'isolated':
            self.pos_embedding_before = nn.Parameter(torch.randn(1, output_image_size ** 2, dim))
            self.pos_embedding_after = nn.Parameter(torch.randn(1, output_image_size ** 2, dim))

        self.time_emb = time_emb
        if self.time_emb:
            self.time_embedding = nn.Parameter(torch.randn(2, dim))

        self.use_scale = use_scale
        if self.use_scale:
            self.scale = nn.Sequential(nn.Linear(2 * output_image_size ** 2, 2))
            self.softmax = nn.Softmax(dim=1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)])
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def get_sinusoid_encoding(self, n_position, d_hid):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, before_x, after_x):
        if self.patch_emb == 'share':
            before_x = self.to_patch_embedding(before_x)
            after_x = self.to_patch_embedding(after_x)
        elif self.patch_emb == 'isolated':
            before_x = self.to_patch_embedding_before(before_x)
            after_x = self.to_patch_embedding_after(after_x)

        b, n, _ = before_x.shape

        if self.pos_emb == 'share' or self.pos_emb == 'sin':
            pos = self.pos_embedding_before_and_after
            if pos.device != before_x.device:
                pos = pos.to(before_x.device)
            before_x = before_x + pos
            after_x = after_x + pos
        elif self.pos_emb == 'isolated':
            before_x = before_x + self.pos_embedding_before
            after_x = after_x + self.pos_embedding_after

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        if not self.use_scale:
            x = torch.cat((cls_tokens, before_x, after_x), dim=1)
        else:
            x = torch.cat((before_x, after_x), dim=1)

        if self.time_emb:
            if not self.use_scale:
                x[:, 1:(n + 1)] += self.time_embedding[0]
                x[:, (n + 1):] += self.time_embedding[1]
            else:
                x[:, :n] += self.time_embedding[0]
                x[:, n:] += self.time_embedding[1]

        x = self.dropout(x)
        x = self.transformer(x)

        if not self.use_scale:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        else:
            scale = self.scale(x.mean(dim=-1))
            scale = self.softmax(scale)
            x = scale[0, 1] * x[:, n:, :].mean(dim=1) + scale[0, 0] * x[:, :n, :].mean(dim=1)

        x = self.to_latent(x)
        return x

class ViT_v2(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim=None, depth=None, heads=None, mlp_dim=None, pool='cls',
                 channels=3, dim_head=64, dropout=0., emb_dropout=0., time_emb=False, pos_emb='share', use_scale=False, transformer=None):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_emb = pos_emb
        if self.pos_emb == 'share':
            self.pos_embedding_before_and_after = nn.Parameter(torch.randn(1, num_patches, dim))
        elif self.pos_emb == 'isolated':
            self.pos_embedding_before = nn.Parameter(torch.randn(1, num_patches, dim))
            self.pos_embedding_after = nn.Parameter(torch.randn(1, num_patches, dim))

        self.time_emb = time_emb
        if self.time_emb:
            self.time_embedding = nn.Parameter(torch.randn(2, dim))

        self.use_scale = use_scale
        if self.use_scale:
            self.scale = nn.Sequential(nn.Linear(2 * num_patches, 2))
            self.softmax = nn.Softmax(dim=1)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        if not exists(transformer):
            assert all([exists(depth), exists(heads), exists(mlp_dim)])
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        else:
            self.transformer = transformer

        self.pool = pool
        self.to_latent = nn.Identity()

    def forward(self, img_before, img_after):
        before_x = self.to_patch_embedding(img_before)
        after_x = self.to_patch_embedding(img_after)

        b, n, _ = before_x.shape

        if self.pos_emb == 'share':
            before_x = before_x + self.pos_embedding_before_and_after
            after_x = after_x + self.pos_embedding_before_and_after
        elif self.pos_emb == 'isolated':
            before_x = before_x + self.pos_embedding_before
            after_x = after_x + self.pos_embedding_after

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        if not self.use_scale:
            x = torch.cat((cls_tokens, before_x, after_x), dim=1)
        else:
            x = torch.cat((before_x, after_x), dim=1)

        if self.time_emb:
            if not self.use_scale:
                x[:, 1:(n + 1)] += self.time_embedding[0]
                x[:, (n + 1):] += self.time_embedding[1]
            else:
                x[:, :n] += self.time_embedding[0]
                x[:, n:] += self.time_embedding[1]

        x = self.dropout(x)
        x = self.transformer(x)

        if not self.use_scale:
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        else:
            scale = self.scale(x.mean(dim=-1))
            scale = self.softmax(scale)
            x = scale[0, 1] * x[:, n:, :].mean(dim=1) + scale[0, 0] * x[:, :n, :].mean(dim=1)

        x = self.to_latent(x)
        return x

# =====================
# New: DIST components (EiT2T + ST + AFFC)
# =====================

class LocalSelfAttention(nn.Module):
    def __init__(self, dim, window_size=7, heads=4, dim_head=32, dropout=0.):
        super().__init__()
        self.window_size = window_size
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x2d = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        x2d = F.pad(x2d, (0, pad_w, 0, pad_h), value=0)
        Hp, Wp = x2d.shape[-2:]
        x_win = rearrange(x2d, 'b c (nh ws1) (nw ws2) -> (b nh nw) (ws1 ws2) c', ws1=ws, ws2=ws)
        x_out = self.attn(x_win)
        x_out = rearrange(x_out, '(b nh nw) (ws1 ws2) c -> b c (nh ws1) (nw ws2)', b=B, nh=Hp//ws, nw=Wp//ws, ws1=ws, ws2=ws)
        x_out = x_out[:, :, :H, :W]
        x_out = rearrange(x_out, 'b c h w -> b (h w) c')
        return x_out

class GlobalSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
    def forward(self, x):
        return self.attn(x)

class MultiScaleConvFuse(nn.Module):
    def __init__(self, dim, scales=(3,5,7)):
        super().__init__()
        self.scales = scales
        self.convs = nn.ModuleList([
            nn.Conv2d(dim, dim, k, padding=k//2, groups=1, bias=False) for k in scales
        ])
        for m in self.convs:
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        self.proj = nn.Linear(dim * len(scales), dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x2d = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        outs = []
        for conv in self.convs:
            feat = conv(x2d)
            outs.append(rearrange(feat, 'b c h w -> b (h w) c'))
        x_cat = torch.cat(outs, dim=-1)
        return self.proj(x_cat)

class EiT2TBlock(nn.Module):
    def __init__(self, in_dim, token_dim, image_size, k_unfold=3, s_unfold=2,
                 local_ws=7, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.image_size = image_size
        self.ms = MultiScaleConvFuse(in_dim)
        self.local = LocalSelfAttention(in_dim, window_size=local_ws, heads=max(1, heads//2), dim_head=max(16, dim_head//2), dropout=dropout)
        self.global_ = GlobalSelfAttention(in_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.alpha = nn.Parameter(torch.tensor(0.5))
        self.unfold = nn.Unfold(kernel_size=k_unfold, stride=s_unfold, padding=k_unfold//2)
        self.proj = nn.Linear(in_dim * (k_unfold**2), token_dim)

    def forward(self, img):
        B, C, H, W = img.shape
        x = rearrange(img, 'b c h w -> b (h w) c')
        x_ms = self.ms(x, H, W)
        x_l = self.local(x_ms, H, W)
        x_g = self.global_(x_ms)
        x = self.alpha * x_l + (1 - self.alpha) * x_g
        x2d = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        patches = self.unfold(x2d)
        patches = rearrange(patches, 'b d n -> b n d')
        tokens = self.proj(patches)
        return tokens

class DynamicPositionEmbedding(nn.Module):
    def __init__(self, dim, H, W):
        super().__init__()
        self.H, self.W = H, W
        self.conv = nn.Sequential(
            Rearrange('b n c -> b c n'),
            Rearrange('b c (h w) -> b c h w', h=H, w=W),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            Rearrange('b c h w -> b (h w) c')
        )
    def forward(self, tokens):
        return self.conv(tokens)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = nn.Parameter(torch.randn(2, dim))
    def forward(self, tokens, t: int):
        return tokens + self.emb[t].view(1, 1, -1)

class STModule(nn.Module):
    def __init__(self, dim, H, W, use_sin_time=False):
        super().__init__()
        self.dpe = DynamicPositionEmbedding(dim, H, W)
        self.te = TimeEmbedding(dim)
        self.use_sin_time = use_sin_time
    def forward(self, tokens_before, tokens_after):
        pb = tokens_before + self.dpe(tokens_before)
        pa = tokens_after  + self.dpe(tokens_after)
        pb = self.te(pb, t=0)
        pa = self.te(pa, t=1)
        return pb, pa

class AFFCHead(nn.Module):
    def __init__(self, dim, num_classes=2):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fuse_fc = nn.Linear(dim * 2, 2)
        self.cls = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, num_classes))
        self.softmax = nn.Softmax(dim=1)

    def gap(self, x):
        x = x.transpose(1, 2)
        x = self.pool(x).squeeze(-1)
        return x

    def forward(self, feat_before, feat_after):
        gb = self.gap(feat_before)
        ga = self.gap(feat_after)
        logits = self.fuse_fc(torch.cat([gb, ga], dim=-1))
        w = self.softmax(logits)
        fused = w[:, :1] * gb + w[:, 1:] * ga
        diff  = ga - gb
        out   = torch.cat([fused, diff], dim=-1)
        return self.cls(out), w

class DISTModel(nn.Module):
    def __init__(self, image_size=224, in_ch=6, dim=256, depth=16, heads=16, dim_head=64, mlp_dim=512,
                 k_unfold=3, s_unfold=2, local_ws=7, num_classes=2, dropout=0.):
        super().__init__()
        self.H = self.W = image_size

        self.eit2t_before = EiT2TBlock(in_dim=in_ch, token_dim=dim, image_size=image_size,
                                       k_unfold=k_unfold, s_unfold=s_unfold,
                                       local_ws=local_ws, heads=heads, dim_head=dim_head, dropout=dropout)
        self.eit2t_after  = EiT2TBlock(in_dim=in_ch, token_dim=dim, image_size=image_size,
                                       k_unfold=k_unfold, s_unfold=s_unfold,
                                       local_ws=local_ws, heads=heads, dim_head=dim_head, dropout=dropout)

        self.grid = self.H // s_unfold
        self.st = STModule(dim=dim, H=self.grid, W=self.grid)

        self.encoder = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)
        self.head = AFFCHead(dim=dim, num_classes=num_classes)
        self._init_pos = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, img_before, img_after):
        tb = self.eit2t_before(img_before)
        ta = self.eit2t_after(img_after)
        tb, ta = self.st(tb, ta)
        tb = self.encoder(tb + self._init_pos)
        ta = self.encoder(ta + self._init_pos)
        logits, w = self.head(tb, ta)
        return logits, w

# =====================
# Loss
# =====================

class FocalLoss(nn.Module):
    def __init__(self, class_num=2, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            alpha = torch.ones(class_num)
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction='none', weight=self.alpha.to(logits.device))
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# =====================
# Wrapper
# =====================

class DiT_basic(nn.Module):
    def __init__(self,
                 basic_model='dist',  # 'dist' | 't2t' | 'vit'
                 patch_emb='isolated',
                 time_emb=True,
                 pos_emb='share',
                 use_scale=False,
                 pool='cls',
                 loss_f='focal',  # 'focal' or 'ce'
                 input_type='both',
                 label_type='MP',
                 output_type='probability',
                 in_ch=6):
        super(DiT_basic, self).__init__()
        print('basic_model: %s\\ninput_type: %s\\nlabel_type: %s\\nloss_f: %s' % (basic_model, input_type, label_type, loss_f))

        self.input_type = input_type
        self.label_type = label_type
        self.output_type = output_type

        if basic_model == 'dist':
            self.net = DISTModel(image_size=224, in_ch=in_ch, dim=256, depth=16, heads=16, dim_head=64, mlp_dim=512)
            self.fc = nn.Identity()
        elif basic_model == 't2t':
            if input_type == 'both':
                self.net = DiT(image_size=224, num_classes=2, dim=256, pool=pool, channels=3, dim_head=64,
                                dropout=0., emb_dropout=0., patch_emb=patch_emb, pos_emb=pos_emb, time_emb=time_emb,
                                use_scale=use_scale, transformer=Transformer(dim=256, depth=16, heads=16, dim_head=64, mlp_dim=512),
                                t2t_layers=((7, 4), (3, 2), (3, 2)))
            else:
                self.net = T2TViT(image_size=224, num_classes=1, dim=256, pool=pool, channels=3, dim_head=64,
                                   dropout=0., emb_dropout=0., transformer=Transformer(dim=256, depth=16, heads=16, dim_head=64, mlp_dim=512),
                                   t2t_layers=((7, 4), (3, 2), (3, 2)))
            self.fc = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 2))
        elif basic_model == 'vit':
            self.net = ViT_v2(image_size=224, patch_size=16, num_classes=2, dim=256, pool=pool, channels=3, dim_head=64,
                               dropout=0., emb_dropout=0., time_emb=time_emb, pos_emb=pos_emb, use_scale=use_scale,
                               transformer=Transformer(dim=256, depth=16, heads=16, dim_head=64, mlp_dim=512))
            self.fc = nn.Sequential(nn.LayerNorm(256), nn.Linear(256, 2))
        else:
            raise ValueError('unknown basic_model: %s' % basic_model)

        weight = torch.tensor([1.0, 1.0])
        if loss_f == 'ce':
            self.loss = nn.CrossEntropyLoss(weight=weight)
        elif loss_f == 'focal':
            self.loss = FocalLoss(class_num=2, alpha=weight, gamma=2)
        else:
            raise ValueError('unknown loss_f')
        self.softmax = nn.Softmax(dim=1)

        self._initialize_weights()

    def _initialize_weights(self):
        print('initialize weights for network!')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)

    def forward(self, before_x, after_x, labels_MP=None, labels_LNM=None):
        if isinstance(self.net, DISTModel):
            out, fuse_w = self.net(before_x, after_x)
        else:
            if self.input_type == 'before':
                out = self.net(before_x)
            elif self.input_type == 'after':
                out = self.net(after_x)
            elif self.input_type == 'both':
                out = self.net(before_x, after_x)
            out = out.view(out.shape[0], -1)
            out = self.fc(out)

        if labels_MP is not None or labels_LNM is not None:
            labels_MP = labels_MP.view(-1)
            labels_LNM = labels_LNM.view(-1)
            prob = self.softmax(out)
            if hasattr(self, 'label_type') and self.label_type == 'MP':
                cls_loss = self.loss(out, labels_MP)
                acc = (prob.argmax(1) == labels_MP).float().mean().item()
            else:
                cls_loss = self.loss(out, labels_LNM)
                acc = (prob.argmax(1) == labels_LNM).float().mean().item()
            return cls_loss, acc
        else:
            if self.output_type == 'probability':
                return self.softmax(out)[:, 1]
            elif self.output_type == 'score':
                return out
            return None

if __name__ == '__main__':
    B, C, H, W = 2, 6, 224, 224
    before = torch.randn(B, C, H, W)
    after  = torch.randn(B, C, H, W)
    model = DiT_basic(basic_model='dist', in_ch=C, output_type='score')
    with torch.no_grad():
        out = model(before, after)
        print('Output shape:', out.shape if hasattr(out, 'shape') else type(out))
