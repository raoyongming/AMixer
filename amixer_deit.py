import math
from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdaSpatialMLP(nn.Module):
    def __init__(self, dim, n=196, k=16, r=4, num_heads=1, mode='softmax', post_proj=False, pre_proj=False, relative=False):
        super().__init__()

        self.relative = relative
        if not relative:
            self.weight_bank = nn.Parameter(torch.randn(k, n, n, dtype=torch.float32) * 0.02)
        else:
            h = w = int(math.sqrt(n))
            assert h * w == n
            # define a parameter table of relative position bias
            self.weight_bank = nn.Parameter(torch.randn(k, (2 * h - 1) * (2 * w - 1), dtype=torch.float32) * 0.02)  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(h)
            coords_w = torch.arange(w)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += h - 1  # shift to start from 0
            relative_coords[:, :, 1] += w - 1
            relative_coords[:, :, 0] *= 2 * w - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.adapter = nn.Sequential(
            nn.Linear(dim, dim//r),
            nn.GELU(),
            nn.Linear(dim//r, k * num_heads)
        )

        self.k = k
        self.dim = dim
        self.num_heads = num_heads
        self.n = n
        self.mode = mode


        if pre_proj:
            self.pre_proj = nn.Linear(dim, dim)
        else:
            self.pre_proj = None

        if post_proj:
            self.post_proj = nn.Linear(dim, dim)
        else:
            self.post_proj = None

        print('[AdaSpatialMLP layer] k=%d, num_heads=%d, mode=%s, pos/pre-proj=%s/%s, relative=%s' % (k, self.num_heads, mode, pre_proj, post_proj, relative))
        
        
    def forward(self, x, mask=None):
        B, n, C = x.shape
        mix_policy = self.adapter(x).reshape(B, n, self.k, self.num_heads)

        if not self.relative:
            weight_bank = self.weight_bank
        else:
            weight_bank = self.weight_bank[:, self.relative_position_index.view(-1)].view(self.k, n, n)  # k,Wh*Ww,Wh*Ww
        
        if self.mode == 'softmax':
            mix_policy = torch.softmax(mix_policy, dim=2)
            weight = torch.einsum('bnkh,knm->bnmh', mix_policy, weight_bank)
        elif self.mode == 'linear':
            weight = torch.einsum('bnkh,knm->bnmh', mix_policy, weight_bank)
        elif self.mode == 'softmax-softmax':
            mix_policy = torch.softmax(mix_policy, dim=2)
            weight = torch.einsum('bnkh,knm->bnmh', mix_policy, weight_bank)
            weight = torch.softmax(weight, dim=1)
        elif self.mode == 'linear-softmax':
            weight = torch.einsum('bnkh,knm->bnmh', mix_policy, weight_bank)
            weight = torch.softmax(weight, dim=1)
        elif self.mode == 'linear-sigmoid':
            weight = torch.einsum('bnkh,knm->bnmh', mix_policy, weight_bank)
            weight = torch.sigmoid(weight)
        elif self.mode == 'linear-normalize':
            weight = torch.einsum('bnkh,knm->bnmh', mix_policy, weight_bank)
            weight = torch.nn.functional.normalize(weight, dim=1, p=2)
        elif self.mode == 'sigmoid':
            mix_policy = torch.sigmoid(mix_policy)
            weight = torch.einsum('bnkh,knm->bnmh', mix_policy, weight_bank)
        else:
            raise NotImplementedError

        if self.pre_proj is not None:
            x = self.pre_proj(x)
        
        x = x.reshape(B, n, self.num_heads, -1)
        x = torch.einsum('bnhc,bnmh->bmhc', x, weight).reshape(B,n,C)

        if self.post_proj is not None:
            x = self.post_proj(x)
        return x

class SpatialMLP(nn.Module):
    def __init__(self, dim, n=196, num_heads=1, relative=False, mode='linear', post_proj=False, pre_proj=False):
        super().__init__()

        self.relative = relative
        if not relative:
            self.weight = nn.Parameter(torch.randn(num_heads, n, n, dtype=torch.float32) * 0.02)
        else:
            h = w = int(math.sqrt(n))
            assert h * w == n
            # define a parameter table of relative position bias
            self.weight = nn.Parameter(torch.randn(num_heads, (2 * h - 1) * (2 * w - 1), dtype=torch.float32) * 0.02)  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(h)
            coords_w = torch.arange(w)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
            coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
            relative_coords[:, :, 0] += h - 1  # shift to start from 0
            relative_coords[:, :, 1] += w - 1
            relative_coords[:, :, 0] *= 2 * w - 1
            relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
            self.register_buffer("relative_position_index", relative_position_index)

        self.dim = dim
        self.n = n
        self.num_heads = num_heads
        self.mode = mode
        assert mode in ['softmax', 'linear', 'normalize']

        if pre_proj:
            self.pre_proj = nn.Linear(dim, dim)
        else:
            self.pre_proj = None

        if post_proj:
            self.post_proj = nn.Linear(dim, dim)
        else:
            self.post_proj = None

        print('[SpatialMLP layer] num_heads=%d, mode=%s, pos/pre-proj=%s/%s, relative=%s' % (self.num_heads, mode, pre_proj, post_proj, relative))
        
    
    def forward(self, x):

        if self.pre_proj is not None:
            x = self.pre_proj(x)
        
        B, n, C = x.shape
        
        x = x.reshape(B, n, self.num_heads, -1)

        if not self.relative:
            weight = self.weight
        else:
            weight = self.weight[:, self.relative_position_index.view(-1)].view(self.num_heads, n, n)  # k,Wh*Ww,Wh*Ww
        
        if self.mode == 'softmax':
            weight = torch.softmax(weight, dim=1)
        elif self.mode == 'linear':
            pass
        elif self.mode == 'normalize':
            weight = torch.nn.functional.normalize(weight, dim=1, p=2)
        else:
            raise NotImplemented
        
        x = torch.einsum('bnhc,hnm->bmhc', x, weight).reshape(B,n,C//self.squeeze)

        if self.post_proj is not None:
            x = self.post_proj(x)

        return x

class MLPBlock(nn.Module):

    def __init__(self, dim, n, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
        num_heads=1, k=16, mode='softmax', ada=False, init_values=1e-5, post_proj=False, pre_proj=False, relative=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        if ada:
            print('using ada mlp')
            self.spatil_mlp = AdaSpatialMLP(dim, n=n, k=k, num_heads=num_heads, mode=mode, post_proj=post_proj, pre_proj=pre_proj, relative=relative)
        else:
            print('using normal mlp')
            self.spatil_mlp = SpatialMLP(dim, n=n, num_heads=num_heads, mode=mode, post_proj=post_proj, pre_proj=pre_proj, relative=relative)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.gamma1 * self.spatil_mlp(self.norm1(x)))
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=partial(nn.LayerNorm, eps=1e-6), patch_norm=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_h = img_size[0] // patch_size[0]
        self.patch_w = img_size[1] // patch_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if patch_norm:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PureMLP(nn.Module):
    
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 mlp_ratio=4., representation_size=None, uniform_drop=False,
                 drop_rate=0., drop_path_rate=0., norm_layer=None, init_values=0.001,
                 dropcls=0, num_heads=1, k=16, mode='softmax', ada=False, post_proj=False, pre_proj=False, relative=False, **kwargs):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            hybrid_backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.patch_h, self.patch_embed.patch_w, embed_dim))

        h = img_size // patch_size
        n = h * h

        if uniform_drop:
            print('using uniform droppath with expect rate', drop_path_rate)
            dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        else:
            print('using linear droppath with expect rate', drop_path_rate * 0.5)
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # dpr = [drop_path_rate for _ in range(depth)]  # stochastic depth decay rule
        
        self.blocks = nn.ModuleList([
            MLPBlock(
                dim=embed_dim, n=n, mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[i], norm_layer=norm_layer, 
                num_heads=num_heads,k=k, mode=mode, ada=ada, init_values=init_values, pre_proj=pre_proj, post_proj=post_proj,
                relative=relative)
            for i in range(depth)])
         
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if dropcls > 0:
            print('dropout %.2f before classifier' % dropcls)
            self.final_dropout = nn.Dropout(p=dropcls)
        else:
            self.final_dropout = nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if hasattr(m, 'weight'):
                if m.weight is not None:
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        if x.shape[1:3] == self.pos_embed.shape[1:3]:
            x = x + self.pos_embed
        else:
            pos = F.interpolate(self.pos_embed.permute(0, 3, 1, 2), size=x.shape[1:3], mode='bilinear', align_corners=True).permute(0, 2, 3, 1)
            x = x + pos

        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x).mean(dim=1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.final_dropout(x)
        x = self.head(x)
        return x

@register_model
def amixer_deit_s(**kwargs):
    model = PureMLP(
        img_size=224, 
        patch_size=16, embed_dim=384, depth=17, mlp_ratio=3, init_values=0.0001,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), num_heads=8, k=16, mode='linear-softmax', 
        ada=True, post_proj=True, pre_proj=True, relative=True, drop_path_rate=0.1,
    )
    return model

