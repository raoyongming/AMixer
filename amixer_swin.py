import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from timm.models.registry import register_model



class AdaSpatialMLP(nn.Module):
    def __init__(self, dim, n=196, k=1, r=4, num_heads=4, mode='softmax', post_proj=True, pre_proj=True, relative=True):
        super().__init__()

        k = int(num_heads * k)

        self.k = k

        self.relative = relative
        if not relative:
            self.weight_bank = nn.Parameter(torch.randn(k, n, n, dtype=torch.float32) * 0.02)
        else:
            h = w = int(math.sqrt(n))
            assert h * w == n
            self.weight_bank = nn.Parameter(torch.randn(k, (2 * h - 1) * (2 * w - 1), dtype=torch.float32) * 0.02)  # 2*Wh-1 * 2*Ww-1, nH

            # get pair-wise relative position index for each token inside the window
            self.init_window_size = h
            relative_position_index = self.build_relative_index(h)
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

    @staticmethod
    def build_relative_index(w):
        # get pair-wise relative position index for each token inside the window
        h = w
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
        return relative_position_index  
        
        
    def forward(self, x, attn_mask=None, ape=None):

        B, H, W, C = x.shape
        x = x.reshape(B, H*W, C)
        n = H * W

        if ape is not None:
            mix_policy = self.adapter(x + ape).reshape(B, n, self.k, self.num_heads)
        else:
            mix_policy = self.adapter(x).reshape(B, n, self.k, self.num_heads)

        if not self.relative:
            weight_bank = self.weight_bank
        else:
            weight_bank = self.weight_bank[:, self.relative_position_index.view(-1)].view(self.k, n, n)  # k,Wh*Ww,Wh*Ww
        
        assert self.mode == 'linear-softmax'
        weight = torch.einsum('bnkh,knm->bnmh', mix_policy, weight_bank)
        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = weight.view(B // nW, nW, n, n, self.num_heads) + attn_mask.unsqueeze(-1).unsqueeze(0)
            attn = attn.view(-1, n, n, self.num_heads)
        weight = torch.softmax(weight, dim=1)
       
        if self.pre_proj is not None:
            x = self.pre_proj(x)
        
        x = x.reshape(B, n, self.num_heads, -1)
        x = torch.einsum('bnhc,bnmh->bmhc', x, weight).reshape(B,n,C//self.squeeze)

        if self.post_proj is not None:
            x = self.post_proj(x)

        x = x.reshape(B, H, W, C)
        return x


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

def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class AMixerBlock(nn.Module):
    r""" Swin MLP Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, 
                 window_size=7, shift_size=0, init_values=0.001,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 act_layer=nn.GELU,
                 ada=False, mode='linear-softmax', post_proj=True, pre_proj=True, 
                 relative=True, squeeze=1, k=None, downsample=False):
        super().__init__()
        self.dim = dim
        self.downsample = downsample
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size

        if input_resolution[0] > window_size and input_resolution[1] > window_size:
            self.shift_size = shift_size
        else:
            self.shift_size = 0
        self.mlp_ratio = mlp_ratio

        self.ada = ada

        if not ada:
            raise NotImplementedError
        else:
            self.spatial_mixer = AdaSpatialMLP(dim, window_size*window_size, 
                                            k=k, num_heads=self.num_heads,
                                            mode=mode, post_proj=post_proj, pre_proj=pre_proj, 
                                            relative=relative, squeeze=squeeze)

        self.gamma1 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)
        self.gamma2 = nn.Parameter(init_values * torch.ones((dim)),requires_grad=True)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def update_window_size(self, w):
        assert self.ada
        H, W = self.input_resolution
        H = H * w // self.window_size
        W = W * w // self.window_size
        self.input_resolution = (H, W)
        self.window_size = w
        if self.shift_size > 0:
            self.shift_size = w // 2
        self.spatial_mlp.update_window_size(w)
        self.padding = [self.window_size - self.shift_size, self.shift_size,
                        self.window_size - self.shift_size, self.shift_size]  # P_l,P_r,P_t,P_b

    def forward(self, x, attn_mask=None, ape=None):
        if not self.ada:
            x = x + self.drop_path(self.gamma1 * self.spatial_mixer(self.norm1(x)))
        else:
            shortcut = x
            x = self.norm1(x)
            B, H, W, C = x.shape

            # pad feature maps to multiples of window size
            pad_l = pad_t = 0
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
            _, Hp, Wp, _ = x.shape

            if self.shift_size > 0:
                shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            else:
                shifted_x = x
                attn_mask = None
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
            x_windows = x_windows.view(-1, self.window_size, self.window_size, C)  # nW*B, window_size, window_size, C
            x_windows = self.spatial_mixer(x_windows, attn_mask, ape)
            shifted_x = window_reverse(x_windows, self.window_size, Hp, Wp)

            if self.shift_size > 0:
                x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = shifted_x

            if pad_r > 0 or pad_b > 0:
                x = x[:, :H, :W, :].contiguous()
            
            x = shortcut + self.drop_path(self.gamma1 * x)

        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x



class BasicLayer(nn.Module):
    """ A basic Swin MLP layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., drop=0., drop_path=0.,
                 downsample=None, use_checkpoint=False, init_values=0.001, policy_ape=True,
                 ada=False, mode='linear-softmax', post_proj=True, pre_proj=True, mlp_baseline=False, 
                 relative=True, k=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.window_size = window_size
        self.shift_size = window_size // 2
        self.ada = ada

        if policy_ape:
            self.ape = nn.Parameter(torch.randn(window_size*window_size, dim)*0.02)
            print('[Basic Layer] using ape for policy adapter')
        else:
            self.ape = None
            print('[Basic Layer] NO ape for policy adapter')

        # build blocks
        self.blocks = nn.ModuleList([
            AMixerBlock(dim=dim, input_resolution=input_resolution,
                         num_heads=num_heads, window_size=window_size,
                         shift_size=0 if (i % 2 == 0) else window_size // 2,
                         mlp_ratio=mlp_ratio,
                         drop=drop, init_values=init_values,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         ada=ada, mode=mode, mlp_baseline=mlp_baseline, pre_proj=pre_proj, 
                         post_proj=post_proj, relative=relative, k=k)
            for i in range(depth)])

        # patch merging layer
        if downsample:
            self.downsample = PatchMerging(input_resolution, dim=dim)
        else:
            self.downsample = None


        attn_mask = None
        if self.ada and self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            H += (self.window_size - H % self.window_size) % self.window_size
            W += (self.window_size - W % self.window_size) % self.window_size

            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            if 'softmax' in mode:
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            else:
                raise NotImplementedError

            attn_mask = attn_mask.transpose(1, 2)
        
        self.register_buffer("attn_mask", attn_mask)
    
    def update_window_size(self, w):
        for blk in self.blocks:
            blk.update_window_size(w)

    def forward(self, x):

        B, H, W, C = x.shape

        attn_mask = self.attn_mask


        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask, self.ape)
            else:
                x = blk(x, attn_mask, self.ape)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class PatchMerging(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, input_resolution=56, dim=64):
        super().__init__()
        self.input_resolution = input_resolution
        self.proj = nn.Linear(dim * 4, dim * 2)
        self.norm = nn.LayerNorm(dim * 4)

    def forward(self, x):
        B, H, W, C = x.size()
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
        B, H, W, C = x.size()
        x = x.view(B, H//2, 2, W//2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H//2, W//2, 4*C)
        x = self.norm(x)
        x = self.proj(x)
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.patch_h = img_size[0] // patch_size[0]
        self.patch_w = img_size[1] // patch_size[1]

        self.patches_resolution = (self.patch_h, self.patch_w)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).permute(0, 2, 3, 1)
        x = self.norm(x)
        return x



class AMixer(nn.Module):
    r""" AMixer

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin MLP layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        drop_rate (float): Dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self,
                 img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., drop_rate=0., drop_path_rate=0.1,
                 ape=False, patch_norm=True, use_bn=False, init_values=0.001,
                 use_checkpoint=False, ada=False, mode='linear-softmax', post_proj=True, pre_proj=True,   
                 policy_ape=True,
                 mlp_baseline=False, relative=True, squeeze=1, k=None, pretrained=None, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, use_bn=use_bn)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        hs = [patches_resolution[0] // (2 ** i_layer) for i_layer in range(self.num_layers)]

        if isinstance(window_size, list):
            window_sizes = [min(w, h) for w,h in zip(window_size, hs)]
        else: 
            window_sizes = [min(window_size, h) for h in hs]
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                    patches_resolution[1] // (2 ** i_layer)),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_sizes[i_layer],
                                mlp_ratio=self.mlp_ratio, policy_ape=policy_ape,
                                drop=drop_rate, use_bn=use_bn, init_values=init_values,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                downsample=True if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint,
                                ada=ada, 
                                mode=mode, pre_proj=pre_proj, post_proj=post_proj, 
                                mlp_baseline=mlp_baseline, relative=relative, squeeze=squeeze, k=k)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.pretrained = pretrained
        print('[AMixer pretrained is]', self.pretrained)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def update_window_size(self, w):
        for layer in self.layers:
            layer.update_window_size(w)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = self.patch_embed(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = self.norm(x)  # B H W C
        x = x.mean(dim=(1, 2))
        return self.head(x)



@register_model
def amixer_t(**kwargs):
    model = AMixer(
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=96, depths=[2, 2, 10, 2], num_heads=[3, 6, 12, 24], mlp_ratio=3, policy_ape=False,
        window_size=[7,7,14,7], drop_path_rate=0.2, ada=True, mode='linear-softmax', post_proj=True, pre_proj=True, relative=True, k=1.5
    )
    return model


@register_model
def amixer_s(**kwargs):
    model = AMixer(
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=96, depths=[2, 2, 26, 2], num_heads=[3, 6, 12, 24], mlp_ratio=3, policy_ape=False,
        window_size=[7,7,14,7], drop_path_rate=0.35, ada=True, mode='linear-softmax', post_proj=True, pre_proj=True, relative=True, k=1.5
    )
    return model

@register_model
def amixer_b(**kwargs):
    model = AMixer(
        img_size=224, patch_size=4, in_chans=3, num_classes=1000,
        embed_dim=128, depths=[2, 2, 26, 2], num_heads=[4, 8, 16, 32], mlp_ratio=3, policy_ape=False,
        window_size=[7,7,14,7], drop_path_rate=0.5, ada=True, mode='linear-softmax', post_proj=True, pre_proj=True, relative=True, k=1.5
    )
    return model


