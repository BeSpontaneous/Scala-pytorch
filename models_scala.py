import math
import logging
from functools import partial
from collections import OrderedDict
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from enum import Enum
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, checkpoint_seq
from timm.models.layers import DropPath, trunc_normal_, lecun_normal_, to_2tuple
from timm.models.registry import register_model

_logger = logging.getLogger(__name__)


def _assert(condition: bool, message: str):
    assert condition, message
    

class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Attention(nn.Module):
    def __init__(self, smallest_ratio, largest_ratio, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.smallest_ratio = smallest_ratio
        self.largest_ratio = largest_ratio
        
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.dim = dim

    def forward(self, x, ratio):
        dim_channels = int(ratio * self.dim)
        
        B, N, C = x.shape
        
        if ratio == self.smallest_ratio or ratio == self.largest_ratio:
            weight_qkv = self.qkv.weight[:dim_channels*3, :dim_channels]
            bias_qkv = self.qkv.bias[:dim_channels*3]
        else:
            weight_qkv = self.qkv.weight[-dim_channels*3:, -dim_channels:]
            bias_qkv = self.qkv.bias[-dim_channels*3:]
        qkv = F.linear(input=x, weight=weight_qkv, bias=bias_qkv)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        if ratio == self.smallest_ratio or ratio == self.largest_ratio:
            weight_proj = self.proj.weight[:dim_channels, :dim_channels]
            bias_proj = self.proj.bias[:dim_channels]
        else:
            weight_proj = self.proj.weight[-dim_channels:, -dim_channels:]
            bias_proj = self.proj.bias[-dim_channels:]
        x = F.linear(input=x, weight=weight_proj, bias=bias_proj)
        x = self.proj_drop(x)
        return x



class LayerScale(nn.Module):
    def __init__(self, smallest_ratio, largest_ratio, dim, init_values=1e-5, inplace=False):
        super().__init__()
        
        self.smallest_ratio = smallest_ratio
        self.largest_ratio = largest_ratio
        
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
        
        self.dim = dim

    def forward(self, x, ratio):
        dim_channels = int(ratio * self.dim)
        
        if ratio == self.smallest_ratio or ratio == self.largest_ratio:
            gamma = self.gamma.data[:dim_channels]
        else:
            gamma = self.gamma.data[-dim_channels:]
        if self.inplace:
            x = x.mul_(gamma)
        else:
            x * gamma
        
        return x



class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            smallest_ratio,
            largest_ratio,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear

        self.smallest_ratio = smallest_ratio
        self.largest_ratio = largest_ratio
        
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features

    def forward(self, x, ratio):
        in_channels = int(ratio * self.in_features)
        hidden_channels = int(ratio * self.hidden_features)
        out_channels = int(ratio * self.out_features)
        
        if ratio == self.smallest_ratio or ratio == self.largest_ratio:
            weight1 = self.fc1.weight[:hidden_channels, :in_channels]
            bias1 = self.fc1.bias[:hidden_channels]
        else:
            weight1 = self.fc1.weight[-hidden_channels:, -in_channels:]
            bias1 = self.fc1.bias[-hidden_channels:]
        x = F.linear(input=x, weight=weight1, bias=bias1)
        x = self.act(x)
        x = self.drop1(x)
        
        if ratio == self.smallest_ratio or ratio == self.largest_ratio:
            weight2 = self.fc2.weight[:out_channels, :hidden_channels]
            bias2 = self.fc2.bias[:out_channels]
        else:
            weight2 = self.fc2.weight[-out_channels:, -hidden_channels:]
            bias2 = self.fc2.bias[-out_channels:]
        x = F.linear(input=x, weight=weight2, bias=bias2)
        x = self.drop2(x)
        
        return x


class Block(nn.Module):

    def __init__(
            self, smallest_ratio, largest_ratio, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.smallest_ratio = smallest_ratio
        self.largest_ratio = largest_ratio
        self.norm1 = norm_layer(dim)
        self.attn = Attention(smallest_ratio, largest_ratio, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(smallest_ratio, largest_ratio, dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(smallest_ratio, largest_ratio, in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(smallest_ratio, largest_ratio, dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.dim = dim
        self.init_values = init_values

    def forward(self, x, ratio):
        dim_channels = int(ratio * self.dim)
        
        residual = x

        if ratio == self.smallest_ratio or ratio == self.largest_ratio:
            weight_norm1 = self.norm1.weight[:dim_channels]
            bias_norm1 = self.norm1.bias[:dim_channels]
        else:
            weight_norm1 = self.norm1.weight[-dim_channels:]
            bias_norm1 = self.norm1.bias[-dim_channels:]
        x = F.layer_norm(x, [dim_channels], weight_norm1, bias_norm1)
        
        x = self.attn(x, ratio)
        if self.init_values:
            x = self.ls1(x, ratio)
        else:
            x = self.ls1(x)
        x = residual + self.drop_path1(x)
        
        residual = x

        if ratio == self.smallest_ratio or ratio == self.largest_ratio:
            weight_norm2 = self.norm2.weight[:dim_channels]
            bias_norm2 = self.norm2.bias[:dim_channels]
        else:
            weight_norm2 = self.norm2.weight[-dim_channels:]
            bias_norm2 = self.norm2.bias[-dim_channels:]
        x = F.layer_norm(x, [dim_channels], weight_norm2, bias_norm2)
        
        x = self.mlp(x, ratio)
        if self.init_values:
            x = self.ls2(x, ratio)
        else:
            x = self.ls2(x)
        x = residual + self.drop_path2(x)
        return x




class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    output_fmt: Format

    def __init__(
            self,
            smallest_ratio: float = 0.25, 
            largest_ratio: float = 1.0,
            img_size: Optional[int] = 224,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
    ):
        super().__init__()
        self.smallest_ratio = smallest_ratio
        self.largest_ratio = largest_ratio
        self.stride = patch_size
        self.patch_size = to_2tuple(patch_size)
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
        self.norm_layer = norm_layer
        self.embed_dim = embed_dim
        
    def forward(self, x, ratio):
        embed_dim_channels = int(ratio*self.embed_dim)
        
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                _assert(H == self.img_size[0], f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                _assert(W == self.img_size[1], f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            else:
                _assert(
                    H % self.patch_size[0] == 0,
                    f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]})."
                )
                _assert(
                    W % self.patch_size[1] == 0,
                    f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]})."
                )
        
        if ratio == self.smallest_ratio or ratio == self.largest_ratio:
            weight_proj = self.proj.weight[:embed_dim_channels, :, :, :]
            bias_proj = self.proj.bias[:embed_dim_channels]
        else:
            weight_proj = self.proj.weight[-embed_dim_channels:, :, :, :]
            bias_proj = self.proj.bias[-embed_dim_channels:]
        x = F.conv2d(x, weight_proj, bias_proj, self.stride)
        
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        
        if self.norm_layer:
            if ratio == self.smallest_ratio or ratio == self.largest_ratio:
                weight_norm = self.norm.weight[:embed_dim_channels]
                bias_norm = self.norm.bias[:embed_dim_channels]
            else:
                weight_norm = self.norm.weight[-embed_dim_channels:]
                bias_norm = self.norm.bias[-embed_dim_channels:]
            x = F.layer_norm(x, [embed_dim_channels], weight_norm, bias_norm)
        else:
            x = self.norm(x)
        
        return x


class VisionTransformer_scala(nn.Module):
    """ Vision Transformer

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    """

    def __init__(
            self, img_size=224, smallest_ratio=0.25, largest_ratio=1.0, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            global_pool (str): type of global pooling for final sequence (default: 'token')
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            init_values: (float): layer-scale init values
            class_token (bool): use class token
            fc_norm (Optional[bool]): pre-fc norm after pool, set if global_pool == 'avg' if None (default: None)
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            weight_init (str): weight init scheme
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            act_layer: (nn.Module): MLP activation layer
        """
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.num_classes = num_classes
        self.smallest_ratio = smallest_ratio
        self.largest_ratio = largest_ratio
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            smallest_ratio=smallest_ratio, largest_ratio=largest_ratio, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                smallest_ratio, largest_ratio, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.use_fc_norm = use_fc_norm

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def _pos_embed(self, x, ratio):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            if ratio == self.smallest_ratio:
                pos_embed = self.pos_embed[:,:,:int(ratio*self.num_features)]
                x = x + pos_embed
                if self.cls_token is not None:
                    cls_token = self.cls_token[:,:,:int(ratio*self.num_features)]
                    x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            elif ratio == self.largest_ratio:
                x = x + self.pos_embed
                if self.cls_token is not None:
                    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            else:
                pos_embed = self.pos_embed[:,:,-int(ratio*self.num_features):]
                x = x + pos_embed
                if self.cls_token is not None:
                    cls_token = self.cls_token[:,:,-int(ratio*self.num_features):]
                    x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if ratio == self.smallest_ratio:
                if self.cls_token is not None:
                    cls_token = self.cls_token[:,:,:int(ratio*self.num_features)]
                    x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                pos_embed = self.pos_embed[:,:,:int(ratio*self.num_features)]
                x = x + pos_embed
            elif ratio == self.largest_ratio:
                if self.cls_token is not None:
                    x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = x + self.pos_embed
            else:
                if self.cls_token is not None:
                    cls_token = self.cls_token[:,:,-int(ratio*self.num_features):]
                    x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)
                pos_embed = self.pos_embed[:,:,-int(ratio*self.num_features):]
                x = x + pos_embed

        return self.pos_drop(x)

    def forward_features(self, x, ratio):
        x = self.patch_embed(x, ratio)
        x = self._pos_embed(x, ratio)
        for blk in self.blocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint_seq(blk, x, ratio)
            else:
                x = blk(x, ratio)
        
        if self.use_fc_norm:
            x = self.norm(x)
        else:
            if ratio == self.smallest_ratio or ratio == self.largest_ratio:
                weight_norm = self.norm.weight[:int(ratio*self.embed_dim)]
                bias_norm = self.norm.bias[:int(ratio*self.embed_dim)]
            else:
                weight_norm = self.norm.weight[-int(ratio*self.embed_dim):]
                bias_norm = self.norm.bias[-int(ratio*self.embed_dim):]
            x = F.layer_norm(x, [int(ratio*self.embed_dim)], weight_norm, bias_norm)
        return x

    def forward_head(self, x, ratio, pre_logits: bool = False):
        if self.global_pool:
            x = x[:, self.num_prefix_tokens:].mean(dim=1) if self.global_pool == 'avg' else x[:, 0]
        
        if self.use_fc_norm:
            if ratio == self.smallest_ratio or ratio == self.largest_ratio:
                weight_fc_norm = self.fc_norm.weight[:int(ratio*self.embed_dim)]
                bias_fc_norm = self.fc_norm.bias[:int(ratio*self.embed_dim)]
            else:
                weight_fc_norm = self.fc_norm.weight[-int(ratio*self.embed_dim):]
                bias_fc_norm = self.fc_norm.bias[-int(ratio*self.embed_dim):]
            x = F.layer_norm(x, [int(ratio*self.embed_dim)], weight_fc_norm, bias_fc_norm)
        else:
            x = self.fc_norm(x)
        
        if pre_logits:
            x = x
        else:
            if ratio == self.smallest_ratio or ratio == self.largest_ratio:
                weight_head = self.head.weight[:, :int(ratio*self.embed_dim)]
                bias_head = self.head.bias[:]
            else:
                weight_head = self.head.weight[:, -int(ratio*self.embed_dim):]
                bias_head = self.head.bias[:]
            x = F.linear(input=x, weight=weight_head, bias=bias_head)

        return x

    def forward(self, x, ratio):
        x = self.forward_features(x, ratio)
        x = self.forward_head(x, ratio)
        return x


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    elif 'moco' in mode:
        return init_weights_vit_moco
    else:
        return init_weights_vit_timm




class DistilledVisionTransformer_scala(VisionTransformer_scala):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)

    def forward_features(self, x, ratio):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x, ratio)

        if ratio == 0.25 or ratio == 1.0:
            cls_token = self.cls_token[:,:,:int(ratio*self.num_features)]
            dist_token = self.dist_token[:,:,:int(ratio*self.num_features)]
            x = torch.cat((cls_token.expand(B, -1, -1), dist_token.expand(B, -1, -1), x), dim=1)
            pos_embed = self.pos_embed[:,:,:int(ratio*self.num_features)]
            x = x + pos_embed
        else:
            cls_token = self.cls_token[:,:,-int(ratio*self.num_features):]
            dist_token = self.dist_token[:,:,-int(ratio*self.num_features):]
            x = torch.cat((cls_token.expand(B, -1, -1), dist_token.expand(B, -1, -1), x), dim=1)
            pos_embed = self.pos_embed[:,:,-int(ratio*self.num_features):]
            x = x + pos_embed

        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x, ratio)

        if ratio == 0.25 or ratio == 1.0:
            weight_norm = self.norm.weight[:int(ratio*self.embed_dim)]
            bias_norm = self.norm.bias[:int(ratio*self.embed_dim)]
            x = F.layer_norm(x, [int(ratio*self.embed_dim)], weight_norm, bias_norm)
        else:
            weight_norm = self.norm.weight[-int(ratio*self.embed_dim):]
            bias_norm = self.norm.bias[-int(ratio*self.embed_dim):]
            x = F.layer_norm(x, [int(ratio*self.embed_dim)], weight_norm, bias_norm)
        
        return x[:, 0], x[:, 1]

    def forward(self, x, ratio):
        x, x_dist = self.forward_features(x, ratio)
        
        if ratio == 0.25 or ratio == 1.0:
            weight_head = self.head.weight[:, :int(ratio*self.embed_dim)]
            bias_head = self.head.bias[:]
            x = F.linear(input=x, weight=weight_head, bias=bias_head)
        else:
            weight_head = self.head.weight[:, -int(ratio*self.embed_dim):]
            bias_head = self.head.bias[:]
            x = F.linear(input=x, weight=weight_head, bias=bias_head)

        if ratio == 0.25 or ratio == 1.0:
            weight_head_dist = self.head_dist.weight[:, :int(ratio*self.embed_dim)]
            bias_head_dist = self.head_dist.bias[:]
            x_dist = F.linear(input=x_dist, weight=weight_head_dist, bias=bias_head_dist)
        else:
            weight_head_dist = self.head_dist.weight[:, -int(ratio*self.embed_dim):]
            bias_head_dist = self.head_dist.bias[:]
            x_dist = F.linear(input=x_dist, weight=weight_head_dist, bias=bias_head_dist)
        if self.training:
            return x, x_dist
        else:
            # during inference, return the average of both classifier predictions
            return (x + x_dist) / 2



@register_model
def deit_so_tiny_patch16_224_scala(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer_scala(
        patch_size=16, embed_dim=96, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_tiny_patch16_224_scala(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer_scala(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_small_patch16_224_scala(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer_scala(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_base_patch16_224_scala(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer_scala(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_tiny_distilled_patch16_224_scala(pretrained=False, pretrained_cfg=None, **kwargs):
    model = DistilledVisionTransformer_scala(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_small_distilled_patch16_224_scala(pretrained=False, pretrained_cfg=None, **kwargs):
    model = DistilledVisionTransformer_scala(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_base_distilled_patch16_224_scala(pretrained=False, pretrained_cfg=None, **kwargs):
    model = DistilledVisionTransformer_scala(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_base_patch16_384_scala(pretrained=False, pretrained_cfg=None, **kwargs):
    model = VisionTransformer_scala(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def deit_base_distilled_patch16_384_scala(pretrained=False, pretrained_cfg=None, **kwargs):
    model = DistilledVisionTransformer_scala(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model



if __name__ == "__main__":
    import argparse
    from fvcore.nn import FlopCountAnalysis, parameter_count_table
    parser = argparse.ArgumentParser(description='PyTorch resnet Training')
    args = parser.parse_args()

    args.num_classes = 1000
    with torch.no_grad():
        model = deit_small_distilled_patch16_224_scala()

        # for name, param in model.named_parameters():
            # print(name)
        
        tensor = (torch.rand(1, 3, 224, 224), 0.5)
        flops = FlopCountAnalysis(model, tensor)
        print("FLOPs: ", flops.total()/1e9)