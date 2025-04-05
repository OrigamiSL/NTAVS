# from loguru import logger
from functools import partial
import math
from typing import Any, Callable, Optional, Tuple, Union
import torch
import torch.nn as nn
# from torch.cuda.amp import autocast
# import torchaudio.transforms as audio_transforms
from einops import rearrange
from einops.layers.torch import Rearrange

# from models.checkpoints import register_model, build_mdl
from models.modeling.audio_backbone.CED.layers import AudioPatchEmbed, DropPath, Mlp, trunc_normal_, to_2tuple

class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.causal = causal

    def forward(self, x, if_return_attn_map = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q, k, v = qkv.unbind(
            0)  # make torchscript happy (cannot use tensor as tuple)

        match_map = q @ k.transpose(-2, -1)
        attn = match_map * self.scale # b, n_h, N, N
        # if mask is not None:
        # # Mask is a tensor of shape [B, T, T]
        # # Different from self.causal == True, the mask might be something like:
        # # [False, False, True]
        # # [False, False, True]
        # # [True, True, True]
        # # We use -inf to pad here, since if we would pad by any number, the entries at rows only containing
        # # [True, True, True] would lead to weights such as: [0.33,0.33,0.33], which is not correct
        # mask_value = torch.as_tensor(-float('inf'))
        # print(mask.shape, attn.shape)
        # attn = attn.masked_fill(mask, mask_value)
        if self.causal:
            mask_value = -torch.finfo(attn.dtype).max
            i, j = attn.shape[-2:]
            mask = torch.ones(i, j, device=q.device,
                              dtype=torch.bool).triu(j - i + 1)
            attn = attn.masked_fill(mask, mask_value)
        attn = attn.softmax(dim=-1)
        # Only for the case that a mask with all True entries on a row is passed.
        # attn = torch.nan_to_num(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if if_return_attn_map:
            return x, match_map
        else:
            return x, None

class Attention_for_visual(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        causal: bool = False,
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.v = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.causal = causal

    def forward(self, x, input_attn = None):
        B, N, C = x.shape
        v = self.v(x).reshape(B, N, self.num_heads,
                                  C // self.num_heads).transpose(1, 2)
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
        #                           C // self.num_heads).permute(2, 0, 3, 1, 4) 
        # q, k, v = qkv.unbind(
        #     0)  # make torchscript happy (cannot use tensor as tuple)

        attn =  input_attn * self.scale # b, n_h, N, N

        attn = attn.softmax(dim=-1)
        # Only for the case that a mask with all True entries on a row is passed.
        # attn = torch.nan_to_num(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attention_type: Callable = Attention,
        if_visual = False,
        attention_kwargs={},
        **kwargs,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.if_visual = if_visual
        if not if_visual:
            self.attn = attention_type(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   **attention_kwargs)
        else:
            self.attn = Attention_for_visual(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   **attention_kwargs)
        self.ls1 = nn.Identity()
        self.drop_path1 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2 = nn.Identity()
        self.drop_path2 = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, if_return_audio = False, if_return_attn_map = False, input_attn = None):
        norm_feature = self.norm1(x)
        if self.if_visual:
            attn_result = self.attn(norm_feature, input_attn = input_attn)
        else:
            attn_result, attn_map = self.attn(norm_feature, if_return_attn_map)
            
        x = x + self.drop_path1(self.ls1(attn_result))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        
        if if_return_attn_map and if_return_audio:
            return x, norm_feature, attn_map

        if if_return_audio:
            return x, norm_feature
        else:
            return x

class Attention_cross_audio_visual(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        causal: bool = False,
        vis_patches: int = 49,
        aud_patches: int = 24,
        bs: int = 8
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv_vis = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_vis = nn.Linear(dim, dim)

        self.qkv_aud = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_aud = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
       
        self.vis_patches = vis_patches
        self.aud_patches = aud_patches
        self.bs = bs

        self.weight = 0.9

    def make_attn(self, q, k, n_frame, v_attn_matrix = None, type_ = None, bs = None, mode = 'cross'):
        if type_ == 'vis':
            init_p = self.vis_patches
            res_p = self.aud_patches
        elif type_ == 'aud':
            init_p = self.aud_patches
            res_p = self.vis_patches

        attn_map = q @ k.transpose(-2, -1)
        attn =  attn_map * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attn_w = attn.contiguous().view(bs, init_p, self.num_heads, n_frame, n_frame)
        # if type_ == 'vis':
        #     attn_w = torch.einsum('abcde, abfcde->afcde', attn_w, v_attn_matrix)
        # elif type_ == 'aud':
        #     v_attn_matrix = v_attn_matrix.transpose(1, 2)
        #     attn_w = torch.einsum('abcde, abfcde->afcde', attn_w, v_attn_matrix)

        # attn_w = attn_w.contiguous().view(bs * res_p, self.num_heads, n_frame, n_frame)

        # reshape
        if mode == 'cross':
            attn = attn.contiguous().view(bs, init_p, self.num_heads, n_frame, n_frame)
            attn = attn.mean(1).unsqueeze(1)
            attn = attn.repeat(1, res_p, 1, 1, 1)
            attn = attn.contiguous().view(bs * res_p, self.num_heads, n_frame, n_frame)

        else:
            attn_w = attn.contiguous().view(bs, init_p, self.num_heads, n_frame, n_frame)
            attn_w = attn_w.mean(1).unsqueeze(1)
            attn_w = attn_w.repeat(1, res_p, 1, 1, 1)
            attn_w = attn_w.contiguous().view(bs * res_p, self.num_heads, n_frame, n_frame)

        return attn, attn_w

    def forward(self, vis, aud, bs, mode = 'cross'):
        b_vis, N, C = vis.shape
        b_aud, _, _ = aud.shape

        aud_size = b_aud // bs
        vis_size = b_vis // bs
        self.aud_patches = aud_size
       
        qkv_vis = self.qkv_vis(vis).reshape(b_vis, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q_vis, k_vis, v_vis = qkv_vis.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        # vm_vis = vw_vis.contiguous().view(bs, vis_size, self.num_heads, N, C // self.num_heads).permute(0, 3, 2, 1, 4)

        qkv_aud = self.qkv_aud(aud).reshape(b_aud, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q_aud, k_aud, v_aud = qkv_aud.unbind(0)
        # vm_aud = vw_aud.contiguous().view(bs, aud_size, self.num_heads, N, C // self.num_heads).permute(0, 3, 2, 1, 4)

        # v_attn_matrix = vm_vis @ vm_aud.transpose(-2, -1)
        # v_attn_matrix = v_attn_matrix.mean(dim = 1).permute(0, 2, 3, 1).unsqueeze(-1).unsqueeze(-1)
        # v_attn_matrix = v_attn_matrix.repeat(1, 1, 1, 1, N, N)

        attn_vis, weighted_audio = self.make_attn(q_vis, k_vis, N, type_ = 'vis', bs = bs, mode = mode)
        attn_aud, weighted_visual = self.make_attn(q_aud, k_aud, N, type_ = 'aud', bs = bs, mode = mode)

        if mode == 'cross':
            vis = (attn_aud @ v_vis).transpose(1, 2).reshape(b_vis, N, C)
            vis = self.proj_vis(vis)
            vis = self.proj_drop(vis)

            aud = (attn_vis @ v_aud).transpose(1, 2).reshape(b_aud, N, C)
            aud = self.proj_aud(aud)
            aud = self.proj_drop(aud)

        elif mode == 'self':
            # print(mode)
            attn_vis = self.weight * attn_vis + (1-self.weight) * weighted_visual
            vis = (attn_vis @ v_vis).transpose(1, 2).reshape(b_vis, N, C)
            vis = self.proj_vis(vis)
            vis = self.proj_drop(vis)

            attn_aud = self.weight * attn_aud + (1-self.weight) * weighted_audio
            aud = (attn_aud @ v_aud).transpose(1, 2).reshape(b_aud, N, C)
            aud = self.proj_aud(aud)
            aud = self.proj_drop(aud)

        return  vis, aud

class Attention_cross_audio_All_visual(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.,
        proj_drop=0.,
        causal: bool = False,
        vis_patches: list = None,
        aud_patches: int = 24,
        bs: int = 8
    ):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv_vis = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_vis = nn.Linear(dim, dim)

        self.qkv_aud = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_aud = nn.Linear(dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
       
        self.vis_patches = vis_patches
        self.aud_patches = aud_patches
        self.bs = bs

    def make_attn(self, q, k, n_frame, type_ = None, bs = None):
        if type_ == 'vis':
            init_p = self.vis_patches
            res_p = self.aud_patches
        elif type_ == 'aud':
            init_p = self.aud_patches
            res_p = self.vis_patches

        attn_map = q @ k.transpose(-2, -1)
        attn =  attn_map * self.scale 
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # reshape
        attn = attn.contiguous().view(bs, init_p, self.num_heads, n_frame, n_frame)
        attn = attn.mean(1).unsqueeze(1)
        attn = attn.repeat(1, res_p, 1, 1, 1)
        attn = attn.contiguous().view(bs * res_p, self.num_heads, n_frame, n_frame)

        return attn

    def forward(self, vis, aud, bs):
        b_vis, N, C = vis.shape
        b_aud, _, _ = aud.shape

        aud_size = b_aud // bs
        self.aud_patches = aud_size
       
        qkv_vis = self.qkv_vis(vis).reshape(b_vis, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q_vis, k_vis, v_vis = qkv_vis.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        qkv_aud = self.qkv_aud(aud).reshape(b_aud, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4) 
        q_aud, k_aud, v_aud = qkv_aud.unbind(0)

        attn_vis = self.make_attn(q_vis, k_vis, N, type_ = 'vis', bs = bs)

        attn_aud = self.make_attn(q_aud, k_aud, N, type_ = 'aud', bs = bs)

        vis = (attn_aud @ v_vis).transpose(1, 2).reshape(b_vis, N, C)
        vis = self.proj_vis(vis)
        vis = self.proj_drop(vis)

        aud = (attn_vis @ v_aud).transpose(1, 2).reshape(b_aud, N, C)
        aud = self.proj_aud(aud)
        aud = self.proj_drop(aud)

        return  vis, aud
    
class Block_cross_audio_visual(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attention_type: Callable = Attention,
        if_visual = False,
        visual_size = None,
        bs = None,
        attention_kwargs={},
        **kwargs,
    ):
        super().__init__()
        # self.norm1_vis = norm_layer(dim)
        # self.norm1_aud = norm_layer(dim)

        self.if_visual = if_visual
        
        self.attn = Attention_cross_audio_visual(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   vis_patches = int(visual_size**2),
                                   bs = bs,
                                   **attention_kwargs)
        
        self.ls1_vis = nn.Identity()
        self.drop_path1_vis = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_vis = norm_layer(dim)
        self.mlp_vis = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2_vis = nn.Identity()
        self.drop_path2_vis = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        
        self.ls1_aud = nn.Identity()
        self.drop_path1_aud = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_aud = norm_layer(dim)
        self.mlp_aud = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        
        self.ls2_aud = nn.Identity()
        self.drop_path2_aud = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        
        self.vis_return_norm = norm_layer(dim)
        self.aud_return_norm = norm_layer(dim)

    def forward(self, visual, audio, bs, mode = 'cross'):

        # norm_vis = self.norm1_vis(visual)
        # norm_aud = self.norm1_aud(audio)
        
        vis, aud = self.attn(visual, audio, bs, mode)
            
        visual = visual + self.drop_path1_vis(self.ls1_vis(vis))
        visual = visual + self.drop_path2_vis(self.ls2_vis(self.mlp_vis(self.norm2_vis(visual))))
        visual = self.vis_return_norm(visual)

        audio = audio + self.drop_path1_aud(self.ls1_aud(aud))
        audio = audio + self.drop_path2_aud(self.ls2_aud(self.mlp_aud(self.norm2_aud(audio))))
        audio = self.aud_return_norm(audio)

        return visual, audio

class Cross_audio_All_visual(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        attention_type: Callable = Attention,
        if_visual = False,
        visual_size = None,
        bs = None,
        attention_kwargs={},
        **kwargs,
    ):
        super().__init__()
    
        self.if_visual = if_visual
        
        self.attn = Attention_cross_audio_visual(dim,
                                   num_heads=num_heads,
                                   qkv_bias=qkv_bias,
                                   attn_drop=attn_drop,
                                   proj_drop=drop,
                                   vis_patches = int(visual_size**2),
                                   bs = bs,
                                   **attention_kwargs)
        # visual
        self.len_visual = len(visual_size)

        self.ls1_vis_list = nn.ModuleList()
        self.drop_path1_vis_list = nn.ModuleList()
        self.norm2_vis_list = nn.ModuleList()
        self.mlp_vis_list = nn.ModuleList()
        self.ls2_vis_list = nn.ModuleList()
        self.drop_path2_vis_list = nn.ModuleList()
        self.vis_return_norm_list = nn.ModuleList()

        for _ in range(self.len_visual):
            self.ls1_vis_list.append(nn.Identity())
            self.drop_path1_vis_list.append(DropPath(
            drop_path) if drop_path > 0. else nn.Identity())
            self.norm2_vis_list.append(norm_layer(dim))
            self.mlp_vis_list.append(Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop))
            self.ls2_vis_list.append(nn.Identity())
            self.drop_path2_vis_list.append(DropPath(
            drop_path) if drop_path > 0. else nn.Identity())
            self.vis_return_norm_list.append(norm_layer(dim))

        # self.ls1_vis = nn.Identity()
        # self.drop_path1_vis = DropPath(
        #     drop_path) if drop_path > 0. else nn.Identity()

        # self.norm2_vis = norm_layer(dim)
        # self.mlp_vis = Mlp(in_features=dim,
        #                hidden_features=int(dim * mlp_ratio),
        #                act_layer=act_layer,
        #                drop=drop)
        
        # self.ls2_vis = nn.Identity()
        # self.drop_path2_vis = DropPath(
        #     drop_path) if drop_path > 0. else nn.Identity()
        # self.vis_return_norm = norm_layer(dim)
        
        # audio
        self.ls1_aud = nn.Identity()
        self.drop_path1_aud = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2_aud = norm_layer(dim)
        self.mlp_aud = Mlp(in_features=dim,
                       hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer,
                       drop=drop)
        self.ls2_aud = nn.Identity()
        self.drop_path2_aud = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.aud_return_norm = norm_layer(dim)

    def forward(self, visual, audio, bs):

        vis, aud = self.attn(visual, audio, bs)

        for i in range(self.len_visual):   
            visual[i] = visual[i] + self.drop_path1_vis_list[i](self.ls1_vis_list[i](vis[i]))
            visual[i] = visual[i] + self.drop_path2_vis_list[i](self.ls2_vis_list[i](self.mlp_vis_list[i](self.norm2_vis_list[i](visual[i]))))
            visual[i] = self.vis_return_norm_list[i](visual[i])

        audio = audio + self.drop_path1_aud(self.ls1_aud(aud))
        audio = audio + self.drop_path2_aud(self.ls2_aud(self.mlp_aud(self.norm2_aud(audio))))
        audio = self.aud_return_norm(audio)

        return visual, audio


class AudioTransformer(nn.Module):

    def __init__(self,
                 outputdim=527,
                 patch_size=16,
                 patch_stride=16,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 init_bn: bool = True,
                 norm_layer=None,
                 act_layer=None,
                 init_values=None,
                 target_length=1012,
                 pooling='mean',
                 wavtransforms=None,
                 spectransforms=None,
                 time_patch_out: Optional[float] = None,
                 freq_patch_out: Optional[float] = None,
                 block_type=Block,
                 attention_type=Attention,
                 eval_avg='mean',
                 n_mels = 64,
                 **kwargs):
        super().__init__()
        assert pooling in ('mean', 'token', 'dm', 'logit')
        self.outputdim = outputdim
        self.pooling = pooling
        self.embed_dim = embed_dim
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        # self.n_mels = kwargs.get('n_mels', 64)
        self.n_mels = n_mels

        n_fft = kwargs.get('n_fft', 512)
        self.hop_size = kwargs.get('hop_size', 160)
        self.win_size = kwargs.get('win_size', 512)
        f_min = kwargs.get('f_min', 0)
        f_max = kwargs.get('f_max', 8000)
        self.center = kwargs.get('center', True)
        self.pad_last = kwargs.get('pad_last', True)
        self.eval_avg = eval_avg
        self.time_patch_out = time_patch_out
        self.freq_patch_out = freq_patch_out

        # self.front_end = FrontEnd(f_min=f_min,
        #                           f_max=f_max,
        #                           center=self.center,
        #                           win_size=self.win_size,
        #                           hop_size=self.hop_size,
        #                           sample_rate=16000,
        #                           n_fft=n_fft,
        #                           n_mels=self.n_mels)

        self.init_bn = nn.Sequential(
            Rearrange('b c f t -> b f c t'),
            torch.nn.BatchNorm2d(self.n_mels, momentum=0.01),
            Rearrange('b f c t -> b c f t'))
        self.target_length = target_length

        patch_stride = to_2tuple(self.patch_stride)[-1]
        # Allowed length in number of frames, otherwise the positional embedding will throw an error
        self.maximal_allowed_length = self.target_length

        # self.n_mels: f, target_length: t
        self.patch_embed = AudioPatchEmbed(input_size=(self.n_mels,
                                                       target_length),
                                           embed_dim=self.embed_dim,
                                           patch_size=self.patch_size,
                                           flatten=False,
                                           patch_stride=self.patch_stride)
        
        self.spectransforms = nn.Sequential(
        ) if spectransforms is None else spectransforms
        self.wavtransforms = nn.Sequential(
        ) if wavtransforms is None else wavtransforms

        if self.pooling == 'token':
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.token_pos_embed = nn.Parameter(
                torch.randn(1, embed_dim) * .02)

        self.time_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, 1, self.patch_embed.grid_size[1]) * .02)
        self.freq_pos_embed = nn.Parameter(
            torch.randn(1, embed_dim, self.patch_embed.grid_size[0], 1) * .02)
        
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.blocks = nn.Sequential(*[
            block_type(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                init_values=init_values,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                attention_type=attention_type,
            ) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.outputlayer = nn.Sequential(nn.LayerNorm(self.embed_dim),
                                         nn.Linear(self.embed_dim, outputdim))
        self.apply(self.init_weights)
        if hasattr(self, 'cls_token'):
            nn.init.normal_(self.cls_token, std=1e-6)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'time_pos_embed', 'cls_token', 'freq_pos_embed', 'token_pos_embed'
        }

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def interpolate_weight_128(self, n_mels):
        self.freq_pos_embed_128 = nn.Parameter(nn.functional.interpolate(
            self.freq_pos_embed, size = [n_mels // 16, 1], mode = 'bilinear'))
        
        new_norm = torch.nn.BatchNorm2d(n_mels, momentum=0.01)

        weight_bn = self.init_bn[1].weight.unsqueeze(0).unsqueeze(0)
        bias_bn = self.init_bn[1].bias.unsqueeze(0).unsqueeze(0)
        mean_bn = self.init_bn[1].running_mean.unsqueeze(0).unsqueeze(0)
        var_bn = self.init_bn[1].running_var.unsqueeze(0).unsqueeze(0)

        weight_bn = nn.functional.interpolate(
            weight_bn, size = [n_mels], mode = 'nearest')
        bias_bn = nn.functional.interpolate(
            bias_bn, size = [n_mels], mode = 'nearest')
        mean_bn = nn.functional.interpolate(
            mean_bn, size = [n_mels], mode = 'nearest')
        var_bn = nn.functional.interpolate(
            var_bn, size = [n_mels], mode = 'nearest')
        
        new_norm.weight = nn.Parameter(weight_bn.squeeze(0).squeeze(0))
        new_norm.bias = nn.Parameter(bias_bn.squeeze(0).squeeze(0))
        # new_norm.running_mean = nn.Parameter(mean_bn.squeeze(0).squeeze(0))
        # new_norm.running_var = nn.Parameter(var_bn.squeeze(0).squeeze(0))
        new_norm.running_mean = mean_bn.squeeze(0).squeeze(0)
        new_norm.running_var = var_bn.squeeze(0).squeeze(0)

        self.init_bn_128 = nn.Sequential(
            Rearrange('b c f t -> b f c t'),
            new_norm,
            Rearrange('b f c t -> b c f t'))
        
    def interpolate_weight_32(self, n_mels):
        self.freq_pos_embed_32 = nn.Parameter(nn.functional.interpolate(
            self.freq_pos_embed, size = [n_mels // 16, 1], mode = 'bilinear'))
        
        new_norm = torch.nn.BatchNorm2d(n_mels, momentum=0.01)

        weight_bn = self.init_bn[1].weight.unsqueeze(0).unsqueeze(0)
        bias_bn = self.init_bn[1].bias.unsqueeze(0).unsqueeze(0)
        mean_bn = self.init_bn[1].running_mean.unsqueeze(0).unsqueeze(0)
        var_bn = self.init_bn[1].running_var.unsqueeze(0).unsqueeze(0)

        weight_bn = nn.functional.interpolate(
            weight_bn, size = [n_mels], mode = 'nearest')
        bias_bn = nn.functional.interpolate(
            bias_bn, size = [n_mels], mode = 'nearest')
        mean_bn = nn.functional.interpolate(
            mean_bn, size = [n_mels], mode = 'nearest')
        var_bn = nn.functional.interpolate(
            var_bn, size = [n_mels], mode = 'nearest')
        
        new_norm.weight = nn.Parameter(weight_bn.squeeze(0).squeeze(0))
        new_norm.bias = nn.Parameter(bias_bn.squeeze(0).squeeze(0))
        new_norm.running_mean = mean_bn.squeeze(0).squeeze(0)
        new_norm.running_var = var_bn.squeeze(0).squeeze(0)

        self.init_bn_32 = nn.Sequential(
            Rearrange('b c f t -> b f c t'),
            new_norm,
            Rearrange('b f c t -> b c f t'))
    def interpolate_weight_16(self, n_mels):
        self.freq_pos_embed_16 = nn.Parameter(nn.functional.interpolate(
            self.freq_pos_embed, size = [n_mels // 16, 1], mode = 'bilinear'))
        
        new_norm = torch.nn.BatchNorm2d(n_mels, momentum=0.01)

        weight_bn = self.init_bn[1].weight.unsqueeze(0).unsqueeze(0)
        bias_bn = self.init_bn[1].bias.unsqueeze(0).unsqueeze(0)
        mean_bn = self.init_bn[1].running_mean.unsqueeze(0).unsqueeze(0)
        var_bn = self.init_bn[1].running_var.unsqueeze(0).unsqueeze(0)

        weight_bn = nn.functional.interpolate(
            weight_bn, size = [n_mels], mode = 'nearest')
        bias_bn = nn.functional.interpolate(
            bias_bn, size = [n_mels], mode = 'nearest')
        mean_bn = nn.functional.interpolate(
            mean_bn, size = [n_mels], mode = 'nearest')
        var_bn = nn.functional.interpolate(
            var_bn, size = [n_mels], mode = 'nearest')
        
        new_norm.weight = nn.Parameter(weight_bn.squeeze(0).squeeze(0))
        new_norm.bias = nn.Parameter(bias_bn.squeeze(0).squeeze(0))
        new_norm.running_mean = mean_bn.squeeze(0).squeeze(0)
        new_norm.running_var = var_bn.squeeze(0).squeeze(0)

        self.init_bn_16 = nn.Sequential(
            Rearrange('b c f t -> b f c t'),
            new_norm,
            Rearrange('b f c t -> b c f t'))
        
    def interpolate_all_scales(self, n_mels_list):
        # for scale in n_mels_list:
        #     self.interpolate_weight(scale)
        self.interpolate_weight_128(128)
        self.interpolate_weight_32(32)
        self.interpolate_weight_16(16)
    
    def interpolate_scale(self, n_mels):
        self.freq_pos_embed = nn.Parameter(nn.functional.interpolate(
        self.freq_pos_embed, size = [n_mels // 16, 1], mode = 'bilinear'))

        weight_bn = self.init_bn[1].weight.unsqueeze(0).unsqueeze(0)
        bias_bn = self.init_bn[1].bias.unsqueeze(0).unsqueeze(0)
        mean_bn = self.init_bn[1].running_mean.unsqueeze(0).unsqueeze(0)
        var_bn = self.init_bn[1].running_var.unsqueeze(0).unsqueeze(0)

        weight_bn = nn.functional.interpolate(
            weight_bn, size = [n_mels], mode = 'nearest')
        bias_bn = nn.functional.interpolate(
            bias_bn, size = [n_mels], mode = 'nearest')
        mean_bn = nn.functional.interpolate(
            mean_bn, size = [n_mels], mode = 'nearest')
        var_bn = nn.functional.interpolate(
            var_bn, size = [n_mels], mode = 'nearest')
        
        self.init_bn[1].weight = nn.Parameter(weight_bn.squeeze(0).squeeze(0))
        self.init_bn[1].bias = nn.Parameter(bias_bn.squeeze(0).squeeze(0))
        self.init_bn[1].running_mean = mean_bn.squeeze(0).squeeze(0)
        self.init_bn[1].running_var = var_bn.squeeze(0).squeeze(0)

    def forward_features(self, x: torch.Tensor, n_mels: int) -> torch.Tensor:
        x = self.patch_embed(x)
        b, c, f, t = x.shape
        x = x + self.time_pos_embed[:, :, :, :t]
        # if n_mels != 64:
        # self.freq_pos_embed = nn.Parameter(nn.functional.interpolate(
        # self.freq_pos_embed, size = [n_mels // 16, 1], mode = 'bilinear'))

        # if n_mels != 64:
        #     if n_mels == 128:
        #         x = x +  self.freq_pos_embed_128[:, :, :, :]
        #     elif n_mels == 32:
        #         x = x +  self.freq_pos_embed_32[:, :, :, :]
        #     elif n_mels == 16:
        #         x = x +  self.freq_pos_embed_16[:, :, :, :]

        # else:
        x = x + self.freq_pos_embed[:, :, :, :]  # Just to support __getitem__ in posembed

        # print('self.freq_pos_embed', self.freq_pos_embed.weight.shape)
        # print('self.freq_pos_embed_bias', self.freq_pos_embed.bias.shape)
        if self.training and self.time_patch_out is not None:
            x = drop_patches(x, dim=-1, frac=self.time_patch_out)
        if self.training and self.freq_patch_out is not None:
            x = drop_patches(x, dim=-2, frac=self.freq_patch_out)
        x = rearrange(x, 'b c f t -> b (f t) c')
        if self.pooling == 'token':
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            cls_token = cls_token + self.token_pos_embed
            x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x)
        x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor) -> torch.Tensor:

        return x.mean(1)
        # if self.pooling == 'token':
        #     x = x[:, 0]
        #     return self.outputlayer(x).sigmoid()
        # elif self.pooling == 'mean':
        #     x = x.mean(1)
        #     return self.outputlayer(x).sigmoid()
        # elif self.pooling == 'logit':
        #     x = x.mean(1)
        #     return self.outputlayer(x)
        # elif self.pooling == 'dm':
        #     # Unpack using the frequency dimension, which is constant
        #     x = rearrange(x,
        #                   'b (f t) d -> b f t d',
        #                   f=self.patch_embed.grid_size[0])
        #     # First poolin frequency, then sigmoid the (B T D) output
        #     x = self.outputlayer(x.mean(1)).sigmoid()
        #     return x.mean(1)
        # else:
        #     return x.mean(1)

    def load_state_dict(self, state_dict, strict=True):
        if 'time_pos_embed' in state_dict and hasattr(
                self, 'time_pos_embed'
        ) and self.time_pos_embed.shape != state_dict['time_pos_embed'].shape:
            # logger.debug(
            #     "Positional Embedding shape not the same with model, resizing!"
            # )
            print('Positional Embedding shape not the same with model, resizing!')
            self.change_pos_embedding(state_dict)
        super().load_state_dict(state_dict, strict=strict)

    def change_pos_embedding(self, state_dict):
        target_time_pos_embed_length = self.time_pos_embed.shape[-1]
        target_freq_pos_embed_length = self.freq_pos_embed.shape[-2]

        pretrained_time_pos_embed = state_dict['time_pos_embed']
        pretrained_freq_pos_embed = state_dict['freq_pos_embed']

        if target_time_pos_embed_length <= pretrained_time_pos_embed.shape[-1]:
            state_dict['time_pos_embed'] = pretrained_time_pos_embed[
                ..., :target_time_pos_embed_length]
        else:
            state_dict['time_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_time_pos_embed,
                size=(1, target_time_pos_embed_length),
                align_corners=False,
                mode='bilinear')
        if target_freq_pos_embed_length <= pretrained_freq_pos_embed.shape[-2]:
            state_dict[
                'freq_pos_embed'] = pretrained_freq_pos_embed[:, :, :
                                                              target_freq_pos_embed_length, :]
        else:
            state_dict['freq_pos_embed'] = torch.nn.functional.interpolate(
                pretrained_freq_pos_embed,
                size=(target_freq_pos_embed_length, 1),
                align_corners=False,
                mode='bilinear')

    def forward_spectrogram(self, x: torch.Tensor, n_mels: int) -> torch.Tensor:
        x = rearrange(x, 'b f t -> b 1 f t')
        # if n_mels!= 64:
        # weight_bn = self.init_bn[1].weight.unsqueeze(0).unsqueeze(0)
        # bias_bn = self.init_bn[1].bias.unsqueeze(0).unsqueeze(0)
        # mean_bn = self.init_bn[1].running_mean.unsqueeze(0).unsqueeze(0)
        # var_bn = self.init_bn[1].running_var.unsqueeze(0).unsqueeze(0)

        # weight_bn = nn.functional.interpolate(
        #     weight_bn, size = [n_mels], mode = 'nearest')
        # bias_bn = nn.functional.interpolate(
        #     bias_bn, size = [n_mels], mode = 'nearest')
        # mean_bn = nn.functional.interpolate(
        #     mean_bn, size = [n_mels], mode = 'nearest')
        # var_bn = nn.functional.interpolate(
        #     var_bn, size = [n_mels], mode = 'nearest')
        
        # self.init_bn[1].weight = nn.Parameter(weight_bn.squeeze(0).squeeze(0))
        # self.init_bn[1].bias = nn.Parameter(bias_bn.squeeze(0).squeeze(0))
        # self.init_bn[1].running_mean = mean_bn.squeeze(0).squeeze(0)
        # self.init_bn[1].running_var = var_bn.squeeze(0).squeeze(0)

        # print(n_mels, x.shape)
        # if n_mels!= 64:
        #     if n_mels == 128:
        #         x = self.init_bn_128(x)
        #     elif n_mels == 32:
        #         x = self.init_bn_32(x)
        #     elif n_mels == 16:
        #         x = self.init_bn_16(x)
        # else:

        # print('before', self.init_bn[1].weight[0:5])
        x = self.init_bn(x)
        # print('after', self.init_bn[1].weight[0:5])
        
        # print(1)
        # print(self.init_bn[1].weight.requires_grad)
        # print(self.init_bn[1].bias.requires_grad)
        # print(self.init_bn[1].running_mean.requires_grad)
        # print(self.init_bn[1].running_var.requires_grad)

        if x.shape[-1] > self.maximal_allowed_length:
            splits = x.split(self.target_length, -1)

            if splits[-1].shape[-1] < self.target_length:
                if self.pad_last:
                    pad = torch.zeros(*x.shape[:-1],
                                      self.target_length,
                                      device=x.device)
                    pad[..., :splits[-1].shape[-1]] = splits[-1]
                    splits = torch.stack((*splits[:-1], pad), dim=0)
                else:
                    splits = torch.stack(splits[:-1], dim=0)
            else:
                splits = torch.stack(splits[:-1], dim=0)
            n_splits = len(splits)
            x = rearrange(splits, 'spl b c f t-> (spl b) c f t')
            x = self.forward_head(self.forward_features(x, n_mels))
            x = rearrange(x, '(spl b) d -> spl b d', spl=n_splits)
            if self.eval_avg == 'mean':
                x = x.mean(0)
            elif self.eval_avg == 'max':
                x = x.max(0)[0]
            else:
                raise ValueError(
                    f'Unknown Eval average function ({self.eval_avg})')

        else:
            x = self.forward_features(x, n_mels)
            # x = self.forward_head(x)
        return x

    # def forward(self, x):
    #     if self.training:
    #         x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
    #     x = self.front_end(x)
    #     if self.training:
    #         x = self.spectransforms(x)
    #     x = self.forward_spectrogram(x)
    #     return x

    def forward(self, x, n_mels = 64):
        x = self.forward_spectrogram(x, n_mels = n_mels)
        return x

# mini
def build_CED(device, n_mels = 64, model_type = 'mini'):

    if model_type == 'mini':
        print('build_CED_type: mini')
        model_kwargs = dict(patch_size=16,
                            embed_dim=256,
                            depth=12,
                            num_heads=4,
                            mlp_ratio=4,
                            outputdim=527,
                            n_mels = 64)
        
        pretrained_pth = './pretrained/audiotransformer_mini_mAP_4896.pt'
    elif model_type == 'base':
        print('build_CED_type: base')
        model_kwargs = dict(patch_size=16,
                        embed_dim=768,
                        depth=12,
                        num_heads=12,
                        mlp_ratio=4,
                        outputdim=527,
                        n_mels = 64)
        
        pretrained_pth = '/home/lhg/work/ssd_new/AVSBench_all/CED_new_cross/NAVS_GTAVM_behind/audiotransformer_base_mAP_4999.pt'

    model_kwargs = dict(model_kwargs)
    mdl = AudioTransformer(**model_kwargs)
    dump = torch.load(pretrained_pth, map_location='cpu')
    if 'model' in dump:
        dump = dump['model']
    if n_mels == 64:
        mdl.load_state_dict(dump, strict=False)
    # else:
    #     mdl.load_state_dict(dump, strict=False)
    #     mdl.interpolate_scale(n_mels)
        
    # mdl.interpolate_all_scales([128, 32, 16])
    # if n_mels == 64:
    #     mdl.load_state_dict(dump, strict=False)
    # else:
    #     new_dump = {}
    #     for k, v in dump.items():
    #         if 'init_bn' in k or 'freq_pos_embed' in k:
    #            pass
    #         else:
    #            new_dump.update({k: v}) 
    #     mdl.load_state_dict(dump, strict=False)
    #     mdl.interpolate_weight(n_mels)

    return mdl.to(device)