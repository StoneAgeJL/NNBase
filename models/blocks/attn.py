import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce, repeat

from .vit_layer import DropPath, Mlp

class DotProduct(nn.Module):
    """ Explicit dot product layer for pretty flops count printing.
    """
    def __init__(self, scale=None):
        super().__init__()
        self.scale = scale
    
    def forward(self, x, y):
        if self.scale is not None:
            x = x * self.scale
        out = x @ y

        return out

    def extra_repr(self) -> str:
        return 'scale={}'.format(
            self.scale
        )

class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)

        ##
        self.scaled_dot_product = DotProduct(scale=head_dim ** -0.5) # 1 / sqrt(k)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.dot_product = DotProduct()
        
        ##
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # -> 3,B,num_head,N,C//num_head
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        attn = self.scaled_dot_product(q, k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        
        x = self.dot_product(attn, v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x, attn

class TimeSformerBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.temporal_norm1 = norm_layer(dim)
        self.temporal_attn = SelfAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.temporal_fc = nn.Linear(dim, dim)
        
        self.norm1 = norm_layer(dim)
        self.attn = SelfAttn(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, T, N, return_attention=False):
        """
        x (Tensor): shape (B, (1 + T N), C)
        T (int): input time length
        N (int): input num patches
        """
        init_cls_tokens, x = torch.split(x, [1, T*N], dim=1)

        # Temporal attention
        xt = rearrange(x, 'b (t n) c -> (b n) t c', t=T, n=N)
        xt, time_attn = self.temporal_attn(self.temporal_norm1(xt))
        xt = self.drop_path(xt)
        xt = rearrange(self.temporal_fc(xt), '(b n) t c -> b (t n) c', t=T, n=N)

        x = x + xt

        # Spatial attention
        cls_token = init_cls_tokens.expand(-1, T, -1) # expand cls_token over time dimension
        cls_token = rearrange(cls_token, 'b t c -> (b t) () c')
        xs = rearrange(x, 'b (t n) c -> (b t) n c', t=T, n=N)

        xs = torch.cat([cls_token, xs], dim=1)
        xs, space_attn = self.attn(self.norm1(xs))

        if return_attention:
            return (time_attn, space_attn)

        xs = self.drop_path(xs)

        cls_token, xs = torch.split(xs, [1, N], dim=1)
        cls_token = reduce(cls_token, '(b t) () c -> b () c', 'mean', t=T) # average cls tkn over time dimension
        xs = rearrange(xs, '(b t) n c -> b (t n) c', t=T, n=N)

        x = torch.cat([init_cls_tokens, x], dim=1) + torch.cat([cls_token, xs], dim=1)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x

