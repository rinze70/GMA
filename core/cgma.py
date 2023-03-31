import torch
from torch import nn

from davit import ConvPosEnc, Mlp
from timm.models.layers import DropPath
from einops import rearrange

class CAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, norm_layer=nn.LayerNorm,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.norm1 = norm_layer(dim)
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

        self.cpe = ConvPosEnc(dim=dim, k=3)

    def forward(self, x):
        size = x.shape[-2:]
        x = rearrange(x, 'n c h w -> n (h w) c')
        B, N, C = x.shape

        x = self.cpe(x, size)
        x = self.norm1(x)

        qk = self.qk(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 3, B, h, N, C 
        q, k = qk[0], qk[1]

        q = q * self.scale
        attention = q.transpose(-1, -2) @ k
        attention = attention.softmax(dim=-1)
        # x = (attention @ q.transpose(-1, -2)).transpose(-1, -2)
        # x = x.transpose(1, 2).reshape(B, N, C)
        # x = self.proj(x)
        return attention

class CAggregate(nn.Module):
    def __init__(
        self,
        dim,
        heads = 4,
        mlp_ratio=4.,
        qkv_bias=False,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        ffn=True,
    ):
        super().__init__()
        self.heads = heads

        self.cpe = nn.ModuleList([ConvPosEnc(dim=dim, k=3),
                                  ConvPosEnc(dim=dim, k=3)])
        self.ffn = ffn
        self.norm1 = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.v = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)

        self.gamma = nn.Parameter(torch.zeros(1))

        if self.ffn:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer)

    def forward(self, attention, x):
        size = x.shape[-2:]
        x = rearrange(x, 'n c h w -> n (h w) c')
        B, N, C = x.shape

        x = self.cpe[0](x, size)
        cur = self.norm1(x)
        v = self.v(cur).reshape(B, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4) # 3, B, h, N, C 
        v = v[0]

        cur = (attention @ v.transpose(-1, -2)).transpose(-1, -2)
        cur = cur.transpose(1, 2).reshape(B, N, C)
        cur = self.proj(cur)

        x = x + self.drop_path(cur)

        x = self.cpe[1](x, size)

        if self.ffn:
            x = x + self.drop_path(self.mlp(self.norm2(x))) * self.gamma
        x = rearrange(x, 'n (h w) c -> n c h w', h=size[0], w=size[1])
        return x

    
if __name__ == '__main__':
    attn = CAttention(dim=128, num_heads=4)
    agg = CAggregate(dim=128, heads=4)
    h, w = 368//8, 496//8

    inp = torch.randn(1, 128, h, w)
    # inp = rearrange(inp, 'n c h w -> n (h w) c')
    attention = attn(inp)
    out = agg(attention, inp)
    # out = rearrange(out, 'n (h w) c -> n c h w', h=h, w=w)
    print(out.shape)
