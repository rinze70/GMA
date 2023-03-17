import torch
from torch import nn, einsum
from torch.nn import Module, Dropout
from einops import rearrange

from attention import PositionEncodingSine, ChannelAttention

class CAttention(ChannelAttention):
    def __init__(self, 
                 d_model, 
                 nhead,
                 use_dropout=False, 
                 attention_dropout=0.1):
        super().__init__(use_dropout=use_dropout, attention_dropout=attention_dropout)

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        self.pos_encoding = PositionEncodingSine(d_model=d_model)

    def forward(self, inp):
        """ Multi-head(group) scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
        Returns:
            queried_values: (N, L, H, D)
        """
        bs = inp.size(0)
        inp = rearrange(self.pos_encoding(inp), 'n c h w -> n (h w) c')

        # multi-head attention
        queries = self.q_proj(inp).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        keys = self.k_proj(inp).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nlhc->ndch", queries, keys)

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        # queried_values = torch.einsum("ndch,nshc->nshd", A, values)

        return A
    
class ChannelAggregate(nn.Module):
    def __init__(
        self,
        d_model,
        nhead = 4,
    ):
        super().__init__()
        self.dim = d_model // nhead
        self.nhead = nhead

        # self.pos_encoding = PositionEncodingSine(d_model=d_model)

        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, attn, fmap):
        bs, c, h, w = fmap.shape
        value = rearrange(fmap, 'n c h w -> n (h w) c')

        values = self.v_proj(value).view(bs, -1, self.nhead, self.dim)

        queried_values = torch.einsum("ndch,nshc->nshd", attn, values)
        message = queried_values.contiguous()
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([value, message], dim=2))
        message = self.norm2(message)
        message = rearrange(message, 'n (h w) c -> n c h w', h=h, w=w)

        return fmap + self.gamma * message
    
if __name__ == "__main__":
    att = CAttention(d_model=128, nhead=1)
    channel_aggregator = ChannelAggregate(d_model=128, nhead=1)
    inp = torch.randn(2, 128, 40, 90)
    motion_features = torch.randn(2, 128, 40, 90)
    attention = att(inp)
    motion_features_global = channel_aggregator(attention, motion_features)

    print(motion_features_global.shape)