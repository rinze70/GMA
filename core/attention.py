import math
import torch
from torch import nn
from torch.nn import Module, Dropout

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]
    
def elu_feature_map(x):
    return torch.nn.functional.elu(x) + 1


class LinearAttention(Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()
    
class FullAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nshd->nlsh", queries, keys)
        if kv_mask is not None:
            QK.masked_fill_(~(q_mask[:, :, None, None] * kv_mask[:, None, :, None]), float('-inf'))

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("nlsh,nshd->nlhd", A, values)

        return queried_values.contiguous()

class ChannelAttention(Module):
    def __init__(self, use_dropout=False, attention_dropout=0.1):
        super().__init__()
        self.use_dropout = use_dropout
        self.dropout = Dropout(attention_dropout)

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-head(group) scaled dot-product attention, a.k.a full attention.
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
        Returns:
            queried_values: (N, L, H, D)
        """

        # Compute the unnormalized attention and apply the masks
        QK = torch.einsum("nlhd,nlhc->ndch", queries, keys)

        # Compute the attention and the weighted average
        softmax_temp = 1. / queries.size(3)**.5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)
        if self.use_dropout:
            A = self.dropout(A)

        queried_values = torch.einsum("ndch,nshc->nshd", A, values)

        return queried_values.contiguous()
    
class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        if attention == 'linear':
            self.attention = LinearAttention()
        elif attention == 'full':
            self.attention = FullAttention()
        elif attention == 'channel':
            self.attention = ChannelAttention()
        else:
            raise NotImplementedError
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

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


# class LocalFeatureTransformer(nn.Module):
#     """A Local Feature Transformer (LoFTR) module."""

#     def __init__(self, config):
#         super(LocalFeatureTransformer, self).__init__()

#         self.config = config
#         self.d_model = config['d_model']
#         self.nhead = config['nhead']
#         self.layer_names = config['layer_names']
#         encoder_layer = LoFTREncoderLayer(config['d_model'], config['nhead'], config['attention'])
#         self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
#         self._reset_parameters()

#     def _reset_parameters(self):
#         for p in self.parameters():
#             if p.dim() > 1:
#                 nn.init.xavier_uniform_(p)

#     def forward(self, feat0, feat1, mask0=None, mask1=None):
#         """
#         Args:
#             feat0 (torch.Tensor): [N, L, C]
#             feat1 (torch.Tensor): [N, S, C]
#             mask0 (torch.Tensor): [N, L] (optional)
#             mask1 (torch.Tensor): [N, S] (optional)
#         """

#         assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

#         for layer, name in zip(self.layers, self.layer_names):
#             if name == 'self':
#                 feat0 = layer(feat0, feat0, mask0, mask0)
#                 feat1 = layer(feat1, feat1, mask1, mask1)
#             elif name == 'cross':
#                 feat0 = layer(feat0, feat1, mask0, mask1)
#                 feat1 = layer(feat1, feat0, mask1, mask0)
#             else:
#                 raise KeyError

#         return feat0, feat1
    
class AttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1, layer_name='self', attention = 'full'):
        super(AttentionLayer, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.layer_name = layer_name

        self.encoder_layer = LoFTREncoderLayer(d_model=d_model, nhead=nhead, attention=attention)


    def forward(self, feat_c0, feat_c1):

        assert self.d_model == feat_c0.size(2), "the feature number of src and transformer must be equal"

        if self.layer_name == 'self':
            feat0 = self.encoder_layer(feat_c0, feat_c0)
            feat1 = self.encoder_layer(feat_c1, feat_c1)

        elif self.layer_name == 'cross':
            feat0 = self.encoder_layer(feat_c0, feat_c1)
            feat1 = self.encoder_layer(feat_c1, feat_c0)

        else:
            raise KeyError
        
        return feat0, feat1


if __name__ == '__main__':
    from einops import rearrange

    d_model = 256  # feature dimension
    nhead=8
    pos_encoding = PositionEncodingSine(d_model=d_model)
    cross_attention = AttentionLayer(d_model=d_model, nhead=nhead, layer_name='cross', attention = 'channel')
    
    feat_c0 = torch.randn(2, 256, 368//8, 496//8)
    feat_c1 = torch.randn(2, 256, 368//8, 496//8)

    b, c, h, w = feat_c0.shape

    feat_c0 = rearrange(pos_encoding(feat_c0), 'n c h w -> n (h w) c')
    feat_c1 = rearrange(pos_encoding(feat_c1), 'n c h w -> n (h w) c')

    feat0, feat1 = cross_attention(feat_c0, feat_c1)

    feat_c0 = rearrange(feat0, 'n (h w) c -> n c h w', h=h, w=w)

    print(feat_c0.shape, feat1.shape)
