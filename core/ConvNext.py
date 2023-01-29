import torch
import torch.nn as nn
import timm
import numpy as np
from SPP import SPPCSPC
from timm.models.convnext import _create_convnext
from timm.models.registry import register_model

class convnext(nn.Module):
    def __init__(self, output_dim, pretrained=False):
        super().__init__()
        self.cnt = timm.create_model('convnext_pico_ols_tiny', pretrained=pretrained)
        self.SPPm = SPPCSPC(128, output_dim)

        del self.cnt.head
        del self.cnt.stages[-1]

        self.cnt.stem[0] = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=2, padding=1)
        self.cnt.stem[1] = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=1, padding=1)

    # def forward(self, x):
    #     x, feat_size = self.pvt.patch_embed(x)
    #     for stage in self.pvt.stages:
    #         x, feat_size = stage(x, feat_size=feat_size)
    #     return x

    def forward(self, x):
        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.cnt.forward_features(x)
        x = self.SPPm(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x

    def compute_params(self):
        num = 0

        for stage in self.cnt.stages:
            for param in stage.parameters():
                num +=  np.prod(param.size())

        return num

@register_model
def convnext_pico_ols_tiny(pretrained=False, **kwargs):
    # timm nano variant with overlapping 3x3 conv stem
    model_args = dict(
        depths=(2, 2, 6, 2), dims=(32, 64, 128, 256), conv_mlp=True,  stem_type='overlap_tiered', **kwargs)
    model = _create_convnext('convnext_pico_ols_tiny', pretrained=pretrained, **model_args)
    return model

if __name__ == '__main__':
    m = convnext(output_dim=256)
    print(m.compute_params())
    # m = timm.create_model('convnext_pico_ols', pretrained=False)
    print(m)
    input = torch.randn(2, 3, 368, 496)
    out = m(input)
    # out = m.forward_features(input)
    print(out.shape)