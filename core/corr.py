import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.utils import bilinear_sampler, coords_grid
# from compute_sparse_correlation import compute_sparse_corr, compute_sparse_corr_torch, compute_sparse_corr_mink

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())


class CorrBlockSingleScale(nn.Module):
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        super().__init__()
        self.radius = radius

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        batch, h1, w1, dim, h2, w2 = corr.shape
        self.corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        corr = self.corr
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        out = corr.view(batch, h1, w1, -1)
        out = out.permute(0, 3, 1, 2).contiguous().float()
        return out

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

class MutiHeadCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = MutiHeadCorrBlock.corr(fmap1, fmap2, num_levels)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
        corr = list(corr.chunk(num_levels, dim=1))

        self.corr_pyramid.append(corr.pop(0))
        for i in range(1, self.num_levels):
            # cor = corr[:,i].view(batch * h1 * w1, 1, h2, w2)
            self.corr_pyramid.append(F.avg_pool2d(corr.pop(0), 2**i, stride=2**i))

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2, hd=4):
        batch, dim, ht, wd = fmap1.shape
        assert dim % hd == 0
        fmap1 = fmap1.view(batch, hd, dim//hd, ht * wd)
        fmap2 = fmap2.view(batch, hd, dim//hd, ht * wd)

        corr = torch.matmul(fmap1.transpose(2, 3), fmap2).permute(0, 2, 1, 3)
        corr = corr.view(batch, ht, wd, hd, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim//hd).float())

if __name__ == "__main__":
    fmap1 = torch.randn(1, 256, 368//8, 496//8)
    fmap2 = torch.randn(1, 256, 368//8, 496//8)
    # MHCB = MutiHeadCorrBlock(fmap1, fmap2, num_levels=4, radius=4)
    fmap3 = torch.randn(1, 256, 4, 368//8, 496//8)
    outs = fmap3.chunk(4, dim=2)
    print(outs.shape)
    # CB = CorrBlock(fmap1, fmap2, num_levels=4, radius=4)
    # print(corr.shape)