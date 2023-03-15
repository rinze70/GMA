import torch
import timm 
from timm.models.layers.halo_attn import HaloAttn
import matplotlib.pyplot as plt
import torch.nn as nn

def visualize_feature_output(feature_output):
    plt.imshow(feature_output[0, 0, :, :].detach().cpu().numpy())
    plt.show()

def alt_visualize(feature_output):
    # plt.imshow(feature_output[0].transpose(0, 2).sum(-1).detach().cpu().numpy())
    plt.imshow(feature_output[0].permute(1, 2, 0).sum(-1).detach().cpu().numpy())
    plt.show()

if __name__ == "__main__":
    fmap = torch.randn(2, 64, 368, 496)
    # visualize_feature_output(fmap)

    # for itr in range(12):
    #     idx = itr//3 * 9
    #     a = fmap[:,idx: idx+9]
    #     print(36-idx-9,36-idx)
    #     # print(a.shape)

    Attn = HaloAttn(3, 64, stride=2)
    out = Attn(fmap)
    print(out.shape)