import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
# from torch_scatter import scatter_softmax, scatter_add


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].expand(batch, -1, -1, -1)


def coords_grid_y_first(batch, ht, wd):
    """Place y grid before x grid"""
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords, dim=0).int()
    return coords[None].expand(batch, -1, -1, -1)


def soft_argmax(corr_me, B, H1, W1):
    # Implement soft argmin
    coords, feats = corr_me.decomposed_coordinates_and_features

    # Computing soft argmin
    flow_pred = torch.zeros(B, 2, H1, W1).to(corr_me.device)
    for batch, (coord, feat) in enumerate(zip(coords, feats)):
        coord_img_1 = coord[:, :2].to(corr_me.device)
        coord_img_2 = coord[:, 2:].to(corr_me.device)
        # relative positions (flow hypotheses)
        rel_pos = (coord_img_2 - coord_img_1)
        # augmented indices
        aug_coord_img_1 = (coord_img_1[:, 0:1] * W1 + coord_img_1[:, 1:2]).long()
        # run softmax on the score
        weight = scatter_softmax(feat, aug_coord_img_1, dim=0)
        rel_pos_weighted = weight * rel_pos
        out = scatter_add(rel_pos_weighted, aug_coord_img_1, dim=0)
        # Need to permute (y, x) to (x, y) for flow
        flow_pred[batch] = out[:, [1,0]].view(H1, W1, 2).permute(2, 0, 1)
    return flow_pred


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow4(flow, mode='bilinear'):
    new_size = (4 * flow.shape[2], 4 * flow.shape[3])
    return 4 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow2(flow, mode='bilinear'):
    new_size = (2 * flow.shape[2], 2 * flow.shape[3])
    return 2 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def downflow8(flow, mode='bilinear'):
    new_size = (flow.shape[2] // 8, flow.shape[3] // 8)
    return F.interpolate(flow, size=new_size, mode=mode, align_corners=True) / 8


def downflow4(flow, mode='bilinear'):
    new_size = (flow.shape[2] // 4, flow.shape[3] // 4)
    return F.interpolate(flow, size=new_size, mode=mode, align_corners=True) / 4


def compute_interpolation_weights(yx_warped):
    # yx_warped: [N, 2]
    y_warped = yx_warped[:, 0]
    x_warped = yx_warped[:, 1]

    # elementwise operations below
    y_f = torch.floor(y_warped)
    y_c = y_f + 1
    x_f = torch.floor(x_warped)
    x_c = x_f + 1

    w0 = (y_c - y_warped) * (x_c - x_warped)
    w1 = (y_warped - y_f) * (x_c - x_warped)
    w2 = (y_c - y_warped) * (x_warped - x_f)
    w3 = (y_warped - y_f) * (x_warped - x_f)

    weights = [w0, w1, w2, w3]
    indices = [torch.stack([y_f, x_f], dim=1), torch.stack([y_c, x_f], dim=1),
               torch.stack([y_f, x_c], dim=1), torch.stack([y_c, x_c], dim=1)]
    weights = torch.cat(weights, dim=1)
    indices = torch.cat(indices, dim=2)
    # indices = torch.cat(indices, dim=0)  # [4*N, 2]

    return weights, indices

# weights, indices = compute_interpolation_weights(xy_warped, b, h_i, w_i)


def compute_inverse_interpolation_img(weights, indices, img, b, h_i, w_i):
    """
    weights: [b, h*w]
    indices: [b, h*w]
    img: [b, h*w, a, b, c, ...]
    """
    w0, w1, w2, w3 = weights
    ff_idx, cf_idx, fc_idx, cc_idx = indices

    k = len(img.size()) - len(w0.size())
    img_0 = w0[(...,) + (None,) * k] * img
    img_1 = w1[(...,) + (None,) * k] * img
    img_2 = w2[(...,) + (None,) * k] * img
    img_3 = w3[(...,) + (None,) * k] * img

    img_out = torch.zeros(b, h_i * w_i, *img.shape[2:]).type_as(img)

    ff_idx = torch.clamp(ff_idx, min=0, max=h_i * w_i - 1)
    cf_idx = torch.clamp(cf_idx, min=0, max=h_i * w_i - 1)
    fc_idx = torch.clamp(fc_idx, min=0, max=h_i * w_i - 1)
    cc_idx = torch.clamp(cc_idx, min=0, max=h_i * w_i - 1)

    img_out.scatter_add_(1, ff_idx[(...,) + (None,) * k].expand_as(img_0), img_0)
    img_out.scatter_add_(1, cf_idx[(...,) + (None,) * k].expand_as(img_1), img_1)
    img_out.scatter_add_(1, fc_idx[(...,) + (None,) * k].expand_as(img_2), img_2)
    img_out.scatter_add_(1, cc_idx[(...,) + (None,) * k].expand_as(img_3), img_3)

    return img_out  # [b, h_i*w_i, ...]

class DeformConv2d(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, bias=None, modulation=False):
        """
        Args:
            modulation (bool, optional): If True, Modulated Defomable Convolution (Deformable ConvNets v2).
        """
        super(DeformConv2d, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.conv = nn.Conv2d(inc, outc, kernel_size=kernel_size, stride=kernel_size, bias=bias)

        self.p_conv = nn.Conv2d(inc, 2*kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
        nn.init.constant_(self.p_conv.weight, 0)
        self.p_conv.register_backward_hook(self._set_lr)

        self.modulation = modulation
        if modulation:
            self.m_conv = nn.Conv2d(inc, kernel_size*kernel_size, kernel_size=3, padding=1, stride=stride)
            nn.init.constant_(self.m_conv.weight, 0)
            self.m_conv.register_backward_hook(self._set_lr)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = (grad_input[i] * 0.1 for i in range(len(grad_input)))
        grad_output = (grad_output[i] * 0.1 for i in range(len(grad_output)))

    def forward(self, x):
        offset = self.p_conv(x)
        if self.modulation:
            m = torch.sigmoid(self.m_conv(x))

        dtype = offset.data.type()
        ks = self.kernel_size
        N = offset.size(1) // 2

        if self.padding:
            x = self.zero_padding(x)

        # (b, 2N, h, w)
        p = self._get_p(offset, dtype)

        # (b, h, w, 2N)
        p = p.contiguous().permute(0, 2, 3, 1)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1

        q_lt = torch.cat([torch.clamp(q_lt[..., :N], 0, x.size(2)-1), torch.clamp(q_lt[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_rb = torch.cat([torch.clamp(q_rb[..., :N], 0, x.size(2)-1), torch.clamp(q_rb[..., N:], 0, x.size(3)-1)], dim=-1).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)

        # clip p
        p = torch.cat([torch.clamp(p[..., :N], 0, x.size(2)-1), torch.clamp(p[..., N:], 0, x.size(3)-1)], dim=-1)

        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_lt[..., N:].type_as(p) - p[..., N:]))
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_rb[..., N:].type_as(p) - p[..., N:]))
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (1 - (q_lb[..., N:].type_as(p) - p[..., N:]))
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (1 + (q_rt[..., N:].type_as(p) - p[..., N:]))

        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)

        # (b, c, h, w, N)
        x_offset = g_lt.unsqueeze(dim=1) * x_q_lt + \
                   g_rb.unsqueeze(dim=1) * x_q_rb + \
                   g_lb.unsqueeze(dim=1) * x_q_lb + \
                   g_rt.unsqueeze(dim=1) * x_q_rt

        # modulation
        if self.modulation:
            m = m.contiguous().permute(0, 2, 3, 1)
            m = m.unsqueeze(dim=1)
            m = torch.cat([m for _ in range(x_offset.size(1))], dim=1)
            x_offset *= m

        x_offset = self._reshape_x_offset(x_offset, ks)
        out = self.conv(x_offset)

        return out

    def _get_p_n(self, N, dtype):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1),
            torch.arange(-(self.kernel_size-1)//2, (self.kernel_size-1)//2+1))
        # (2N, 1)
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2*N, 1, 1).type(dtype)

        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h*self.stride+1, self.stride),
            torch.arange(1, w*self.stride+1, self.stride))
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)

        return p_0

    def _get_p(self, offset, dtype):
        N, h, w = offset.size(1)//2, offset.size(2), offset.size(3)

        # (1, 2N, 1, 1)
        p_n = self._get_p_n(N, dtype)
        # (1, 2N, h, w)
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + p_n + offset
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        # (b, c, h*w)
        x = x.contiguous().view(b, c, -1)

        # (b, h, w, N)
        index = q[..., :N]*padded_w + q[..., N:]  # offset_x*w + offset_y
        # (b, c, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1).contiguous().view(b, c, -1)

        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)

        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, ks):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s+ks].contiguous().view(b, c, h, w*ks) for s in range(0, N, ks)], dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h*ks, w*ks)

        return x_offset
    
if __name__ == '__main__':
    x = torch.randn(1, 3, 62, 94)
    Dconv = DeformConv2d(3, 64, modulation=True)
    out = Dconv(x)
    print(out.size())