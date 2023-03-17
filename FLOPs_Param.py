import torch
import argparse
import sys

sys.path.append('core')
from network import RAFTGMA

from thop import profile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--model_name', help="define model name", default="GMA")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    args = parser.parse_args()

    model = RAFTGMA(args)
    input = torch.randn(1, 3, 368, 496) # 1024*436
    macs, params = profile(model, inputs=(input, input, 12, None, True, True))
    # print(f"macs = {macs/1e9}G")
    print(f"flops = {2*macs/1e9}G")
    print(f"params = {params/1e6}M")