import argparse
import json
import os
import torch

from RevCol_models.revcol import revcol_tiny
from efficientnetv2 import EfficientNetV2
from measure_FLOPs import *
import time

INIT_TIMES = 100
LAT_TIMES = 1000
NUM_CLASSES = 100
SHAPE = [32, 3, 224, 224]

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def measure_latency_in_ms(model, input_shape, is_cuda):
    lat = AverageMeter()
    model.eval()

    x = torch.randn(input_shape)
    if is_cuda:
        model = model.cuda()
        x = x.cuda()
    else:
        model = model.cpu()
        x = x.cpu()

    with torch.no_grad():
        for _ in range(INIT_TIMES):
            output = model(x)

        for _ in range(LAT_TIMES):
            tic = time.time()
            output = model(x)
            toc = time.time()
            lat.update(toc - tic, x.size(0))

    return lat.avg * 1000  # save as ms

def get_lat_ms_flops(model, CUDA=True):
    lat = measure_latency_in_ms(model, input_shape=SHAPE, is_cuda=CUDA)
    flop_shape = SHAPE
    if len(flop_shape) == 4:
        flop_shape[0] = 1
    try:
        flops = calculate_FLOPs_in_M(model, flop_shape)
    except:
        print("FLOPs failed but that is okay")
        flops = -1
    return lat, flops


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    args, unparsed = parser.parse_known_args()

    if args.model_name == 'EffNetV2-b0':
        model = EfficientNetV2('b0',
                               in_channels=3,
                               n_classes=NUM_CLASSES,
                               pretrained=False)
    elif args.model_name == 'EffNetV2-b1':
        model = EfficientNetV2('b1',
                               in_channels=3,
                               n_classes=NUM_CLASSES,
                               pretrained=False)
    elif args.model_name == 'RevCol-T':
        model = revcol_tiny(False, num_classes=NUM_CLASSES)
    elif args.model_name == 'MBNetV2':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=False)

    lat, flops = get_lat_ms_flops(model, CUDA=True)

    with open('results.txt', 'a') as fw:
        fw.write(f'{args.model_name}    Latency: {lat}      FLOPs: {flops}\n')

