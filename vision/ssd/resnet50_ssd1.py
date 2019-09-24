import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU
from ..nn.mobilenet import MobileNetV1
from .ssd import SSD
from .predictor import Predictor
from .config import vgg_ssd_config as config
import torchvision.models as models
from ..nn.resnet1 import resnet18
import torch.nn as nn


def create_resnet18_ssd(num_classes, is_test=False):

    resnet = resnet18(pretrained=True)
    modules = list(resnet.children())[:-2]      # delete the last fc layer.

    # pool5 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
    # conv6 = nn.Conv2d(512, 1024, kernel_size=1, padding=0, dilation=6)
    # # # conv7 = nn.Conv2d(1024, 1024, kernel_size=1)

    # # modules += [conv6, nn.ReLU()]
    # modules += [pool5, conv6, nn.ReLU()]
                
    base_net = nn.Sequential(*modules)

    source_layer_indexes = [
        6,
        7,
        8
    ]
        
    extras = ModuleList([
            Sequential(
                Conv2d(in_channels=512, out_channels=256, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=512, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
                ReLU()
            ),
            Sequential(
                Conv2d(in_channels=256, out_channels=128, kernel_size=1),
                ReLU(),
                Conv2d(in_channels=128, out_channels=256, kernel_size=3),
                ReLU()
            )
        ])

    regression_headers = ModuleList([
        Conv2d(in_channels=128, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=128, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1),
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1), # TODO: change to kernel_size=1, padding=0?
    ])


    return SSD(num_classes, base_net, source_layer_indexes,
               extras, classification_headers, regression_headers, is_test=is_test, config=config)


def create_resnet18_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=None):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor

