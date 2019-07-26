import torch
from torch.nn import Conv2d, Sequential, ModuleList, ReLU, BatchNorm2d
from ..nn.vgg import vgg

from ..ssd import SSD
from ..ssd.predictor import Predictor
from ..ssd.config import vgg_ssd_config as config


vgg_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512]
base_net = ModuleList(vgg(vgg_config))

print(base_net)

