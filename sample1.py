import torch.nn as nn
import torch

layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

i = torch.Tensor(32,32,3)

j = torch.Tensor(32,32,3)


print(torch.cat([i,j]).shape)
