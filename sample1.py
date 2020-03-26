import torch.nn as nn
import torch

layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

i = torch.Tensor(10, 3, 28, 28)

j = torch.Tensor(10, 3, 28, 28)


print(torch.stack([i,j]).shape)
