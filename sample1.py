import torch.nn as nn
import torch

layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)

i = torch.Tensor(32, 3, 28, 28)

i = i.to(torch.device('cpu'))

out = layer(i)
print(out.shape)