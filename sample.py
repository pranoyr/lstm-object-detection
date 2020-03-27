import torch.nn as nn
import torch

x = torch.Tensor(1,3,32,32)

with torch.no_grad():
    layer = nn.Conv2d(3,3,5,1,1)
    x = layer(x)

print(x)