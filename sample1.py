import torch.nn as nn
import torch
import torch.nn.functional as F

layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)




i = torch.Tensor(1,3,32,33)

output = layer(i)
output = torch.relu(output)
print(output.shape)



