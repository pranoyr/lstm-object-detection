import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet101

# layer = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3)




# i = torch.Tensor(1,3,32,33)

# output = layer(i)
# output = torch.relu(output)
# print(output.shape)


resnet = resnet101(pretrained=True)
modules = list(resnet.children())[:-3] 
base_net = nn.Sequential(*modules)

print(base_net)

x = torch.Tensor(1,3,300,300)
x = base_net(x)
print(x.shape)



