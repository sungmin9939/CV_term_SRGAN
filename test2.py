import torch
import torch.nn as nn

input = torch.rand(1,3,64,64)
conv = nn.Conv2d(3, 64, 3, 2)
output = conv(input)
print(output.shape)