import torch
from torchvision.models.vgg import vgg19
import torch.nn as nn



class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = vgg19(pretrained=True).eval()
        self.vgg = nn.Sequential(*list(vgg.features)[:-1])
        self.mse = nn.MSELoss()


    def forward(self, fake, real):
        return self.mse(self.vgg(fake), self.vgg(real))

        